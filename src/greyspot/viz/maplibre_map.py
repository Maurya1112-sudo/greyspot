"""MapLibre GL JS risk map - the upgrade from the earlier Folium/Leaflet
prototype (`viz/map_demo.py`, kept for the quick sanity-check use case).

This is the mapping library the project dossier's architecture actually
specifies (Section 7: "MapLibre GL JS ... interactive vector maps without
locking the project to a proprietary renderer"), so this brings the visual
layer a step closer to the eventual product rather than staying a
throwaway demo. Concrete upgrades over the Folium version:
  - GPU-accelerated vector rendering (smoother pan/zoom on 7,500+ segments)
  - Data-driven styling (`interpolate` on the score property) instead of
    per-feature Python-side colour computation
  - Native collision clustering instead of a fixed random sample
  - A model/data provenance panel (dossier Section 6's auditability
    requirement: "every exported number traceable to ... generation
    timestamp")
  - A confidence/coverage legend note, not just a risk-colour legend
    (dossier Section 10: risk and confidence should never be conflated)

Base tiles: **OpenFreeMap** (`tiles.openfreemap.org`, "positron" style) -
free forever, no API key, no rate limit, no registration
(https://openfreemap.org/quick_start/). Originally this used CARTO's free
"Positron" raster tiles; CARTO's anonymous tile endpoint started returning
a literal "API KEY REQUIRED" watermark baked into every tile image some
time before 2026-08-31 (verified by fetching a tile directly - the PNG
itself carries the watermark, it isn't a referrer/CORS issue) - see
`docs/decision_log.md`'s "CARTO basemap regression" entry. OpenFreeMap
serves real vector tiles (crisper at every zoom than the old raster tiles,
and it renders labels/place names the old raw raster style never had)
rather than a raw tile URL template, so the whole style is referenced by
URL instead of hand-built. Basemap tiles are still fetched live over the
network (an offline basemap would mean shipping a tileset), so the map
needs internet access for the background map image, but not for MapLibre
itself or for the road/collision data.

**MapLibre GL JS itself is bundled locally, not loaded from a CDN**
(`vendor/maplibre-gl.js` + `.css`, downloaded 2026-08-31, BSD-3-Clause
licensed - see the file header). The first version of this map referenced
`unpkg.com` directly and failed silently in at least one real viewer
(`Uncaught ReferenceError: maplibregl is not defined` - some browsers/
webviews block cross-origin `<script src>` loads from local/`file://`
pages even though the same origin is reachable via `fetch`). Inlining the
~800KB library removes that entire failure class - the map now has zero
external script dependency.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd

_VENDOR_DIR = Path(__file__).parent / "vendor"


def _read_vendor_asset(filename: str) -> str:
    path = _VENDOR_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it with, e.g.:\n"
            f"  curl -o {path} https://unpkg.com/maplibre-gl@4.7.1/dist/{filename}\n"
            "(see this module's docstring for why it must be bundled, not CDN-loaded)."
        )
    return path.read_text(encoding="utf-8")


_STYLE_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>Greyspot — {borough_name}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<script>{maplibre_js}</script>
<style>{maplibre_css}</style>
<style>
  /* ── Design tokens ─────────────────────────────────── */
  /* Dark panels on a light Positron basemap: high contrast, instrument-panel
     feel. Avoids the generic white-card-on-light-map that every auto-generated
     risk map uses. Risk palette shifted from traffic-light (#00703c / #f47738 /
     #ca3535) to deliberate jewel tones: deep forest → warm amber → brick red.
     No-data segments use a cool mid-grey (#9EA5AE) that reads as absent-data,
     not as low-risk. */
  :root {{
    --panel:      #1A1D24;
    --panel-edge: #272C36;
    --t1:         #E8EBF0;
    --t2:         #8891A0;
    --t3:         #4A5162;
    --accent:     #5B8DD9;
    --risk-lo:    #1B6B4A;
    --risk-mi:    #C4882A;
    --risk-hi:    #A83228;
    --no-data:    #9EA5AE;
  }}

  html, body {{
    margin: 0; padding: 0; height: 100%;
    font-family: 'Space Grotesk', system-ui, sans-serif;
    font-size: 13px; color: var(--t1);
  }}
  #map {{ position: absolute; top: 0; bottom: 0; width: 100%; }}

  /* ── Shared panel chrome ───────────────────────────── */
  .panel {{
    position: absolute; z-index: 1;
    background: var(--panel);
    border: 1px solid var(--panel-edge);
    border-radius: 3px;
  }}

  /* ── Info panel ────────────────────────────────────── */
  #info-panel {{ top: 16px; left: 16px; width: 248px; padding: 14px 16px; }}

  #info-panel .eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; font-weight: 500;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--accent); margin-bottom: 5px;
  }}
  #info-panel h1 {{
    font-size: 17px; font-weight: 600; line-height: 1.2;
    margin: 0 0 12px 0; color: var(--t1);
  }}
  .meta-row {{
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 5px 0; border-top: 1px solid var(--panel-edge);
    gap: 8px;
  }}
  .meta-label {{ color: var(--t2); font-size: 12px; flex-shrink: 0; }}
  .meta-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: var(--t1); text-align: right;
    word-break: break-all;
  }}
  #info-panel .caveat {{
    margin-top: 10px; padding-top: 9px;
    border-top: 1px solid var(--panel-edge);
    font-size: 10.5px; color: var(--t3); line-height: 1.55;
  }}

  /* ── Legend ─────────────────────────────────────────── */
  #legend {{
    bottom: 28px; left: 50%; transform: translateX(-50%);
    padding: 9px 16px 7px; white-space: nowrap;
  }}
  .legend-title {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; font-weight: 500;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--t2); text-align: center; margin-bottom: 7px;
  }}
  .legend-bar {{
    height: 7px; width: 200px; border-radius: 2px;
    background: linear-gradient(to right, var(--risk-lo), var(--risk-mi), var(--risk-hi));
    margin: 0 auto;
  }}
  .legend-ticks {{
    display: flex; justify-content: space-between;
    width: 200px; margin: 4px auto 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: var(--t2);
  }}
  .legend-note {{
    text-align: center; margin-top: 5px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: var(--t3);
  }}

  /* ── Layer toggles ──────────────────────────────────── */
  #layer-toggle {{
    top: 16px; right: 16px;
    padding: 6px 7px; display: flex; flex-direction: column; gap: 3px;
  }}
  .toggle-btn {{
    display: flex; align-items: center; gap: 8px;
    padding: 5px 9px; border-radius: 2px; cursor: pointer;
    border: 1px solid transparent; background: transparent;
    color: var(--t1); font-family: 'Space Grotesk', sans-serif;
    font-size: 12px; font-weight: 500;
    transition: background 0.12s, border-color 0.12s;
  }}
  .toggle-btn.active {{
    background: rgba(91,141,217,0.1);
    border-color: rgba(91,141,217,0.25);
  }}
  .toggle-btn:hover {{ background: rgba(255,255,255,0.05); }}
  .toggle-dot {{ width: 9px; height: 9px; border-radius: 2px; flex-shrink: 0; }}
  .dot-roads {{
    background: linear-gradient(135deg, var(--risk-lo) 40%, var(--risk-hi) 100%);
  }}
  .dot-col {{ background: #C51B8A; }}

  /* ── Popups ─────────────────────────────────────────── */
  .maplibregl-popup-content {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 12px; padding: 10px 12px;
    background: var(--panel); color: var(--t1);
    border: 1px solid var(--panel-edge);
    border-radius: 3px; box-shadow: none;
  }}
  .maplibregl-popup-anchor-bottom .maplibregl-popup-tip {{
    border-top-color: var(--panel-edge);
  }}
  .maplibregl-popup-close-button {{ color: var(--t2); font-size: 16px; }}
  .popup-type {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: var(--t2); margin-bottom: 3px;
  }}
  .popup-score {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 15px; font-weight: 500; color: var(--t1); margin-bottom: 5px;
  }}
  .popup-caveat {{ font-size: 10.5px; color: var(--t3); }}

  /* ── MapLibre nav control ──────────────────────────── */
  .maplibregl-ctrl-group {{
    background: var(--panel) !important;
    border: 1px solid var(--panel-edge) !important;
    border-radius: 3px !important; box-shadow: none !important;
  }}
  .maplibregl-ctrl-group button {{ width: 30px !important; height: 30px !important; }}
</style>
</head>
<body>
<div id="map"></div>

<div id="info-panel" class="panel">
  <div class="eyebrow">Greyspot</div>
  <h1>{borough_name}</h1>
  <div class="meta-row">
    <span class="meta-label">Model</span>
    <span class="meta-value">{model_version}</span>
  </div>
  <div class="meta-row">
    <span class="meta-label">Test year</span>
    <span class="meta-value">{test_year}</span>
  </div>
  <div class="meta-row">
    <span class="meta-label">Generated</span>
    <span class="meta-value">{generated_at}</span>
  </div>
  <div class="caveat">Research signal only — relative risk, not an absolute
    prediction or safety guarantee. See docs/methodology.md</div>
</div>

<div id="legend" class="panel">
  <div class="legend-title">Predicted relative risk</div>
  <div class="legend-bar"></div>
  <div class="legend-ticks"><span>lower</span><span>higher</span></div>
  <div class="legend-note">grey = no recent estimate (not "safe")</div>
</div>

<div id="layer-toggle" class="panel">
  <button class="toggle-btn active" id="btn-roads"
          onclick="toggleLayers(this, ['roads-layer'])">
    <span class="toggle-dot dot-roads"></span>Road risk
  </button>
  <button class="toggle-btn active" id="btn-col"
          onclick="toggleLayers(this, ['collision-points','collision-clusters','collision-cluster-count'])">
    <span class="toggle-dot dot-col"></span>Collisions
  </button>
</div>

<script>
const roadsData = {roads_geojson};
const collisionsData = {collisions_geojson};

const map = new maplibregl.Map({{
  container: 'map',
  style: 'https://tiles.openfreemap.org/styles/positron',
  center: {center},
  zoom: {zoom}
}});

map.addControl(new maplibregl.NavigationControl(), 'top-right');

map.on('load', () => {{
  // ── Road risk layer ───────────────────────────────── //
  map.addSource('roads', {{ type: 'geojson', data: roadsData }});
  map.addLayer({{
    id: 'roads-layer',
    type: 'line',
    source: 'roads',
    paint: {{
      'line-width': ['interpolate', ['linear'], ['zoom'], 11, 2, 14, 3.5, 17, 7],
      'line-color': [
        'case',
        ['==', ['get', 'has_score'], 0], '#9EA5AE',
        ['interpolate', ['linear'], ['get', 'score'],
          0, '#1B6B4A', {score_mid}, '#C4882A', {score_max}, '#A83228']
      ],
      'line-opacity': 0.9
    }}
  }});

  // ── Collision layers ──────────────────────────────── //
  map.addSource('collisions', {{
    type: 'geojson', data: collisionsData,
    cluster: true, clusterRadius: 40, clusterMaxZoom: 15
  }});
  map.addLayer({{
    id: 'collision-clusters', type: 'circle',
    source: 'collisions', filter: ['has', 'point_count'],
    paint: {{
      'circle-color': '#1A1D24',
      'circle-stroke-width': 1.5, 'circle-stroke-color': '#4A5162',
      'circle-opacity': 0.92,
      'circle-radius': ['step', ['get', 'point_count'], 10, 10, 14, 50, 18, 200, 24]
    }}
  }});
  map.addLayer({{
    id: 'collision-cluster-count', type: 'symbol',
    source: 'collisions', filter: ['has', 'point_count'],
    layout: {{ 'text-field': ['get', 'point_count_abbreviated'], 'text-size': 11 }},
    paint: {{ 'text-color': '#8891A0' }}
  }});
  map.addLayer({{
    id: 'collision-points', type: 'circle',
    source: 'collisions', filter: ['!', ['has', 'point_count']],
    paint: {{
      'circle-color': ['match', ['get', 'severity'],
        'fatal', '#E63946', 'serious', '#C51B8A', '#F768A1'],
      'circle-radius': 4, 'circle-opacity': 0.88,
      'circle-stroke-width': 1, 'circle-stroke-color': 'rgba(255,255,255,0.15)'
    }}
  }});

  // ── Interactivity ─────────────────────────────────── //
  map.on('click', 'roads-layer', (e) => {{
    const p = e.features[0].properties;
    new maplibregl.Popup()
      .setLngLat(e.lngLat)
      .setHTML(
        '<div class="popup-type">Segment ' + p.segment_id +
          ' &middot; ' + (p.highway || 'unknown') + '</div>' +
        '<div class="popup-score">' +
          (p.has_score
            ? 'Risk ' + Number(p.score).toFixed(3)
            : 'No recent estimate') +
        '</div>' +
        '<div class="popup-caveat">Relative risk signal — not a safety guarantee.</div>'
      ).addTo(map);
  }});
  map.on('click', 'collision-points', (e) => {{
    const p = e.features[0].properties;
    new maplibregl.Popup()
      .setLngLat(e.lngLat)
      .setHTML(
        '<div class="popup-type">Year ' + (p.year || 'n/a') + '</div>' +
        '<div class="popup-score">' + (p.severity || 'collision') + '</div>'
      ).addTo(map);
  }});

  ['roads-layer', 'collision-points', 'collision-clusters'].forEach(id => {{
    map.on('mouseenter', id, () => map.getCanvas().style.cursor = 'pointer');
    map.on('mouseleave', id, () => map.getCanvas().style.cursor = '');
  }});
}});

function toggleLayers(btn, ids) {{
  const on = btn.classList.toggle('active');
  ids.forEach(id => map.setLayoutProperty(id, 'visibility', on ? 'visible' : 'none'));
}}
</script>
</body>
</html>
"""


def build_maplibre_map(
    edges: gpd.GeoDataFrame,
    segment_scores: pd.DataFrame,
    collisions: pd.DataFrame,
    borough_name: str,
    model_version: str,
    test_year: int,
    out_path: Path,
    score_col: str = "score",
    lon_col: str = "longitude",
    lat_col: str = "latitude",
) -> Path:
    """Render the MapLibre risk map to a single self-contained HTML file."""
    merged = edges.merge(segment_scores[["segment_id", score_col]], on="segment_id", how="left")
    merged["has_score"] = merged[score_col].notna().astype(int)
    merged[score_col] = merged[score_col].fillna(0.0)
    score_max = max(float(merged[score_col].max()), 1e-6)
    score_mid = score_max / 2

    road_cols = ["segment_id", "highway", score_col, "has_score", "geometry"]
    road_cols = [c for c in road_cols if c in merged.columns]
    roads_geojson = merged[road_cols].rename(columns={score_col: "score"}).to_json()

    pts = collisions.dropna(subset=[lon_col, lat_col]).copy()
    collisions_gdf = gpd.GeoDataFrame(
        pts[["severity_label", "collision_year"]].rename(
            columns={"severity_label": "severity", "collision_year": "year"}
        ),
        geometry=gpd.points_from_xy(pts[lon_col], pts[lat_col]),
        crs="EPSG:4326",
    )
    collisions_geojson = collisions_gdf.to_json()

    centroid = merged.geometry.union_all().centroid

    # Defensive: a literal "</script" anywhere in injected content (an odd
    # road name, a future library update) would prematurely close the
    # surrounding <script> tag. None of today's inputs contain it, but the
    # escape is free and this is standard practice for embedding arbitrary
    # text inside <script> - "<\/script" is valid JS/JSON and un-splits it.
    def _script_safe(s: str) -> str:
        return s.replace("</script", "<\\/script")

    html = _STYLE_TEMPLATE.format(
        maplibre_js=_script_safe(_read_vendor_asset("maplibre-gl.js")),
        maplibre_css=_read_vendor_asset("maplibre-gl.css"),
        borough_name=borough_name,
        model_version=model_version,
        test_year=test_year,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        roads_geojson=_script_safe(roads_geojson),
        collisions_geojson=_script_safe(collisions_geojson),
        center=json.dumps([centroid.x, centroid.y]),
        zoom=13,
        score_mid=f"{score_mid:.4f}",
        score_max=f"{score_max:.4f}",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
