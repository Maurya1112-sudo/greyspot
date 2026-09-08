import { useEffect, useRef, useState } from "react";
// maplibre-gl v6 ships only named exports (no default export) - `import
// maplibregl from "maplibre-gl"` throws "does not provide an export named
// 'default'" under Vite's ESM handling. `import *` gives back the same
// maplibregl.Map / maplibregl.NavigationControl usage pattern without
// rewriting every call site.
import * as maplibregl from "maplibre-gl";
import type { Map as MapLibreMap, MapGeoJSONFeature, MapLayerMouseEvent } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
// The actual root cause of the long-running "blank map" bug (2026-09-01):
// maplibre-gl spawns a Web Worker to parse vector tiles off the main
// thread, and under Vite it requests that worker from a URL Vite never
// generates a matching file for - a 404 that silently stalls the entire
// tile pipeline forever (style/sprite/TileJSON all load fine over the
// main thread and gave every earlier surface-level check a false pass;
// only actually inspecting `map.loaded()` and Resource Timing for a real
// `.pbf` tile request exposed it - no `.pbf` request was ever attempted).
// This is MapLibre's own documented Vite fix
// (https://maplibre.org/maplibre-gl-js/docs/) - `?worker&url` routes the
// worker file through Vite's own worker bundler instead of serving it
// verbatim (which is missing its sibling maplibre-gl-shared.mjs chunk).
import maplibreWorkerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";

maplibregl.setWorkerUrl(maplibreWorkerUrl);

interface MapViewProps {
  roadsGeoJSON: GeoJSON.FeatureCollection | null;
  selectedSegmentId: string | null;
  onSelectSegment: (segmentId: string) => void;
  loading?: boolean;
}

// OpenFreeMap's "liberty" vector style: free forever, no API key, no rate
// limit (https://openfreemap.org/quick_start/). Switched from "positron"
// 2026-09-01 on direct user feedback ("this maps looks so boring...make
// map like more google maps like, roads name and all the stuff") -
// positron is a deliberately minimal, near-monochrome basemap (the same
// family as CARTO's old Positron raster tiles this project started with);
// liberty is OpenFreeMap's flagship, most detailed style - full road-type
// colour hierarchy with casings, POI icons, transit lines, one-way arrows,
// and (see below) its own built-in 3D buildings layer - much closer to
// the Google Maps-like density and colour the user asked for, while still
// being the same free, key-less, no-rate-limit OpenFreeMap infrastructure
// as before (https://openfreemap.org - both styles are served from the
// exact same "openmaptiles" vector tiles, just styled differently).
const BASEMAP_STYLE = "https://tiles.openfreemap.org/styles/liberty";

// The style's own layer stack (fetched and inspected 2026-09-01) ends in
// this order: background/park/water/building fills -> road lines -> a
// block of label/shield symbol layers starting at "waterway_line_label"
// -> place-name labels. `map.addLayer()` with no `beforeId` appends to the
// very end of that stack - which is exactly how the risk overlay ended up
// painted on top of every basemap label, including street names,
// obscuring them (the user's reported bug). Inserting `beforeId` here
// puts the overlay above every fill/line layer but below every label, the
// correct visual order for a data overlay on a labelled basemap.
const FIRST_LABEL_LAYER_ID = "waterway_line_label";

// Reads a colour from App.css's actual CSS custom properties rather than
// duplicating a hex literal here - MapLibre's paint-expression format
// can't reference var(--...) directly (found by /impeccable audit,
// 2026-09-01: the map's RAG colours were hardcoded independently of the
// CSS tokens, a silent-drift risk if the tokens are ever revised without
// updating this file too). The literal fallback only fires if the
// property is somehow missing (e.g. App.css failed to load) - it should
// never differ from App.css's own value in normal operation.
function cssColor(varName: string, fallback: string): string {
  if (typeof document === "undefined") return fallback;
  const value = getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
  return value || fallback;
}

function ragLabel(score: number | undefined, hasScore: boolean): { label: string; className: string } {
  if (!hasScore || score === undefined) return { label: "No estimate", className: "govuk-tag--grey" };
  if (score < 33) return { label: "Lower", className: "govuk-tag--green" };
  if (score < 66) return { label: "Medium", className: "govuk-tag--orange" };
  return { label: "Higher", className: "govuk-tag--red" };
}

interface HoverInfo {
  x: number;
  y: number;
  name: string;
  highway: string;
  score?: number;
  hasScore: boolean;
}

interface SelectedPoint {
  name: string;
  lng: number;
  lat: number;
}

export function MapView({ roadsGeoJSON, selectedSegmentId, onSelectSegment, loading }: MapViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const loadedRef = useRef(false);
  const glowFlashTimerRef = useRef<number | undefined>(undefined);
  const [hoverInfo, setHoverInfo] = useState<HoverInfo | null>(null);
  const [selectedPoint, setSelectedPoint] = useState<SelectedPoint | null>(null);
  const [buildings3D, setBuildings3D] = useState(false);

  // Create the map once.
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: BASEMAP_STYLE,
      center: [-0.14, 51.5],
      zoom: 12.5,
      // Lets the built-in NavigationControl's drag-to-rotate gesture also
      // pitch the map (right-click-drag / two-finger drag) - needed for
      // the 3D buildings toggle below to actually read as 3D rather than
      // a flat footprint viewed from directly above.
      pitchWithRotate: true,
    });
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
    mapRef.current = map;

    map.on("load", () => {
      loadedRef.current = true;
      map.addSource("roads", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      map.addLayer(
        {
          id: "roads-layer",
          type: "line",
          source: "roads",
          paint: {
            // Zoom-responsive width, thickened from the original 1.5-5px
            // range - at a whole-borough zoom the risk overlay is the point
            // of the map and needs to read clearly against the basemap, not
            // disappear into it (see docs/decision_log.md, 2026-08-31 UI
            // redesign entry).
            "line-width": ["interpolate", ["linear"], ["zoom"], 11, 2, 14, 3.5, 17, 7],
            // RAG (Red-Amber-Green) scale - the standard UK civil-service
            // convention for a priority/risk rating, more contextually
            // legible to this product's government-analyst audience than an
            // arbitrary blue-yellow-red diverging scale. Read from App.css's
            // own CSS custom properties (see cssColor() above) so this
            // never drifts from the queue/evidence-panel's colours - these
            // are the graphic (line) tokens, not the darkened `-text`
            // variants (those are for text contrast only, see DESIGN.md).
            "line-color": [
              "case",
              ["==", ["get", "has_score"], 0],
              cssColor("--risk-none-hex", "#82878c"),
              [
                "interpolate", ["linear"], ["get", "priority_score"],
                0, cssColor("--risk-low-hex", "#067e3f"),
                50, cssColor("--risk-mid-hex", "#dd7b2b"),
                100, cssColor("--risk-high-hex", "#c92f33"),
              ],
            ],
            "line-opacity": ["case", ["==", ["get", "segment_id"], ["literal", ""]], 1, 0.9],
          },
        },
        FIRST_LABEL_LAYER_ID,
      );
      // A thin road is genuinely hard to click - the visible line is only
      // 2-7px wide (line-width above), a small fraction of a typical
      // pointer's own margin of error, especially on a touchpad or at a
      // whole-borough zoom. This invisible, much wider line shares the
      // exact same source/geometry purely as a bigger hit target - the
      // standard technique for thin-line click targets (the same idea
      // behind e.g. mapbox-gl-draw's own hit-area layers). Hover/click
      // listeners below bind to THIS layer, not the thin visible one;
      // MapLibre still hit-tests a layer's rendered geometry even at
      // opacity 0, since hit-testing works off the paint-computed
      // geometry, not final pixel alpha.
      map.addLayer(
        {
          id: "roads-hitbox",
          type: "line",
          source: "roads",
          paint: {
            "line-width": ["interpolate", ["linear"], ["zoom"], 11, 14, 14, 20, 17, 28],
            "line-opacity": 0,
          },
        },
        FIRST_LABEL_LAYER_ID,
      );

      // Selected-segment highlight: 2026-09-08 redesign after live
      // feedback that a single flat white outline all but disappeared
      // against the basemap's own light greys/creams and white label
      // halos (a screenshot showed it blending into "Tottenham Court
      // Road Station" 's own light background). A single colour can
      // never guarantee contrast against an unpredictable basemap
      // underneath it - the standard cartographic fix is a dual-stroke
      // (a dark casing so the highlight reads against ANY light
      // background, with a bright core on top so it reads against the
      // dark casing itself) plus a soft outer glow, the same "selected"
      // language mapping tools from Google Maps to QGIS use, layered
      // bottom (glow) to top (core) in this order:
      map.addLayer(
        {
          id: "roads-selected-glow",
          type: "line",
          source: "roads",
          filter: ["==", ["get", "segment_id"], ""],
          paint: {
            "line-width": ["interpolate", ["linear"], ["zoom"], 11, 10, 14, 16, 17, 24],
            "line-color": cssColor("--accent-hex", "#2e8a8b"),
            "line-blur": 6,
            "line-opacity": 0.5,
            "line-opacity-transition": { duration: 300, delay: 0 },
          },
        },
        FIRST_LABEL_LAYER_ID,
      );
      map.addLayer(
        {
          id: "roads-selected-casing",
          type: "line",
          source: "roads",
          filter: ["==", ["get", "segment_id"], ""],
          paint: {
            "line-width": ["interpolate", ["linear"], ["zoom"], 11, 5, 14, 8, 17, 13],
            "line-color": cssColor("--selection-casing-hex", "#0e141c"),
          },
        },
        FIRST_LABEL_LAYER_ID,
      );
      map.addLayer(
        {
          id: "roads-selected-outline",
          type: "line",
          source: "roads",
          filter: ["==", ["get", "segment_id"], ""],
          paint: {
            "line-width": ["interpolate", ["linear"], ["zoom"], 11, 2, 14, 3.5, 17, 6],
            "line-color": cssColor("--selection-outline-hex", "#fdfbf9"),
            "line-opacity": 1,
          },
        },
        FIRST_LABEL_LAYER_ID,
      );

      // 3D buildings - the "liberty" style already ships its own
      // "building-3d" fill-extrusion layer (verified 2026-09-01 by
      // fetching the style JSON directly), visible by default past zoom
      // 14 with a flat hsl(35,8%,85%) colour. Two changes here: (1) start
      // it hidden, so the toggle button below is a deliberate reveal
      // rather than an automatic one the user never asked for, and (2) a
      // richer paint job for real visual depth ("make 3d models more
      // graphics" - a height-based colour ramp from a warm low-rise tone
      // to a cooler tall-building tone, plus the vertical-gradient shading
      // MapLibre's fill-extrusion spec supports but the base style leaves
      // at its default) - reusing the style's own working layer rather
      // than layering a second, competing extrusion layer on the same
      // source (which would z-fight and double-render).
      // Street-name labels were still effectively invisible after the
      // 2026-09-01 z-ordering fix above (which only stopped the risk
      // overlay from painting over labels - it never affected whether a
      // label draws at all). Inspecting the "liberty" style JSON directly
      // shows why: "highway-name-minor" (ordinary residential streets -
      // the vast majority of this project's segments, which score as
      // "unclassified road" in the priority queue) has minzoom 15, and
      // "highway-name-path" has minzoom 15.5, while a whole-borough
      // fitBounds view (see the roadsGeoJSON effect below) lands well
      // below that, around zoom 12-13. Only "highway-name-major"
      // (minzoom 12.2 - trunk/primary/secondary roads, a small minority)
      // was ever visible at the default view - reading as "no street
      // names" for a borough dominated by residential segments. Lowering
      // these two thresholds to match major roads' 12.2 makes the
      // labelling correct for this product's actual data, not a style
      // fixed up for a generic all-road-types web map.
      if (map.getLayer("highway-name-minor")) {
        map.setLayerZoomRange("highway-name-minor", 12.2, 24);
        // The vendor style only sets text-halo-blur on this layer, with no
        // text-halo-color - MapLibre's default halo colour is fully
        // transparent, so the #666 grey text got no actual contrast
        // boost, just a soft blur. A solid light halo is what makes the
        // text read on both parkland green and building beige.
        map.setPaintProperty("highway-name-minor", "text-halo-color", "#f8f4f0");
        map.setPaintProperty("highway-name-minor", "text-halo-width", 1.4);
      }
      if (map.getLayer("highway-name-path")) {
        map.setLayerZoomRange("highway-name-path", 13, 24);
      }

      if (map.getLayer("building-3d")) {
        map.setLayoutProperty("building-3d", "visibility", "none");
        map.setPaintProperty("building-3d", "fill-extrusion-height", ["coalesce", ["get", "render_height"], 6]);
        map.setPaintProperty("building-3d", "fill-extrusion-base", ["coalesce", ["get", "render_min_height"], 0]);
        map.setPaintProperty("building-3d", "fill-extrusion-vertical-gradient", true);
        map.setPaintProperty("building-3d", "fill-extrusion-opacity", 0.92);
        // Ramp re-tinted 2026-09-01 for the Fieldscope redesign: low-rise
        // buildings pick up the paper zone's warm parchment tone, tall
        // buildings deepen toward the chassis zone's cool graphite - the
        // skyline itself echoes the product's own paper/chassis material
        // split rather than an unrelated colour choice.
        map.setPaintProperty("building-3d", "fill-extrusion-color", [
          "interpolate", ["linear"], ["coalesce", ["get", "render_height"], 6],
          0, "#ede9e3",
          20, "#c7beac",
          60, "#5c6773",
          150, "#1a2129",
        ]);
      }

      // Hover is deliberately read-only (name/type/score, no links or
      // buttons inside it) after 2026-09-01 feedback: a Street View link
      // living *inside* a card that tracks the cursor is a moving target -
      // by the time the pointer travels from the road (where hover fires)
      // to the link a few pixels away, it has usually already left the
      // road feature's hit area, the card vanishes, and the link was
      // never clickable. Street View now lives in a fixed-position panel
      // (`selectedPoint`, set on click below) that never moves - trivially
      // clickable, at the cost of needing a click first.
      // Bound to "roads-hitbox" (the wide invisible layer above), not the
      // thin visible "roads-layer" - see that layer's own comment for why.
      map.on("mousemove", "roads-hitbox", (e: MapLayerMouseEvent) => {
        const feature = e.features?.[0] as MapGeoJSONFeature | undefined;
        if (!feature) return;
        map.getCanvas().style.cursor = "pointer";
        const props = feature.properties ?? {};
        setHoverInfo({
          x: e.point.x,
          y: e.point.y,
          name: (props.name as string) || "Unnamed road",
          highway: (props.highway as string) || "unclassified",
          score: props.has_score ? (props.priority_score as number) : undefined,
          hasScore: Boolean(props.has_score),
        });
      });
      map.on("mouseleave", "roads-hitbox", () => {
        map.getCanvas().style.cursor = "";
        setHoverInfo(null);
      });
      map.on("click", "roads-hitbox", (e) => {
        const feature = e.features?.[0] as MapGeoJSONFeature | undefined;
        const props = feature?.properties ?? {};
        const segmentId = props.segment_id as string | undefined;
        if (segmentId) onSelectSegment(segmentId);
        setSelectedPoint({
          name: (props.name as string) || "this location",
          lng: e.lngLat.lng,
          lat: e.lngLat.lat,
        });
      });
    });

    return () => {
      map.remove();
      mapRef.current = null;
      loadedRef.current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Update the road data whenever a new borough's GeoJSON arrives.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !roadsGeoJSON) return;
    const applyData = () => {
      const source = map.getSource("roads") as maplibregl.GeoJSONSource | undefined;
      source?.setData(roadsGeoJSON);
      const bounds = new maplibregl.LngLatBounds();
      let any = false;
      for (const feature of roadsGeoJSON.features) {
        if (feature.geometry.type === "LineString") {
          for (const coord of feature.geometry.coordinates) {
            bounds.extend(coord as [number, number]);
            any = true;
          }
        }
      }
      if (any) {
        map.fitBounds(bounds, { padding: 40, duration: 500 });
        // fitBounds picks whatever zoom fits the WHOLE borough - for a
        // borough the size of Westminster that measured at zoom ~11.1 on
        // the real data (confirmed via queryRenderedFeatures during the
        // 2026-09-08 label-visibility fix), which is below even
        // "highway-name-major"'s own minzoom of 12.2: no street name,
        // major or minor, rendered anywhere on the default view - the
        // labelling fix above only helps once the user has already zoomed
        // in some. A floor here means the view a user actually lands on
        // reads as a real street map from the first paint, not just after
        // they discover they need to zoom in.
        map.once("moveend", () => {
          if (map.getZoom() < 13) map.easeTo({ zoom: 13, duration: 300 });
        });
      }
    };
    if (loadedRef.current) applyData();
    else map.once("load", applyData);
  }, [roadsGeoJSON]);

  // Highlight the selected segment, and bring it into view.
  //
  // Previously only set the highlight filter - harmless when a road is
  // selected by clicking it directly on the map (already in view), but a
  // real gap when selected from the priority queue instead: a road
  // outside the current viewport would highlight with zero visible
  // effect, giving no feedback that a selection even happened (found
  // during a full flow-verification pass, 2026-09-01 - "make sure every
  // flow works", not just the map's own interactions in isolation).
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !loadedRef.current) return;
    const id = selectedSegmentId ?? "";
    for (const layerId of ["roads-selected-glow", "roads-selected-casing", "roads-selected-outline"]) {
      if (map.getLayer(layerId)) {
        map.setFilter(layerId, ["==", ["get", "segment_id"], id]);
      }
    }
    // A brief flash on the glow layer when a NEW segment is selected - a
    // filter change alone re-renders the matched feature at its resting
    // paint values with no transition (transitions fire on a PAINT
    // PROPERTY value change, not a filter change), so without this the
    // highlight would just appear, giving no "something just happened"
    // cue. Bump opacity up, then let the layer's own
    // line-opacity-transition (300ms) ease it back down to rest - the
    // same brief flash-then-settle language QGIS/Figma use for "this is
    // now selected", not a distracting continuous pulse.
    window.clearTimeout(glowFlashTimerRef.current);
    if (selectedSegmentId && map.getLayer("roads-selected-glow")) {
      map.setPaintProperty("roads-selected-glow", "line-opacity", 0.85);
      glowFlashTimerRef.current = window.setTimeout(() => {
        if (mapRef.current?.getLayer("roads-selected-glow")) {
          mapRef.current.setPaintProperty("roads-selected-glow", "line-opacity", 0.5);
        }
      }, 60);
    }
    if (!selectedSegmentId) {
      // A borough switch resets selectedSegmentId to null (see App.tsx) -
      // without this, the fixed-position Street View chip kept pointing at
      // a road in the *previous* borough (found during this verification
      // pass: switching Westminster -> Camden left "Street View: Outer
      // Circle" - a Regent's Park road - showing after the switch, a stale
      // cross-borough reference the brief's "no stale data" flow check is
      // exactly about even though it names the map/queue/evidence panel,
      // not this chip by name).
      setSelectedPoint(null);
      return;
    }
    if (!roadsGeoJSON) return;
    const feature = roadsGeoJSON.features.find((f) => f.properties?.segment_id === selectedSegmentId);
    if (feature && feature.geometry.type === "LineString") {
      const coords = feature.geometry.coordinates as [number, number][];
      const mid = coords[Math.floor(coords.length / 2)];
      map.easeTo({ center: mid, zoom: Math.max(map.getZoom(), 15), duration: 700 });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSegmentId]);

  // 3D buildings toggle - flips the style's own "building-3d" layer's
  // visibility and eases the camera into a pitched view (a 3D building is
  // indistinguishable from its own flat footprint at 0 pitch, looking
  // straight down).
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !loadedRef.current || !map.getLayer("building-3d")) return;
    map.setLayoutProperty("building-3d", "visibility", buildings3D ? "visible" : "none");
    map.easeTo({ pitch: buildings3D ? 55 : 0, duration: 600 });
  }, [buildings3D]);

  const hoverRag = hoverInfo ? ragLabel(hoverInfo.score, hoverInfo.hasScore) : null;

  return (
    <div className="map-view-wrapper">
      <div ref={containerRef} className="map-view" role="application" aria-label="Road risk map" />

      {/* Borough-switch loading feedback - previously only the priority
          queue showed any "Loading..." state; the map itself gave zero
          indication a new borough's roads were being fetched, and could
          look frozen for the ~1-2s a switch takes (found in the full
          flow-verification pass, 2026-09-01). Overlays the *existing*
          map rather than blanking it, so the last borough stays visible
          (and orientating) until the new one is ready. */}
      {loading && (
        <div className="map-loading-overlay" role="status" aria-live="polite">
          <span className="map-loading-spinner" aria-hidden="true" />
          <span>Loading borough…</span>
        </div>
      )}

      {/* Hover preview - was click-only before (2026-09-01 feedback: "now
          I have to click to see"). Follows the cursor; read-only by
          design (see the click handler above for why Street View moved
          out of here) - a click still drives the full evidence panel,
          unchanged. */}
      {hoverInfo && (
        <div
          className="map-hover-card"
          style={{ left: hoverInfo.x + 14, top: hoverInfo.y + 14 }}
          role="status"
        >
          <div className="map-hover-card-title">{hoverInfo.name}</div>
          <div className="map-hover-card-row">
            <span className="map-hover-card-type">{hoverInfo.highway}</span>
            {hoverRag && <span className={`govuk-tag ${hoverRag.className}`}>{hoverRag.label}</span>}
            {hoverInfo.hasScore && hoverInfo.score !== undefined && (
              <span className="map-hover-card-score">{hoverInfo.score.toFixed(0)}</span>
            )}
          </div>
        </div>
      )}

      {/* 3D buildings toggle - a real <button> (keyboard-operable,
          aria-pressed), styled to match the legend card rather than
          MapLibre's own control chrome. */}
      <button
        type="button"
        className="map-3d-toggle"
        aria-pressed={buildings3D}
        onClick={() => setBuildings3D((v) => !v)}
      >
        {buildings3D ? "2D view" : "3D buildings"}
      </button>

      {/* Street View - fixed in place (bottom-right), not attached to the
          cursor, specifically so it is always reachable by a normal mouse
          move-then-click - the hover card above cannot host an
          interactive control reliably (2026-09-01 feedback: "when I try
          to [click], tooltip moves with cursor as well so I cant"). Set on
          every road click; stays until the next click replaces it. */}
      {selectedPoint && (
        <a
          className="map-streetview-chip"
          href={`https://www.google.com/maps/@?api=1&map_action=pano&viewpoint=${selectedPoint.lat},${selectedPoint.lng}`}
          target="_blank"
          rel="noreferrer noopener"
        >
          Street View: {selectedPoint.name} ↗
        </a>
      )}

      {/* The live dashboard previously had no legend at all - only the
          standalone Python map export did. Same RAG scale as the priority
          queue and evidence panel, so the colour language is consistent
          across every surface (see App.css / DESIGN.md). */}
      <div className="map-legend" aria-hidden="true">
        <div className="map-legend-title">Priority score</div>
        <div className="map-legend-bar" />
        <div className="map-legend-row">
          <span>Lower</span>
          <span>Higher</span>
        </div>
        <div className="map-legend-nodata">
          <span className="map-legend-nodata-swatch" />
          <span>No recent modelled estimate</span>
        </div>
      </div>
    </div>
  );
}
