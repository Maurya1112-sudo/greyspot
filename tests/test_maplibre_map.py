import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from greyspot.viz.maplibre_map import build_maplibre_map


def _toy_edges():
    return gpd.GeoDataFrame(
        {
            "segment_id": ["1_2_0", "2_3_0"],
            "highway": ["residential", "primary"],
            "geometry": [
                LineString([(-0.15, 51.50), (-0.14, 51.51)]),
                LineString([(-0.14, 51.51), (-0.13, 51.52)]),
            ],
        },
        crs="EPSG:4326",
    )


def test_build_maplibre_map_writes_self_contained_html_with_data_embedded(tmp_path):
    edges = _toy_edges()
    scores = pd.DataFrame({"segment_id": ["1_2_0"], "score": [3.5]})  # segment 2_3_0 has no score
    collisions = pd.DataFrame(
        {
            "longitude": [-0.145],
            "latitude": [51.505],
            "severity_label": ["serious"],
            "collision_year": [2023],
        }
    )
    out_path = tmp_path / "map.html"

    result_path = build_maplibre_map(
        edges, scores, collisions, borough_name="Testborough", model_version="xgboost-v1",
        test_year=2024, out_path=out_path,
    )

    assert result_path == out_path
    html = out_path.read_text(encoding="utf-8")
    # The library must be genuinely bundled inline, not CDN-referenced -
    # this is the exact bug reported against the first version of this map
    # ("Uncaught ReferenceError: maplibregl is not defined" in a real
    # viewer that silently refused the cross-origin <script src> load).
    assert "unpkg.com" not in html
    assert "<script src=" not in html
    assert "MapLibre GL JS" in html  # the vendor file's own license header, proves it's inlined
    assert "Testborough" in html
    assert "xgboost-v1" in html
    assert '"segment_id": "1_2_0"' in html or '"segment_id":"1_2_0"' in html
    assert "roadsData" in html and "collisionsData" in html


def test_build_maplibre_map_marks_unscored_segments_with_has_score_zero(tmp_path):
    edges = _toy_edges()
    scores = pd.DataFrame({"segment_id": ["1_2_0"], "score": [3.5]})  # 2_3_0 deliberately missing
    collisions = pd.DataFrame({"longitude": [], "latitude": [], "severity_label": [], "collision_year": []})
    out_path = tmp_path / "map2.html"

    build_maplibre_map(
        edges, scores, collisions, borough_name="X", model_version="v",
        test_year=2024, out_path=out_path,
    )
    html = out_path.read_text(encoding="utf-8")
    # the unscored segment must appear with has_score 0, not be silently dropped
    assert '"has_score": 0' in html or '"has_score":0' in html


def test_build_maplibre_map_escapes_literal_script_close_tag_in_data(tmp_path):
    # A road name (or any injected string) containing "</script>" must not
    # be able to prematurely close the surrounding <script> tag and break
    # the page - a defensive fix alongside the CDN-bundling one.
    edges = gpd.GeoDataFrame(
        {
            "segment_id": ["1_2_0"],
            "highway": ["</script><script>alert(1)</script>"],
            "geometry": [LineString([(-0.15, 51.50), (-0.14, 51.51)])],
        },
        crs="EPSG:4326",
    )
    scores = pd.DataFrame({"segment_id": ["1_2_0"], "score": [1.0]})
    collisions = pd.DataFrame({"longitude": [], "latitude": [], "severity_label": [], "collision_year": []})
    out_path = tmp_path / "map3.html"

    build_maplibre_map(
        edges, scores, collisions, borough_name="X", model_version="v", test_year=2024, out_path=out_path,
    )
    html = out_path.read_text(encoding="utf-8")
    assert "</script><script>alert(1)</script>" not in html
    assert "alert(1)" in html  # the content survives, just de-fanged as a script boundary
