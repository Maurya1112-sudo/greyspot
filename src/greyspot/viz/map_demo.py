"""Minimal Folium risk map: Westminster roads coloured by modelled score,
with real collision points overlaid. This is the Section-4 "Map" workspace
reduced to its simplest possible honest version — no priority-score
decomposition, uncertainty layer or scenario tooling yet (those are later
phases); just: here is the network, here is the modelled signal, here is
the ground truth it was trained on.
"""
from __future__ import annotations

from pathlib import Path

import branca.colormap as cm
import folium
import geopandas as gpd
import pandas as pd


def build_risk_map(
    edges: gpd.GeoDataFrame,
    segment_scores: pd.DataFrame,
    collisions: pd.DataFrame,
    score_col: str = "score",
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    max_collision_points: int = 500,
) -> folium.Map:
    merged = edges.merge(segment_scores[["segment_id", score_col]], on="segment_id", how="left")
    merged[score_col] = merged[score_col].fillna(0.0)

    centroid = merged.geometry.union_all().centroid
    fmap = folium.Map(location=[centroid.y, centroid.x], zoom_start=14, tiles="cartodbpositron")

    vmax = max(merged[score_col].max(), 1e-9)
    colormap = cm.LinearColormap(
        colors=["#2c7bb6", "#ffffbf", "#d7191c"], vmin=0, vmax=vmax,
        caption="Modelled relative risk score (illustrative, not a safety claim)",
    )
    colormap.add_to(fmap)

    for _, row in merged.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        coords = [(lat, lon) for lon, lat in row.geometry.coords]
        folium.PolyLine(
            coords,
            color=colormap(row[score_col]),
            weight=3,
            opacity=0.8,
            tooltip=f"segment {row['segment_id']} | score={row[score_col]:.2f}",
        ).add_to(fmap)

    pts = collisions.dropna(subset=[lon_col, lat_col])
    if len(pts) > max_collision_points:
        pts = pts.sample(max_collision_points, random_state=42)
    for _, row in pts.iterrows():
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=2,
            color="black",
            fill=True,
            fill_opacity=0.6,
            tooltip=f"{row.get('severity_label', 'collision')} | {row.get('collision_year', '')}",
        ).add_to(fmap)

    return fmap


def save_map(fmap: folium.Map, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_path))
    return out_path
