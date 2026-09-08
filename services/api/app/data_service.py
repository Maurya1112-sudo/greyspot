"""Loads the precomputed serving artifacts (`scripts/build_serving_artifacts.py`)
backing the FastAPI endpoints.

**This replaces an earlier version that trained XGBoost per-request** on a
different, older feature pipeline (`scripts/run_pipeline.py`). That path
predates this project's final model and was never reconnected after the
GAT+GRU+ZIP architecture became the validated one; running it would have
served a model this project's own paper does not describe. The real final
model is used to build a static artifact once
(`scripts/build_serving_artifacts.py`), which this module loads - no
training happens inside a request.

**Both a model ranking and a baseline ranking are always available.** The
paper's central finding is that a parameter-free crash-count sort matches
or beats the model; hiding that from the product would misrepresent the
project's own research. `rank_by="baseline"` on the priority-queue
endpoint serves the baseline ranking through the identical response shape,
so a frontend toggle needs no special-casing.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import geopandas as gpd
import pandas as pd

from greyspot.ingest.boroughs import Borough, get_borough, slug

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
SERVED_DIR = ROOT / "data" / "served"


@dataclass
class BoroughData:
    borough: Borough
    edges: gpd.GeoDataFrame          # segment_id + geometry + scored columns
    scored_table: pd.DataFrame       # segment_id + model_score/baseline_score/evidence, no geometry
    model_info: dict


class BoroughDataCache:
    """Process-lifetime cache of the precomputed artifacts. Loading is a
    parquet/geojson read (milliseconds to low seconds), not model training,
    so there is no meaningful "first request is slow" behaviour to hide -
    unlike the XGBoost predecessor, which trained a model on first use."""

    _cache: ClassVar[dict[str, BoroughData]] = {}

    @classmethod
    def get(cls, borough_name: str) -> BoroughData:
        borough = get_borough(borough_name)  # raises ValueError for an unknown name
        if borough.name not in cls._cache:
            logger.info("Loading served artifact for %s...", borough.name)
            cls._cache[borough.name] = cls._load(borough)
        return cls._cache[borough.name]

    @classmethod
    def _load(cls, borough: Borough) -> BoroughData:
        bslug = slug(borough.name)
        out_dir = SERVED_DIR / bslug
        info_path = out_dir / "model_info.json"
        geo_path = out_dir / "segments.geojson"
        parquet_path = out_dir / "segments.parquet"
        if not (info_path.exists() and geo_path.exists() and parquet_path.exists()):
            raise FileNotFoundError(
                f"No served artifact for {borough.name} yet. Run "
                f"`python scripts/build_serving_artifacts.py {borough.name!r}` first."
            )
        model_info = json.loads(info_path.read_text(encoding="utf-8"))
        edges = gpd.read_file(geo_path)
        scored_table = pd.read_parquet(parquet_path)
        return BoroughData(borough=borough, edges=edges, scored_table=scored_table, model_info=model_info)


MODEL_VERSION = "gat-gru-zip-final-v1"  # kept in sync with build_serving_artifacts.MODEL_VERSION
