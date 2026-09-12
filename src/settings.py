"""Single source of truth for which brand the pipeline is built for.

Brand is a parameter, not a constant baked into paths. Switching brands means
a new taxonomy and a fresh labelling pass - it is not free - but nothing in the
code should have to change.

Override with the BRAND environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BRAND = os.environ.get("BRAND", "AppleSupport")

DATA_RAW = ROOT / "data" / "raw" / "twcs.parquet"
DATA_PROC = ROOT / "data" / "processed"
GOLDEN_DIR = ROOT / "golden"
RESULTS = ROOT / "results"
CONFIG = ROOT / "config"


def pairs_path(brand: str | None = None) -> Path:
    return DATA_PROC / f"pairs_{brand or BRAND}.parquet"


def index_path(brand: str | None = None) -> Path:
    return DATA_PROC / f"bm25_index_{brand or BRAND}.pkl"


def golden_pool(brand: str | None = None) -> Path:
    return GOLDEN_DIR / f"golden_pool_{brand or BRAND}.parquet"


def golden_labelled(brand: str | None = None) -> Path:
    return GOLDEN_DIR / f"golden_labelled_{brand or BRAND}.parquet"
