"""Fetch the raw Twitter Customer Support corpus.

Source: HuggingFace `SunidhiSriram/twcs`, a byte-faithful mirror of the Kaggle
`thoughtvector/customer-support-on-twitter` file (2,811,774 rows, 7 columns).
Using the mirror avoids the Kaggle credential dance; the schema is identical.

The mirror ships a single `twcs.csv`. We convert once to parquet at
data/raw/twcs.parquet, which every later stage reads (parquet load is ~10x
faster and preserves dtypes, which matters for the nullable id columns).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "SunidhiSriram/twcs"
EXPECTED_ROWS = 2_811_774
EXPECTED_COLS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "twcs.parquet"


def _resolve_sources() -> tuple[list[str], str]:
    """Discover the data files in the repo and their format.

    The mirror's layout has changed before (csv vs parquet shards), so discover
    it rather than hard-coding a path that silently 404s.
    """
    files = list_repo_files(REPO_ID, repo_type="dataset")
    parquet = sorted(f for f in files if f.endswith(".parquet"))
    if parquet:
        return parquet, "parquet"
    csv = sorted(f for f in files if f.endswith(".csv"))
    if csv:
        return csv, "csv"
    raise RuntimeError(
        f"No data files found in {REPO_ID}. Files present: {files}"
    )


def download(force: bool = False) -> Path:
    if OUT.exists() and not force:
        print(f"[download] already present: {OUT}")
        return OUT

    sources, fmt = _resolve_sources()
    print(f"[download] format={fmt} sources={sources}")

    frames = []
    for src in sources:
        local = hf_hub_download(REPO_ID, src, repo_type="dataset")
        print(f"[download] fetched {src}")
        if fmt == "parquet":
            frames.append(pd.read_parquet(local))
        else:
            # in_response_to_tweet_id has nulls; keep ids as nullable Int64 so
            # thread reconstruction can join on them without float coercion.
            frames.append(
                pd.read_csv(
                    local,
                    dtype={
                        "tweet_id": "Int64",
                        "author_id": "string",
                        "text": "string",
                        "response_tweet_id": "string",
                        "in_response_to_tweet_id": "Int64",
                    },
                    low_memory=False,
                )
            )

    df = pd.concat(frames, ignore_index=True)

    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Schema drift in {REPO_ID}: missing {missing}. Got {list(df.columns)}"
        )
    df = df[EXPECTED_COLS]

    if len(df) != EXPECTED_ROWS:
        # Not fatal - the mirror could be re-uploaded - but the report quotes this
        # number, so surface any drift loudly instead of letting it pass.
        print(
            f"[download] WARNING: expected {EXPECTED_ROWS:,} rows, got {len(df):,}",
            file=sys.stderr,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"[download] wrote {len(df):,} rows -> {OUT}")
    return OUT


if __name__ == "__main__":
    download(force="--force" in sys.argv)
