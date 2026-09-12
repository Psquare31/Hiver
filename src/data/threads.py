"""Reconstruct conversation threads from the flat tweet table.

The raw file is one row per tweet with two link columns:
  in_response_to_tweet_id : parent (nullable)
  response_tweet_id       : comma-separated children (nullable, denormalised)

We rebuild threads by walking parent links to a root, then collecting the chain
in time order. Two details that matter and are easy to get wrong:

1. `created_at` is NOT monotonic with tweet_id, and threads interleave. Sorting
   a thread by timestamp rather than by id is what produces a readable
   conversation.

2. The unit of work for this project is not a whole thread. It is
   (customer's opening message  ->  the brand's first reply), because that is
   the decision an inbound-triage agent actually faces: a message arrives, and
   it must be classified, answered, or escalated with no future context. Using
   the whole thread would leak the resolution into the input.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "twcs.parquet"
OUT_DIR = ROOT / "data" / "processed"


def load_brand_universe(brand: str) -> pd.DataFrame:
    """All tweets belonging to threads that involve `brand`.

    A thread qualifies if the brand authored at least one tweet in it. We pull
    the brand's tweets first, then their parents and children, which is far
    cheaper than building a global thread index over 2.8M rows.
    """
    df = pd.read_parquet(RAW)
    df["created_at"] = pd.to_datetime(df["created_at"], format="mixed", utc=True)

    brand_tweets = df[df["author_id"] == brand]
    if brand_tweets.empty:
        raise ValueError(f"No tweets found for brand {brand!r}")

    # Parents of brand replies = the customer messages we care about.
    parent_ids = set(brand_tweets["in_response_to_tweet_id"].dropna().astype("int64"))

    # Children of brand replies = customer follow-ups.
    child_ids: set[int] = set()
    for raw in brand_tweets["response_tweet_id"].dropna().astype(str):
        for part in raw.split(","):
            part = part.strip()
            if part and part.isdigit():
                child_ids.add(int(part))

    keep = set(brand_tweets["tweet_id"].astype("int64")) | parent_ids | child_ids
    sub = df[df["tweet_id"].astype("int64").isin(keep)].copy()
    return sub


def build_pairs(brand: str) -> pd.DataFrame:
    """Produce (customer opening message -> brand's first reply) pairs.

    Returns one row per pair with the fields every later stage needs.
    """
    sub = load_brand_universe(brand)

    brand_replies = sub[sub["author_id"] == brand].copy()
    brand_replies = brand_replies.dropna(subset=["in_response_to_tweet_id"])
    brand_replies["parent_id"] = brand_replies["in_response_to_tweet_id"].astype("int64")

    # If the brand replied several times to one customer tweet, keep the
    # earliest - that is the "first response" a triage agent would produce.
    brand_replies = brand_replies.sort_values("created_at")
    first = brand_replies.drop_duplicates(subset=["parent_id"], keep="first")

    # Join rather than loop. An earlier row-wise version with per-row .loc
    # lookups was fine on Spotify (41k pairs) and did not finish in 10 minutes
    # on AppleSupport (2.5x the volume) - the lookups are what scale badly,
    # not the data.
    parents = sub.copy()
    parents["parent_id"] = parents["tweet_id"].astype("int64")
    parents = parents.drop_duplicates(subset=["parent_id"], keep="first")
    # Only customer-authored parents are valid inbound messages.
    parents = parents[parents["inbound"].astype(bool)]

    merged = first.merge(
        parents[["parent_id", "author_id", "text", "created_at"]],
        on="parent_id",
        how="inner",
        suffixes=("_agent", "_customer"),
    )
    if merged.empty:
        return pd.DataFrame()

    pairs = pd.DataFrame(
        {
            "pair_id": brand + "-" + merged["parent_id"].astype(str),
            "brand": brand,
            "customer_tweet_id": merged["parent_id"],
            "customer_author": merged["author_id_customer"],
            "customer_text": merged["text_customer"].astype(str),
            "customer_at": merged["created_at_customer"],
            "agent_tweet_id": merged["tweet_id"].astype("int64"),
            "agent_text": merged["text_agent"].astype(str),
            "agent_at": merged["created_at_agent"],
        }
    )

    pairs["response_lag_min"] = (
        (pairs["agent_at"] - pairs["customer_at"]).dt.total_seconds() / 60
    ).round(1)
    # A handful of rows have the reply timestamped before the customer message
    # (clock skew in the original scrape). Drop them rather than let negative
    # lags pollute any time-based split.
    pairs = pairs[pairs["response_lag_min"] >= 0]

    return pairs.sort_values("customer_at").reset_index(drop=True)


def main(brand: str = "SpotifyCares") -> Path:
    pairs = build_pairs(brand)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"pairs_{brand}.parquet"
    pairs.to_parquet(out, index=False)

    print(f"[threads] brand={brand}")
    print(f"[threads] pairs={len(pairs):,}")
    print(f"[threads] date range: {pairs['customer_at'].min()} -> {pairs['customer_at'].max()}")
    print(f"[threads] median response lag: {pairs['response_lag_min'].median():.1f} min")
    print(f"[threads] wrote {out}")
    return out


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "SpotifyCares")
