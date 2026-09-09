"""BM25 retrieval over how this brand historically resolved similar issues.

This is what makes a drafted reply *grounded* rather than invented. The index
holds (customer message -> agent reply) pairs, we retrieve on the customer
side, and the agent side becomes evidence in the drafting prompt.

THREE CHOICES THAT MATTER FOR THE EVALUATION:

1. ONLY SUBSTANTIVE REPLIES ARE INDEXED. Roughly half of Spotify's replies are
   channel handoffs ("DM us your account email"). Indexing those would teach the
   drafter that the correct response to anything is "please DM us" - and because
   the same handoffs dominate the ground truth, a judge would score that highly.
   Filtering them is the single most important line in this file.

2. RETRIEVAL IS TIME-BOUND, PER QUERY. Only threads strictly earlier than the
   message being answered are eligible. Without this the system retrieves
   replies written after the fact, which is leakage: at serving time those
   replies do not exist.
   The filter is applied at SEARCH time, not by pruning the index. A single
   global cutoff was tried first and failed badly - the corpus spans only
   ~2 months and the golden set covers all of it, so cutting at the earliest
   golden timestamp left 90 usable pairs out of 7,652. Per-query filtering
   keeps the whole index available while preserving the guarantee exactly.

3. LEXICAL, NOT NEURAL. BM25 needs no embedding API, runs offline, and - the
   real reason - its evidence is inspectable. A reviewer can see exactly which
   historical reply drove a draft. Dense retrieval is listed in "what I'd do
   next", not smuggled in here.
"""

from __future__ import annotations

import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.brand import has_substance  # noqa: E402
from data.text import normalise  # noqa: E402

PAIRS = ROOT / "data" / "processed" / "pairs_SpotifyCares.parquet"
INDEX_PATH = ROOT / "data" / "processed" / "bm25_index.pkl"

TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenise(text: str) -> list[str]:
    return TOKEN_RE.findall(normalise(text).lower())


@dataclass
class Exemplar:
    """One retrieved historical resolution."""

    pair_id: str
    customer_text: str
    agent_text: str
    score: float
    customer_at: pd.Timestamp


class ResolutionIndex:
    def __init__(self, df: pd.DataFrame, bm25: BM25Okapi):
        self.df = df.reset_index(drop=True)
        self.bm25 = bm25

    # -- construction ------------------------------------------------------

    @classmethod
    def build(
        cls,
        pairs_path: Path = PAIRS,
        cutoff: pd.Timestamp | None = None,
        exclude_pair_ids: set[str] | None = None,
    ) -> "ResolutionIndex":
        df = pd.read_parquet(pairs_path)

        before = len(df)
        df = df[df["agent_text"].map(has_substance)]
        substantive = len(df)

        if cutoff is not None:
            df = df[df["customer_at"] < cutoff]
        if exclude_pair_ids:
            df = df[~df["pair_id"].isin(exclude_pair_ids)]

        df = df.reset_index(drop=True)
        print(
            f"[retrieve] {before:,} pairs -> {substantive:,} substantive "
            f"-> {len(df):,} indexed (after cutoff/exclusions)"
        )
        if df.empty:
            raise ValueError("Index is empty - check cutoff and filters")

        corpus = [tokenise(t) for t in df["customer_text"]]
        return cls(df, BM25Okapi(corpus))

    def save(self, path: Path = INDEX_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"df": self.df, "bm25": self.bm25}, f)
        return path

    @classmethod
    def load(cls, path: Path = INDEX_PATH) -> "ResolutionIndex":
        with path.open("rb") as f:
            blob = pickle.load(f)
        return cls(blob["df"], blob["bm25"])

    # -- query -------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 4,
        min_score: float = 1.0,
        before: pd.Timestamp | None = None,
        pool: int = 200,
    ) -> list[Exemplar]:
        """Top-k historical resolutions for a customer message.

        `before` enforces the temporal guarantee PER QUERY: only threads that
        occurred strictly earlier than the message being answered are eligible.
        A single global cutoff cannot work here - the corpus is ~2 months wide
        and the golden set spans all of it, so cutting at the earliest golden
        timestamp left 90 usable pairs out of 7,652. Filtering per query keeps
        the full index available while preserving the guarantee exactly.

        The honest consequence, reported rather than hidden: early-period
        messages retrieve less precedent than late-period ones, because less
        history existed. That is what a real deployment would experience.

        `min_score` suppresses irrelevant evidence. A retrieval that matched
        nothing hands the drafter an empty list, so the prompt can say "no
        precedent found" - safer than grounding a reply in an unrelated thread.
        """
        tokens = tokenise(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        # Over-fetch, then apply the time filter, then trim to k. Filtering
        # after ranking is what lets one index serve every cutoff.
        candidates = scores.argsort()[::-1][:pool]
        out = []
        for i in candidates:
            if len(out) >= k:
                break
            if scores[i] < min_score:
                break  # ranked order: everything after is worse
            row = self.df.iloc[i]
            if before is not None and row["customer_at"] >= before:
                continue
            out.append(
                Exemplar(
                    pair_id=row["pair_id"],
                    customer_text=normalise(row["customer_text"]),
                    agent_text=normalise(row["agent_text"]),
                    score=round(float(scores[i]), 3),
                    customer_at=row["customer_at"],
                )
            )
        return out


def build_default() -> ResolutionIndex:
    """Index for the golden-set evaluation.

    Golden pairs are excluded outright (they are the test set). The temporal
    guarantee is enforced per query at search time via `before=`, not by a
    global cutoff here - see ResolutionIndex.search for why.
    """
    golden = pd.read_parquet(ROOT / "golden" / "golden_labelled.parquet")
    idx = ResolutionIndex.build(exclude_pair_ids=set(golden["pair_id"]))
    idx.save()
    return idx


if __name__ == "__main__":
    golden = pd.read_parquet(ROOT / "golden" / "golden_labelled.parquet")
    idx = build_default()

    print(f"[retrieve] index date range: "
          f"{idx.df['customer_at'].min()} -> {idx.df['customer_at'].max()}")

    # Leakage guarantee 1: no golden thread may appear in the index at all.
    overlap = set(idx.df["pair_id"]) & set(golden["pair_id"])
    assert not overlap, f"LEAKAGE: {len(overlap)} golden pairs in index"

    # Leakage guarantee 2: every retrieved exemplar must predate its query.
    # Checked empirically across the whole golden set, not just asserted.
    counts, violations = [], 0
    for _, row in golden.iterrows():
        ex = idx.search(row["customer_text"], k=4, before=row["customer_at"])
        counts.append(len(ex))
        violations += sum(1 for e in ex if e.customer_at >= row["customer_at"])
    assert violations == 0, f"LEAKAGE: {violations} exemplars dated at/after their query"

    import numpy as np
    counts = np.array(counts)
    print(f"[retrieve] leakage checks passed over all {len(golden)} golden rows")
    print(f"[retrieve] exemplars retrieved per query: "
          f"mean={counts.mean():.2f} median={np.median(counts):.0f} "
          f"zero={100*(counts==0).mean():.1f}%")

    # Early-period messages legitimately retrieve less precedent. Quantified
    # here because it is a real limitation the report has to state.
    early = golden["customer_at"] < golden["customer_at"].quantile(0.25)
    print(f"[retrieve] mean exemplars - earliest quartile: {counts[early.values].mean():.2f}"
          f" | rest: {counts[~early.values].mean():.2f}")

    demo = "my songs keep pausing on my iphone after the ios update"
    print(f"[retrieve] demo query: {demo!r}")
    for e in idx.search(demo, k=3):
        print(f"  score={e.score:6.2f}  C: {e.customer_text[:68]}")
        print(f"                  A: {e.agent_text[:68]}")
