"""Dense and hybrid retrieval, embedded on the local GPU.

BM25 was the original choice because it needs no API, runs offline, and its
evidence is inspectable. Those reasons still hold. But "we didn't try dense
retrieval" is a weaker statement than "we tried it and here is what it bought",
so this module makes the comparison a measured result rather than a claim in a
future-work section.

Runs entirely on the local GPU. No API, no rate limit, no cost - which also
means the hybrid retriever stays inside the offline-reproducible design.

The comparison that matters is not "which retriever ranks better" in the
abstract - we have no relevance judgements. It is whether the retrieved
evidence changes what the drafter can ground on. So `compare_retrievers`
reports overlap, score behaviour, and the share of queries where each method
finds any usable precedent at all.
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agent.retrieve import Exemplar, ResolutionIndex  # noqa: E402
from data.text import normalise  # noqa: E402
from settings import DATA_PROC, BRAND  # noqa: E402

# Small, fast, and strong for short-text similarity. 384-dim keeps the whole
# Apple corpus (~40k substantive pairs) at ~60MB in fp32 - trivially resident
# on a 6GB card alongside the model.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def embed_path(brand: str | None = None) -> Path:
    return DATA_PROC / f"dense_{brand or BRAND}.npz"


class DenseIndex:
    def __init__(self, df: pd.DataFrame, embeddings: np.ndarray, model_name: str):
        self.df = df.reset_index(drop=True)
        self.emb = embeddings          # L2-normalised, so dot product = cosine
        self.model_name = model_name
        self._model = None

    # -- model is loaded lazily so a cached-index run needs no GPU at all ---

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    @classmethod
    def build(
        cls,
        pairs: pd.DataFrame,
        model_name: str = MODEL_NAME,
        batch_size: int = 256,
    ) -> "DenseIndex":
        from sentence_transformers import SentenceTransformer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[dense] embedding {len(pairs):,} pairs on {device}")
        model = SentenceTransformer(model_name, device=device)

        texts = [normalise(t) for t in pairs["customer_text"]]
        emb = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine becomes a dot product
            show_progress_bar=True,
        ).astype(np.float32)

        idx = cls(pairs, emb, model_name)
        idx._model = model
        return idx

    def save(self, path: Path | None = None) -> Path:
        path = path or embed_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, emb=self.emb, model=np.array([self.model_name]))
        self.df.to_parquet(path.with_suffix(".parquet"), index=False)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "DenseIndex":
        path = path or embed_path()
        blob = np.load(path, allow_pickle=False)
        df = pd.read_parquet(path.with_suffix(".parquet"))
        return cls(df, blob["emb"], str(blob["model"][0]))

    def search(
        self,
        query: str,
        k: int = 4,
        before: pd.Timestamp | None = None,
        min_score: float = 0.35,
        pool: int = 200,
    ) -> list[Exemplar]:
        """Cosine top-k, with the same per-query temporal guarantee as BM25."""
        q = self.model.encode(
            [normalise(query)], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)[0]
        scores = self.emb @ q

        candidates = np.argpartition(-scores, min(pool, len(scores) - 1))[:pool]
        candidates = candidates[np.argsort(-scores[candidates])]

        out = []
        for i in candidates:
            if len(out) >= k:
                break
            if scores[i] < min_score:
                break
            row = self.df.iloc[int(i)]
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


class HybridIndex:
    """Reciprocal rank fusion over BM25 and dense results.

    RRF rather than score interpolation because BM25 scores and cosine
    similarities live on incomparable scales, and tuning a blend weight against
    a 210-row golden set would be fitting noise.
    """

    def __init__(self, bm25: ResolutionIndex, dense: DenseIndex, k_rrf: int = 60):
        self.bm25 = bm25
        self.dense = dense
        self.k_rrf = k_rrf

    def search(
        self, query: str, k: int = 4, before: pd.Timestamp | None = None
    ) -> list[Exemplar]:
        lex = self.bm25.search(query, k=k * 3, before=before)
        vec = self.dense.search(query, k=k * 3, before=before)

        fused: dict[str, tuple[float, Exemplar]] = {}
        for rank, e in enumerate(lex):
            fused[e.pair_id] = (1.0 / (self.k_rrf + rank + 1), e)
        for rank, e in enumerate(vec):
            prior = fused.get(e.pair_id, (0.0, e))
            fused[e.pair_id] = (prior[0] + 1.0 / (self.k_rrf + rank + 1), prior[1])

        ranked = sorted(fused.values(), key=lambda x: -x[0])[:k]
        out = []
        for score, e in ranked:
            out.append(
                Exemplar(
                    pair_id=e.pair_id,
                    customer_text=e.customer_text,
                    agent_text=e.agent_text,
                    score=round(score, 5),
                    customer_at=e.customer_at,
                )
            )
        return out


def compare_retrievers(golden: pd.DataFrame, bm25: ResolutionIndex,
                       dense: DenseIndex, k: int = 4) -> pd.DataFrame:
    """How much does dense retrieval actually change the evidence?

    We have no relevance judgements, so this does not claim one retriever is
    better. It reports what a reviewer can check: coverage (does any precedent
    get found), and overlap (are the two methods even looking at the same
    threads).
    """
    hybrid = HybridIndex(bm25, dense)
    rows = []
    for r in golden.itertuples():
        lex = bm25.search(r.customer_text, k=k, before=r.customer_at)
        vec = dense.search(r.customer_text, k=k, before=r.customer_at)
        hyb = hybrid.search(r.customer_text, k=k, before=r.customer_at)
        li, vi = {e.pair_id for e in lex}, {e.pair_id for e in vec}
        rows.append(
            {
                "golden_id": r.golden_id,
                "n_bm25": len(lex),
                "n_dense": len(vec),
                "n_hybrid": len(hyb),
                "overlap": len(li & vi),
                "jaccard": len(li & vi) / len(li | vi) if (li | vi) else 0.0,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys as _sys
    from settings import golden_labelled, pairs_path

    from data.brand import has_substance

    brand = _sys.argv[1] if len(_sys.argv) > 1 else BRAND
    pairs = pd.read_parquet(pairs_path(brand))
    golden = pd.read_parquet(golden_labelled(brand))

    sub = pairs[pairs["agent_text"].map(has_substance)]
    sub = sub[~sub["pair_id"].isin(set(golden["pair_id"]))].reset_index(drop=True)

    idx = DenseIndex.build(sub)
    idx.save()
    print(f"[dense] saved {idx.emb.shape} -> {embed_path(brand)}")

    bm25 = ResolutionIndex.load()
    cmp = compare_retrievers(golden, bm25, idx)
    print("\n[dense] retriever comparison over the golden set")
    print(cmp[["n_bm25", "n_dense", "n_hybrid", "overlap", "jaccard"]].mean().round(3).to_string())
    print(f"\n  queries with zero BM25 evidence : {(cmp['n_bm25']==0).mean():.1%}")
    print(f"  queries with zero dense evidence: {(cmp['n_dense']==0).mean():.1%}")
    cmp.to_csv(ROOT / "results" / f"retriever_comparison_{brand}.csv", index=False)
