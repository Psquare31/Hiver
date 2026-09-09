"""Bottom-up intent discovery.

The brief asks for "a small set of intents that you define from the data", so
the taxonomy must be visibly derived rather than invented from prior beliefs
about music streaming.

Method: TF-IDF over customer opening messages, KMeans across a sweep of k,
then print each cluster's distinctive terms and sampled messages. A human (me)
reads that output and writes the taxonomy. Clustering is a *reading aid*, not
the taxonomy itself - k-means on short noisy tweets produces clusters that are
partly topical and partly stylistic, so accepting them as labels directly
would bake in artefacts.

The output of this script is what I read to author config/taxonomy.yaml.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.text import normalise  # noqa: E402

PAIRS = ROOT / "data" / "processed" / "pairs_SpotifyCares.parquet"
OUT = ROOT / "results" / "intent_clusters.txt"

# Domain stopwords: brand handles and pure-politeness tokens dominate TF-IDF
# otherwise and produce clusters that differ only in greeting style.
EXTRA_STOP = [
    "spotify", "hi", "hey", "hello", "please", "pls", "thanks", "thank",
    "thx", "just", "im", "ive", "dont", "cant", "wont", "got", "get",
    "like", "know", "need", "want", "help", "guys", "yes", "no", "ok",
    "okay", "app", "amp",
]


def load_messages(n: int = 4000, seed: int = 0) -> pd.Series:
    pairs = pd.read_parquet(PAIRS)
    msgs = pairs["customer_text"].map(normalise)
    # Very short messages ("help!", "?") carry no intent signal and form a
    # junk cluster that crowds out real structure.
    msgs = msgs[msgs.str.split().str.len() >= 5]
    return msgs.sample(min(n, len(msgs)), random_state=seed).reset_index(drop=True)


def sweep(msgs: pd.Series, ks=range(6, 15), seed: int = 0) -> pd.DataFrame:
    vec = TfidfVectorizer(
        max_features=5000,
        stop_words=list(TfidfVectorizer(stop_words="english").get_stop_words())
        + EXTRA_STOP,
        ngram_range=(1, 2),
        min_df=5,
    )
    X = vec.fit_transform(msgs)
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        rows.append(
            {
                "k": k,
                "inertia": round(km.inertia_, 1),
                "silhouette": round(silhouette_score(X, labels, sample_size=2000,
                                                     random_state=seed), 4),
            }
        )
    return pd.DataFrame(rows), vec, X


def describe(msgs: pd.Series, vec, X, k: int, seed: int = 0, n_terms: int = 12,
             n_examples: int = 6) -> str:
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(X)
    terms = np.array(vec.get_feature_names_out())
    lines = [f"\n{'='*78}\nk = {k}\n{'='*78}"]
    order = km.cluster_centers_.argsort()[:, ::-1]
    for c in range(k):
        idx = np.where(labels == c)[0]
        top = terms[order[c, :n_terms]]
        lines.append(f"\n--- cluster {c}  (n={len(idx)}, {100*len(idx)/len(msgs):.1f}%) ---")
        lines.append("  terms: " + ", ".join(top))
        rng = np.random.default_rng(seed)
        for i in rng.choice(idx, size=min(n_examples, len(idx)), replace=False):
            lines.append(f"    * {msgs.iloc[i][:110]}")
    return "\n".join(lines)


if __name__ == "__main__":
    msgs = load_messages()
    print(f"[intents] clustering {len(msgs):,} customer messages")
    stats, vec, X = sweep(msgs)
    print(stats.to_string(index=False))

    chunks = [stats.to_string(index=False)]
    for k in (8, 9, 10):
        chunks.append(describe(msgs, vec, X, k))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(chunks), encoding="utf-8")
    print(f"\n[intents] wrote cluster report -> {OUT}")
