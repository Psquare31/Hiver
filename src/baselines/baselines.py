"""Baselines the real system has to beat.

The brief asks for two. This dataset demands a third, and it is the most
informative of the set.

  1. TRIVIAL      - majority intent, one fixed canned reply. The floor.
  2. SIMPLE       - TF-IDF + logistic regression for intent (out-of-fold),
                    1-nearest-neighbour retrieval for the reply. No LLM at all.
  3. ALWAYS_DM    - replies "please DM us" to literally everything.

Why ALWAYS_DM earns its place: 47.3% of AppleSupport's real replies are
channel handoffs (measured, src/data/brand.py). So a bot that only ever says "DM us" is not obviously wrong
by the standard of "what did the brand actually do" - it is imitating the modal
brand behaviour. Any judge that rewards it highly is telling us the metric
rewards non-answers. It is included precisely so the report can show that, and
it is the sharpest single piece of evidence in the misleading-number section.

Note on SIMPLE's classifier: it is trained and evaluated by 5-fold
cross-validation over the golden set, and every prediction reported is
out-of-fold. Training on the same 210 rows we score against would inflate it
into a fake ceiling.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agent.retrieve import ResolutionIndex  # noqa: E402
from data.text import normalise  # noqa: E402

import os
BRAND = os.environ.get("BRAND", "AppleSupport")
GOLDEN = ROOT / "golden" / f"golden_labelled_{BRAND}.parquet"

CANNED_REPLY = (
    "Hey there! Sorry to hear you're having trouble. We're looking into this "
    "and will get back to you as soon as we can."
)

# Verbatim in the style of AppleSupport's single most common reply shape.
# Using the brand's own words matters: a clumsy strawman handoff would be easy
# for the judge to mark down, and the whole point of this baseline is that it
# is genuinely hard to distinguish from what the brand really does.
DM_REPLY = (
    "We're here for you. Send us a DM, and we'll take a closer look at the "
    "issue together there."
)


@dataclass
class Prediction:
    """One system's output for one message. Shared shape across all systems."""

    golden_id: str
    intent: str
    route: str
    reply: str
    reason: str = ""
    exemplar_ids: tuple[str, ...] = ()


def trivial(golden: pd.DataFrame) -> list[Prediction]:
    """Majority intent, fixed reply, never escalate.

    Never-escalate is the correct trivial choice: escalating everything would
    trivially maximise escalation recall and look deceptively strong on the
    metric we care most about, so the floor must sit at the opposite extreme.
    """
    majority = golden["intent"].mode()[0]
    return [
        Prediction(
            golden_id=r.golden_id,
            intent=majority,
            route="auto",
            reply=CANNED_REPLY,
            reason="trivial baseline: constant prediction",
        )
        for r in golden.itertuples()
    ]


def always_dm(golden: pd.DataFrame) -> list[Prediction]:
    """The brand's modal behaviour, applied unconditionally."""
    majority = golden["intent"].mode()[0]
    return [
        Prediction(
            golden_id=r.golden_id,
            intent=majority,
            route="auto",
            reply=DM_REPLY,
            reason="always-DM baseline: constant handoff",
        )
        for r in golden.itertuples()
    ]


def simple(golden: pd.DataFrame, index: ResolutionIndex | None = None,
           seed: int = 0) -> list[Prediction]:
    """TF-IDF + logistic regression intent; nearest historical reply verbatim.

    The reply half is deliberately extractive: it returns a real agent reply
    from a similar past thread, unmodified. That is the strongest thing you can
    do without a generative model, and it sets an honest bar - if an LLM cannot
    beat "paste the most similar historical answer", the LLM is not earning its
    cost.
    """
    texts = golden["customer_text"].map(normalise).tolist()
    y = golden["intent"].tolist()

    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )

    # Out-of-fold predictions only. Rare classes can have fewer members than
    # n_splits, so cap folds at the smallest class count.
    min_class = pd.Series(y).value_counts().min()
    n_splits = int(max(2, min(5, min_class)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred_intents = cross_val_predict(pipe, texts, y, cv=cv)

    # Route by the taxonomy prior for the predicted intent - no learned routing,
    # which is the point of a "simple" baseline.
    import yaml

    tax = yaml.safe_load((ROOT / "config" / "taxonomy.yaml").read_text(encoding="utf-8"))
    prior = {i["id"]: i["default_route"] for i in tax["intents"]}

    if index is None:
        index = ResolutionIndex.load()

    out = []
    for row, intent in zip(golden.itertuples(), pred_intents):
        hits = index.search(row.customer_text, k=1, before=row.customer_at)
        if hits:
            reply, ex_ids = hits[0].agent_text, (hits[0].pair_id,)
        else:
            reply, ex_ids = CANNED_REPLY, ()
        out.append(
            Prediction(
                golden_id=row.golden_id,
                intent=str(intent),
                route=prior.get(str(intent), "escalate"),
                reply=reply,
                reason=f"simple baseline: taxonomy prior for {intent}",
                exemplar_ids=ex_ids,
            )
        )
    return out


BASELINES = {
    "trivial": trivial,
    "always_dm": always_dm,
    "simple": simple,
}


if __name__ == "__main__":
    golden = pd.read_parquet(GOLDEN)
    index = ResolutionIndex.load()

    for name, fn in BASELINES.items():
        preds = fn(golden, index) if name == "simple" else fn(golden)
        acc = np.mean([p.intent == g for p, g in zip(preds, golden["intent"])])
        esc = np.mean([p.route == "escalate" for p in preds])
        print(f"{name:10} intent_acc={acc:.3f}  escalate_rate={esc:.3f}")
        print(f"           sample reply: {preds[0].reply[:80]}")
