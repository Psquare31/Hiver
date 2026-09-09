"""Metrics, with the honesty machinery built in rather than bolted on.

FOUR THINGS THIS MODULE INSISTS ON:

1. MACRO-F1, NOT ACCURACY, for intent. The classes are heavily imbalanced
   (feature_request 31, playlist_library 10). Accuracy lets a model coast on
   the big classes; macro-F1 makes every class count equally, which is what a
   support team actually cares about.

2. EVERY HEADLINE NUMBER CARRIES A BOOTSTRAP CI. n=210 means roughly +/-7
   points on a proportion. Reporting a bare point estimate at this sample size
   is the most common way an evaluation misleads without lying.

3. WEIGHTED AND UNWEIGHTED SIDE BY SIDE. The golden set deliberately
   oversamples hard cases, so raw numbers understate live performance. Weights
   reproject onto the natural distribution. Both are reported; the gap between
   them IS the finding, not an embarrassment to hide.

4. ROUTING IS SCORED ASYMMETRICALLY. Auto-handling something that needed a
   human ("harmful auto") is far more costly than escalating something a bot
   could have answered. They are counted and reported separately, never
   averaged into a single accuracy that hides the tradeoff.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

RNG_SEED = 12345
N_BOOTSTRAP = 2000


@dataclass
class Interval:
    point: float
    lo: float
    hi: float

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}]"

    def as_dict(self) -> dict[str, float]:
        return {"point": self.point, "ci_lo": self.lo, "ci_hi": self.hi}


def bootstrap_ci(
    values: np.ndarray,
    statistic=np.mean,
    weights: np.ndarray | None = None,
    n: int = N_BOOTSTRAP,
    alpha: float = 0.05,
    seed: int = RNG_SEED,
) -> Interval:
    """Percentile bootstrap over the sample.

    Resamples rows (not predictions), which is the right unit: the uncertainty
    we care about is "would a different 210 messages give a different answer".
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(values))

    if weights is None:
        point = float(statistic(values))
    else:
        weights = np.asarray(weights, dtype=float)
        point = float(np.average(values, weights=weights))

    draws = np.empty(n)
    for i in range(n):
        pick = rng.choice(idx, size=len(idx), replace=True)
        if weights is None:
            draws[i] = statistic(values[pick])
        else:
            w = weights[pick]
            draws[i] = np.average(values[pick], weights=w) if w.sum() > 0 else np.nan

    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(point, float(lo), float(hi))


def bootstrap_macro_f1(
    y_true: list[str], y_pred: list[str], n: int = N_BOOTSTRAP, seed: int = RNG_SEED
) -> Interval:
    """Macro-F1 with a CI. Resamples rows and recomputes F1 each draw."""
    y_true_a, y_pred_a = np.asarray(y_true), np.asarray(y_pred)
    point = f1_score(y_true_a, y_pred_a, average="macro", zero_division=0)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true_a))
    draws = np.empty(n)
    for i in range(n):
        pick = rng.choice(idx, size=len(idx), replace=True)
        draws[i] = f1_score(
            y_true_a[pick], y_pred_a[pick], average="macro", zero_division=0
        )
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return Interval(float(point), float(lo), float(hi))


@dataclass
class RoutingResult:
    """Routing scored with the cost asymmetry made explicit."""

    harmful_auto_rate: Interval          # needed human, got bot  <- the dangerous one
    unnecessary_escalation_rate: Interval  # bot could have handled it
    escalation_recall: Interval
    escalation_precision: Interval
    auto_handled_rate: Interval
    n: int

    def as_dict(self) -> dict:
        return {
            "harmful_auto_rate": self.harmful_auto_rate.as_dict(),
            "unnecessary_escalation_rate": self.unnecessary_escalation_rate.as_dict(),
            "escalation_recall": self.escalation_recall.as_dict(),
            "escalation_precision": self.escalation_precision.as_dict(),
            "auto_handled_rate": self.auto_handled_rate.as_dict(),
            "n": self.n,
        }


def routing_metrics(
    y_true: list[str], y_pred: list[str], weights: np.ndarray | None = None
) -> RoutingResult:
    """Score auto/escalate decisions.

    Convention: 'escalate' is the positive class, because it is the action
    whose omission causes harm.
    """
    t = np.array([1 if v == "escalate" else 0 for v in y_true])
    p = np.array([1 if v == "escalate" else 0 for v in y_pred])

    # Harmful auto: truth said escalate, system auto-handled.
    harmful = ((t == 1) & (p == 0)).astype(float)
    # Unnecessary escalation: truth said auto, system escalated.
    unnecessary = ((t == 0) & (p == 1)).astype(float)

    tp = float(((t == 1) & (p == 1)).sum())
    fn = float(((t == 1) & (p == 0)).sum())
    fp = float(((t == 0) & (p == 1)).sum())

    recall_vals = np.where(t == 1, (p == 1).astype(float), np.nan)
    precision_vals = np.where(p == 1, (t == 1).astype(float), np.nan)

    return RoutingResult(
        harmful_auto_rate=bootstrap_ci(harmful, weights=weights),
        unnecessary_escalation_rate=bootstrap_ci(unnecessary, weights=weights),
        escalation_recall=bootstrap_ci(recall_vals[~np.isnan(recall_vals)]),
        escalation_precision=(
            bootstrap_ci(precision_vals[~np.isnan(precision_vals)])
            if (tp + fp) > 0
            else Interval(float("nan"), float("nan"), float("nan"))
        ),
        auto_handled_rate=bootstrap_ci((p == 0).astype(float), weights=weights),
        n=len(t),
    )


def safe_auto_rate(
    route_true: list[str],
    route_pred: list[str],
    weights: np.ndarray | None = None,
    harm_budget: float = 0.05,
) -> dict:
    """THE HEADLINE METRIC: share of traffic safely auto-handled.

    Defined as the auto-handled rate, but only creditable while the harmful-auto
    rate stays within `harm_budget`. Reported together with whether the budget
    was met - an auto-handle rate achieved by ignoring the budget is not a
    result, it is a liability.
    """
    r = routing_metrics(route_true, route_pred, weights=weights)
    within = r.harmful_auto_rate.point <= harm_budget
    return {
        "auto_handled_rate": r.auto_handled_rate.as_dict(),
        "harmful_auto_rate": r.harmful_auto_rate.as_dict(),
        "harm_budget": harm_budget,
        "within_budget": bool(within),
        "safe_auto_rate": r.auto_handled_rate.point if within else 0.0,
    }


def intent_report(
    y_true: list[str], y_pred: list[str], labels: list[str] | None = None
) -> pd.DataFrame:
    """Per-class precision/recall/F1/support, sorted by support."""
    labels = labels or sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for i, lab in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append(
            {
                "intent": lab,
                "precision": round(prec, 3),
                "recall": round(rec, 3),
                "f1": round(f1, 3),
                "support": int(cm[i, :].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("support", ascending=False).reset_index(drop=True)


def confusion_frame(
    y_true: list[str], y_pred: list[str], labels: list[str] | None = None
) -> pd.DataFrame:
    labels = labels or sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"true_{l}" for l in labels],
                        columns=[f"pred_{l}" for l in labels])


def evaluate_system(
    golden: pd.DataFrame,
    pred_intents: list[str],
    pred_routes: list[str],
    name: str = "system",
) -> dict:
    """Full metric bundle for one system, weighted and unweighted."""
    y_intent = golden["intent"].tolist()
    y_route = golden["route"].tolist()
    w = golden["weight"].to_numpy()

    correct = np.array([a == b for a, b in zip(y_intent, pred_intents)], dtype=float)

    return {
        "system": name,
        "n": len(golden),
        "intent_accuracy_raw": bootstrap_ci(correct).as_dict(),
        "intent_accuracy_weighted": bootstrap_ci(correct, weights=w).as_dict(),
        "intent_macro_f1": bootstrap_macro_f1(y_intent, pred_intents).as_dict(),
        "routing_raw": routing_metrics(y_route, pred_routes).as_dict(),
        "routing_weighted": routing_metrics(y_route, pred_routes, weights=w).as_dict(),
        "headline_safe_auto": safe_auto_rate(y_route, pred_routes, weights=w),
    }
