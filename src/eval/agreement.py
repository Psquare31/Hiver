"""Judge-human agreement, and the ceiling on what agreement can mean.

This module answers the brief's hardest requirement: evidence that the LLM
judge agrees with a human.

THE PART MOST EVALUATIONS OMIT - THE CEILING.
A kappa of 0.6 between judge and human is meaningless on its own. If the human
only agrees with THEMSELVES 0.65 of the time on repeated items, then 0.6 is
close to the maximum attainable and the judge is excellent. If the human is
self-consistent at 0.95, then 0.6 is poor. So the labelling sheet silently
duplicates 15 items, and `test_retest` measures the annotator against their own
earlier answers. Every judge-human kappa is reported against that ceiling.

Cohen's kappa is used for the binary send/no-send decision, and Krippendorff's
alpha with ordinal weighting for the 1-5 dimensions, where treating a 4-vs-5
disagreement as equal to a 1-vs-5 disagreement would badly understate agreement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import krippendorff
except ImportError:  # pragma: no cover
    krippendorff = None


@dataclass
class AgreementResult:
    metric: str
    value: float
    ci_lo: float
    ci_hi: float
    n: int
    raw_agreement: float

    def __str__(self) -> str:
        return (
            f"{self.metric}={self.value:.3f} [{self.ci_lo:.3f}, {self.ci_hi:.3f}] "
            f"(raw={self.raw_agreement:.3f}, n={self.n})"
        )

    def as_dict(self) -> dict:
        return {
            "metric": self.metric,
            "value": self.value,
            "ci_lo": self.ci_lo,
            "ci_hi": self.ci_hi,
            "n": self.n,
            "raw_agreement": self.raw_agreement,
        }


def cohens_kappa(a: list, b: list) -> float:
    """Cohen's kappa for two raters over the same items."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0:
        return float("nan")
    labels = sorted(set(a.tolist()) | set(b.tolist()))
    if len(labels) < 2:
        # Both raters used one label. Kappa is undefined; raw agreement is 1.
        return float("nan")

    idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n))
    for x, y in zip(a, b):
        cm[idx[x], idx[y]] += 1
    total = cm.sum()
    po = np.trace(cm) / total
    pe = float((cm.sum(axis=0) * cm.sum(axis=1)).sum()) / (total**2)
    if np.isclose(pe, 1.0):
        return float("nan")
    return float((po - pe) / (1 - pe))


def bootstrap_kappa(
    a: list, b: list, n_boot: int = 2000, seed: int = 12345
) -> AgreementResult:
    a, b = np.asarray(a), np.asarray(b)
    point = cohens_kappa(a, b)
    raw = float(np.mean(a == b)) if len(a) else float("nan")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(a))
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(idx, size=len(idx), replace=True)
        k = cohens_kappa(a[pick], b[pick])
        if not np.isnan(k):
            draws.append(k)
    if draws:
        lo, hi = np.percentile(draws, [2.5, 97.5])
    else:
        lo = hi = float("nan")
    return AgreementResult("cohens_kappa", point, float(lo), float(hi), len(a), raw)


def ordinal_alpha(ratings_a: list, ratings_b: list) -> float:
    """Krippendorff's alpha with ordinal weighting for the 1-5 scales."""
    if krippendorff is None:
        return float("nan")
    data = np.array([ratings_a, ratings_b], dtype=float)
    try:
        return float(
            krippendorff.alpha(reliability_data=data, level_of_measurement="ordinal")
        )
    except Exception:
        return float("nan")


def test_retest(human: pd.DataFrame, id_col: str = "golden_id",
                value_col: str = "send_as_is") -> AgreementResult:
    """Annotator self-consistency from silently duplicated items.

    This is the ceiling: no judge can be expected to agree with the human more
    than the human agrees with themselves.
    """
    dupes = human[human.duplicated(subset=[id_col], keep=False)]
    if dupes.empty:
        return AgreementResult("test_retest_kappa", float("nan"), float("nan"),
                               float("nan"), 0, float("nan"))

    first, second = [], []
    for _, grp in dupes.groupby(id_col):
        if len(grp) >= 2:
            ordered = grp.sort_values("presented_order") if "presented_order" in grp else grp
            first.append(ordered.iloc[0][value_col])
            second.append(ordered.iloc[1][value_col])

    res = bootstrap_kappa(first, second)
    res.metric = "test_retest_kappa"
    return res


def judge_vs_human(
    judge: pd.DataFrame,
    human: pd.DataFrame,
    id_col: str = "golden_id",
    system_col: str = "system",
) -> dict:
    """Compare one judge's verdicts to the human labels on shared items.

    Joins on (golden_id, system) so a reply is only ever compared against the
    human score for that same reply. Duplicated human rows are collapsed to
    their first response, so the test-retest items do not double-count.
    """
    keys = [id_col] + ([system_col] if system_col in human.columns else [])
    h = human.drop_duplicates(subset=keys, keep="first")
    merged = judge.merge(h, on=keys, suffixes=("_judge", "_human"))
    if merged.empty:
        raise ValueError(f"No overlapping rows between judge and human on {keys}")

    out = {
        "n_compared": len(merged),
        "binary": bootstrap_kappa(
            merged["send_as_is_judge"].tolist(),
            merged["send_as_is_human"].tolist(),
        ).as_dict(),
        "dimensions": {},
    }

    for dim in ("grounded", "actionable", "safe", "tone"):
        jc, hc = f"{dim}_judge", f"{dim}_human"
        if jc in merged.columns and hc in merged.columns:
            j = pd.to_numeric(merged[jc], errors="coerce")
            hh = pd.to_numeric(merged[hc], errors="coerce")
            mask = j.notna() & hh.notna()
            if mask.sum() < 2:
                continue
            out["dimensions"][dim] = {
                "krippendorff_ordinal": ordinal_alpha(
                    j[mask].tolist(), hh[mask].tolist()
                ),
                "exact_agreement": float((j[mask] == hh[mask]).mean()),
                "within_one": float((abs(j[mask] - hh[mask]) <= 1).mean()),
                "mean_judge": float(j[mask].mean()),
                "mean_human": float(hh[mask].mean()),
                # Positive = judge is more generous than the human. Reported
                # because a judge that is uniformly kinder still ranks systems
                # correctly, while one that is erratic does not.
                "judge_bias": float((j[mask] - hh[mask]).mean()),
                "n": int(mask.sum()),
            }
    return out


def interpret(kappa: float) -> str:
    """Landis & Koch bands, stated so the report cannot quietly inflate them."""
    if np.isnan(kappa):
        return "undefined"
    if kappa < 0.0:
        return "worse than chance"
    if kappa < 0.20:
        return "slight"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate"
    if kappa < 0.80:
        return "substantial"
    return "almost perfect"
