"""Where does the agent become safe enough to deploy?

The headline metric answers "is this system safe at the chosen operating
point". At the default confidence floor of 0.55 the answer is no for every
system, including the agent - which is honest but non-discriminating, and
leaves the more useful question unasked:

    what fraction of traffic CAN be auto-handled while staying inside the
    harm budget, and what does it cost to get there?

The confidence floor was picked a priori at 0.55. This sweeps it against the
golden set and reports the whole curve, so the operating point is chosen from
evidence rather than assumed. Everything here is recomputation over stored
predictions - no API calls - so it is free to run and free to re-run when the
budget changes.

Read the output as a trade: raising the floor escalates more, which buys a
lower harmful-auto rate and costs automation. The report quotes the point where
harmful-auto first fits the budget, plus the price paid in auto-rate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval.metrics import bootstrap_ci  # noqa: E402

import os  # noqa: E402

BRAND = os.environ.get("BRAND", "AppleSupport")
GOLDEN = ROOT / "golden" / f"golden_labelled_{BRAND}.parquet"
PRED = ROOT / "results" / "predictions.parquet"
TAXONOMY = ROOT / "config" / "taxonomy.yaml"
OUT = ROOT / "results" / "threshold_sweep.csv"


def reroute(merged: pd.DataFrame, floor: float, priors: dict[str, str]) -> np.ndarray:
    """Re-apply the routing policy at a different confidence floor.

    Mirrors Router.route's ordering: a fired rule wins, then the floor, then
    the intent prior. Rule hits are taken from the recorded `route_rule` so
    this stays consistent with what actually ran.
    """
    out = []
    for r in merged.itertuples():
        rule = str(getattr(r, "route_rule", "") or "")
        # A hard rule fired at run time - it is independent of the floor.
        if rule not in ("intent_prior", "low_confidence", "baseline", ""):
            out.append("escalate")
            continue
        if float(r.confidence) < floor:
            out.append("escalate")
            continue
        out.append(priors.get(str(r.intent_pred), "escalate"))
    return np.array(out)


def sweep(system: str = "agent", harm_budget: float = 0.05) -> pd.DataFrame:
    golden = pd.read_parquet(GOLDEN)
    preds = pd.read_parquet(PRED)
    tax = yaml.safe_load(TAXONOMY.read_text(encoding="utf-8"))
    priors = {i["id"]: i["default_route"] for i in tax["intents"]}

    grp = preds[preds["system"] == system]
    merged = golden.merge(grp, on="golden_id", suffixes=("_true", "_pred"))
    truth = merged["route_true"].to_numpy()
    weights = merged["weight"].to_numpy()

    rows = []
    for floor in np.round(np.arange(0.0, 1.01, 0.05), 2):
        pred = reroute(merged, floor, priors)
        harmful = ((truth == "escalate") & (pred == "auto")).astype(float)
        auto = (pred == "auto").astype(float)
        unnecessary = ((truth == "auto") & (pred == "escalate")).astype(float)

        h = bootstrap_ci(harmful, weights=weights)
        a = bootstrap_ci(auto, weights=weights)
        rows.append(
            {
                "confidence_floor": floor,
                "auto_rate": round(a.point, 3),
                "auto_ci_lo": round(a.lo, 3),
                "auto_ci_hi": round(a.hi, 3),
                "harmful_auto": round(h.point, 3),
                "harmful_ci_hi": round(h.hi, 3),
                "unnecessary_escalation": round(float(unnecessary.mean()), 3),
                "within_budget": bool(h.point <= harm_budget),
                # The stricter test: could the TRUE harm rate exceed the budget
                # given sampling error? At n=210 this matters, and a point
                # estimate that just squeaks under is not a safety argument.
                "within_budget_ci": bool(h.hi <= harm_budget),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    system = sys.argv[1] if len(sys.argv) > 1 else "agent"
    df = sweep(system)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    pd.set_option("display.width", 140)
    print(f"Confidence-floor sweep for '{system}' (harm budget 5%)\n")
    print(df.to_string(index=False))

    ok = df[df["within_budget"]]
    ok_ci = df[df["within_budget_ci"]]
    print()
    if ok.empty:
        print("  No floor brings the point estimate inside the 5% harm budget.")
    else:
        best = ok.sort_values("auto_rate", ascending=False).iloc[0]
        print(f"  Best point estimate inside budget: floor={best.confidence_floor}"
              f" -> auto {best.auto_rate:.1%}, harmful {best.harmful_auto:.1%}")
    if ok_ci.empty:
        print("  No floor is inside the budget once the CI upper bound is used.")
        print("  At n=210 that is the honest read: the budget cannot be "
              "demonstrated, only estimated.")
    else:
        best = ok_ci.sort_values("auto_rate", ascending=False).iloc[0]
        print(f"  Best with CI upper bound inside budget: floor={best.confidence_floor}"
              f" -> auto {best.auto_rate:.1%}")
    print(f"\n[threshold] wrote {OUT}")
