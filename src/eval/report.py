"""Turn stored predictions and judgements into the tables the report quotes.

Everything here reads from results/ - nothing recomputes model output - so the
same numbers appear in the README, the report, and the console regardless of
when it is run.

Also computes the two analyses that exist specifically to undercut the headline
number, because a report that only contains flattering tables is not evidence:
  * judge self-preference (same-family vs cross-family scores)
  * the always-DM baseline's judge score, which shows how much of the metric is
    satisfied by a non-answer
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results"
PRED = RESULTS / "predictions.parquet"
JUDGE = RESULTS / "judgements.parquet"
import os
BRAND = os.environ.get("BRAND", "AppleSupport")
HUMAN = ROOT / "golden" / "human_labels.csv"

DIMS = ["grounded", "actionable", "safe", "tone"]


def judge_table() -> pd.DataFrame | None:
    if not JUDGE.exists():
        return None
    j = pd.read_parquet(JUDGE)
    primary = j[j["judge"] == "primary"]
    if primary.empty:
        return None

    rows = []
    for system, grp in primary.groupby("system"):
        row = {"system": system, "n": len(grp)}
        for d in DIMS:
            if d in grp.columns:
                row[d] = round(pd.to_numeric(grp[d], errors="coerce").mean(), 2)
        row["send_rate"] = round((grp["send_as_is"] == "yes").mean(), 3)
        row["parse_fail"] = int(grp["parse_failed"].sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("send_rate", ascending=False)


def self_preference() -> pd.DataFrame | None:
    """Same-family vs cross-family judging of identical replies.

    A positive delta means the primary judge (Gemini) scores replies more
    generously than an independent judge from another family does. Since the
    drafter is also Gemini, that delta is the self-preference bias, and it is
    reported rather than assumed away.
    """
    if not JUDGE.exists():
        return None
    j = pd.read_parquet(JUDGE)
    if j["judge"].nunique() < 2:
        return None

    prim = j[j["judge"] == "primary"]
    cross = j[j["judge"] == "cross_family"]
    if cross.empty:
        return None

    merged = prim.merge(cross, on=["system", "golden_id"], suffixes=("_p", "_c"))
    rows = []
    for system, grp in merged.groupby("system"):
        row = {"system": system, "n": len(grp)}
        for d in DIMS:
            cp, cc = f"{d}_p", f"{d}_c"
            if cp in grp and cc in grp:
                row[f"{d}_primary"] = round(pd.to_numeric(grp[cp], errors="coerce").mean(), 2)
                row[f"{d}_cross"] = round(pd.to_numeric(grp[cc], errors="coerce").mean(), 2)
                row[f"{d}_delta"] = round(row[f"{d}_primary"] - row[f"{d}_cross"], 2)
        row["send_primary"] = round((grp["send_as_is_p"] == "yes").mean(), 3)
        row["send_cross"] = round((grp["send_as_is_c"] == "yes").mean(), 3)
        row["send_delta"] = round(row["send_primary"] - row["send_cross"], 3)
        rows.append(row)
    return pd.DataFrame(rows)


def agreement_section() -> dict | None:
    """Judge-human agreement, with the annotator's own ceiling."""
    if not (HUMAN.exists() and JUDGE.exists()):
        return None

    from eval.agreement import interpret, judge_vs_human, test_retest

    human = pd.read_csv(HUMAN)
    part_b = human[human["section"] == "B"].copy()
    if part_b.empty:
        return None
    part_b["system"] = part_b["system"].astype(str)

    out = {"ceiling": test_retest(part_b).as_dict()}
    out["ceiling"]["interpretation"] = interpret(out["ceiling"]["value"])

    j = pd.read_parquet(JUDGE)
    out["judges"] = {}
    for judge_name, grp in j.groupby("judge"):
        try:
            res = judge_vs_human(grp, part_b)
        except ValueError:
            continue
        res["binary"]["interpretation"] = interpret(res["binary"]["value"])
        out["judges"][str(judge_name)] = res
    return out


def cost_table() -> pd.DataFrame | None:
    """Tokens and equivalent paid cost per system.

    We pay nothing (free tiers), so reporting spend would report zero and say
    nothing. Tokens and equivalent-cost-at-published-rates are what actually
    transfer to someone deciding what to run in production.
    """
    if not PRED.exists():
        return None
    import yaml

    cfg = yaml.safe_load((ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))
    p = pd.read_parquet(PRED)
    if p["prompt_tokens"].sum() == 0:
        return None

    drafter = cfg["roles"]["drafter"]
    spec = cfg["models"][drafter]
    rows = []
    for system, grp in p.groupby("system"):
        pt = int(grp["prompt_tokens"].sum())
        ct = int(grp["completion_tokens"].sum())
        if pt == 0 and ct == 0:
            continue
        cost = (pt * spec["input_price_per_1m"] + ct * spec["output_price_per_1m"]) / 1e6
        rows.append(
            {
                "system": system,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "tokens_per_message": round((pt + ct) / len(grp), 1),
                "usd_per_1k_messages": round(cost / len(grp) * 1000, 4),
            }
        )
    return pd.DataFrame(rows) if rows else None


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print("RESULTS")
    print("=" * 74)

    summary_path = RESULTS / "summary_table.csv"
    if summary_path.exists():
        print("\n## Task metrics (intent + routing)\n")
        print(pd.read_csv(summary_path).to_string(index=False))

    jt = judge_table()
    if jt is not None:
        print("\n## Judged reply quality (primary judge)\n")
        print(jt.to_string(index=False))
        jt.to_csv(RESULTS / "judge_table.csv", index=False)
        dm = jt[jt["system"] == "always_dm"]
        if not dm.empty:
            print(
                f"\n  NOTE: the always-DM baseline - which answers nothing - scores "
                f"send_rate={float(dm['send_rate'].iloc[0]):.3f}. "
                "How close that sits to the real system is the single best "
                "measure of how much the metric rewards non-answers."
            )
    else:
        print("\n## Judged reply quality\n  (not yet run - needs API keys)")

    sp = self_preference()
    if sp is not None:
        print("\n## Judge self-preference (same-family minus cross-family)\n")
        print(sp.to_string(index=False))
        sp.to_csv(RESULTS / "self_preference.csv", index=False)

    ag = agreement_section()
    if ag is not None:
        print("\n## Judge-human agreement\n")
        c = ag["ceiling"]
        print(
            f"  CEILING - annotator self-consistency (test-retest): "
            f"kappa={c['value']:.3f} [{c['ci_lo']:.3f},{c['ci_hi']:.3f}] "
            f"({c['interpretation']}, n={c['n']})"
        )
        print("  No judge can be expected to exceed this.\n")
        for name, res in ag["judges"].items():
            b = res["binary"]
            print(
                f"  {name:14} kappa={b['value']:.3f} "
                f"[{b['ci_lo']:.3f},{b['ci_hi']:.3f}] ({b['interpretation']}), "
                f"raw={b['raw_agreement']:.3f}, n={res['n_compared']}"
            )
        (RESULTS / "agreement.json").write_text(
            json.dumps(ag, indent=2), encoding="utf-8"
        )
    else:
        print("\n## Judge-human agreement\n  (needs golden/human_labels.csv + judgements)")

    ct = cost_table()
    if ct is not None:
        print("\n## Cost (equivalent, at published paid rates)\n")
        print(ct.to_string(index=False))
        ct.to_csv(RESULTS / "cost_table.csv", index=False)

    print()


if __name__ == "__main__":
    main()
