"""End-to-end runner: systems -> predictions -> judge -> metrics -> results.

Stages are separately invocable so a rate-limited run can be resumed. Every LLM
call goes through the shared cache, so re-running after an interruption costs
nothing for work already done.

  python src/run_pipeline.py predict   # classify + draft + route, all systems
  python src/run_pipeline.py judge     # score every reply with the judge(s)
  python src/run_pipeline.py metrics   # compute tables from stored predictions
  python src/run_pipeline.py all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.classify import IntentClassifier  # noqa: E402
from agent.draft import ReplyDrafter  # noqa: E402
from agent.retrieve import ResolutionIndex  # noqa: E402
from agent.route import Router  # noqa: E402
from baselines.baselines import BASELINES  # noqa: E402
from eval.judge import ReplyJudge  # noqa: E402
from eval.metrics import evaluate_system, intent_report  # noqa: E402
from llm.client import LLMClient  # noqa: E402

import os
BRAND = os.environ.get("BRAND", "AppleSupport")
GOLDEN = ROOT / "golden" / f"golden_labelled_{BRAND}.parquet"
RESULTS = ROOT / "results"
PRED_PATH = RESULTS / "predictions.parquet"
AGREEMENT_IDS = ROOT / "golden" / "agreement_ids.txt"
JUDGE_PATH = RESULTS / "judgements.parquet"


def run_agent(
    golden: pd.DataFrame,
    client: LLMClient,
    index: ResolutionIndex,
    classifier_key: str | None = None,
    drafter_key: str | None = None,
    system_name: str = "agent",
) -> list[dict]:
    """The full system: classify -> retrieve -> draft -> route."""
    clf = IntentClassifier(client, model_key=classifier_key)
    drafter = ReplyDrafter(client, index, model_key=drafter_key)
    router = Router()

    rows = []
    for r in tqdm(list(golden.itertuples()), desc=system_name):
        c = clf.classify(r.customer_text)
        d = drafter.draft(r.customer_text, c.intent, before=r.customer_at)
        decision = router.route(r.customer_text, c.intent, c.confidence)
        rows.append(
            {
                "system": system_name,
                "golden_id": r.golden_id,
                "intent": c.intent,
                "confidence": c.confidence,
                "route": decision.route,
                "route_reason": decision.reason,
                "route_rule": decision.rule_id,
                "reply": d.reply,
                "exemplar_ids": ",".join(d.exemplar_ids),
                "grounded": d.grounded,
                "prompt_tokens": d.prompt_tokens,
                "completion_tokens": d.completion_tokens,
                "classifier_model": c.served_by,
                "drafter_model": d.served_by,
            }
        )
    return rows


def stage_predict(args) -> None:
    golden = pd.read_parquet(GOLDEN)
    index = ResolutionIndex.load()
    rows: list[dict] = []

    # Baselines first - no API keys required, so they always produce output.
    for name, fn in BASELINES.items():
        preds = fn(golden, index) if name == "simple" else fn(golden)
        for p in preds:
            rows.append(
                {
                    "system": name,
                    "golden_id": p.golden_id,
                    "intent": p.intent,
                    "confidence": 1.0,
                    "route": p.route,
                    "route_reason": p.reason,
                    "route_rule": "baseline",
                    "reply": p.reply,
                    "exemplar_ids": ",".join(p.exemplar_ids),
                    "grounded": bool(p.exemplar_ids),
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "classifier_model": "",
                    "drafter_model": "",
                }
            )
        print(f"[predict] {name}: {len(preds)} rows")

    if not args.baselines_only:
        client = LLMClient(offline=args.offline)
        problems = {} if args.offline else client.validate_roster()
        if problems:
            print("[predict] ROSTER PROBLEMS:")
            for prov, msgs in problems.items():
                for m in msgs:
                    print(f"  {prov}: {m}")
            if not args.ignore_roster:
                raise SystemExit(
                    "Configured models are unavailable. Fix config/models.yaml "
                    "or pass --ignore-roster to continue with the rest."
                )
        rows += run_agent(golden, client, index, system_name="agent")
        print("[predict] budget:", json.dumps(client.budget_report(), indent=2))

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PRED_PATH, index=False)
    print(f"[predict] wrote {len(df)} rows -> {PRED_PATH}")


def stage_judge(args) -> None:
    golden = pd.read_parquet(GOLDEN).set_index("golden_id")
    preds = pd.read_parquet(PRED_PATH)
    index = ResolutionIndex.load()
    client = LLMClient(offline=args.offline)

    judges = {"primary": client.config["roles"]["judge_primary"]}
    if not args.primary_only:
        judges["cross_family"] = client.config["roles"]["judge_cross_family"]
        if args.with_ceiling:
            judges["ceiling"] = client.config["roles"]["judge_ceiling"]

    # The primary judge can also be scoped. A full run is 210 rows x 4 systems
    # = 840 calls, which free-tier quota often cannot absorb in one day. Scoping
    # by GOLDEN ROW (not by prediction row) keeps all systems comparable on
    # exactly the same messages - sampling predictions directly would score
    # different systems on different messages and make the comparison invalid.
    judged_ids = None
    if args.judge_sample and golden.shape[0] > args.judge_sample:
        judged_ids = set(
            golden.sample(n=args.judge_sample, random_state=20260910).index
        )
        print(
            f"[judge] scoping to {len(judged_ids)} golden rows x "
            f"{preds['system'].nunique()} systems "
            f"= {len(judged_ids) * preds['system'].nunique()} calls"
        )

    rows = []
    for judge_name, model_key in judges.items():
        judge = ReplyJudge(client, model_key=model_key)
        subset = preds if judged_ids is None else preds[preds["golden_id"].isin(judged_ids)]
        if judge_name != "primary":
            # Secondary judges run on a capped, deterministic subset. Groq's
            # free tier allows ~171 judge calls/day, so scoring all 840 is not
            # possible - and is not needed: the self-preference delta and the
            # human-agreement study only require overlap with the primary
            # judge, not full coverage.
            ids_file = Path(args.agreement_ids) if args.agreement_ids else AGREEMENT_IDS
            if ids_file.exists():
                # Cover exactly the rows the human is scoring, so judge-vs-human
                # agreement has overlap to compute on.
                keep = set(ids_file.read_text(encoding="utf-8").split())
                subset = preds[preds["golden_id"].isin(keep)]
            elif args.secondary_sample and len(preds) > args.secondary_sample:
                # Stratified by system so every system gets equal coverage -
                # a plain random draw leaves too few rows per system to compare.
                per = max(1, args.secondary_sample // preds["system"].nunique())
                subset = (
                    preds.groupby("system", group_keys=False)
                    .apply(lambda g: g.sample(n=min(per, len(g)), random_state=20260910))
                    .sort_values(["system", "golden_id"])
                )
            print(
                f"[judge] {judge_name}: scoring {len(subset)} of {len(preds)} "
                "(capped by free-tier token budget)"
            )

        for r in tqdm(list(subset.itertuples()), desc=f"judge:{judge_name}"):
            g = golden.loc[r.golden_id]
            ex = index.search(g["customer_text"], k=3, before=g["customer_at"])
            v = judge.judge(g["customer_text"], r.reply, g["agent_text"], ex)
            rows.append(
                {
                    "judge": judge_name,
                    "judge_model": model_key,
                    "system": r.system,
                    "golden_id": r.golden_id,
                    **v.scores,
                    "send_as_is": v.send_as_is,
                    "rationale": v.rationale,
                    "parse_failed": v.parse_failed,
                    "prompt_tokens": v.prompt_tokens,
                    "completion_tokens": v.completion_tokens,
                }
            )

    df = pd.DataFrame(rows)
    df.to_parquet(JUDGE_PATH, index=False)
    print(f"[judge] wrote {len(df)} verdicts -> {JUDGE_PATH}")
    print("[judge] budget:", json.dumps(client.budget_report(), indent=2))


def stage_metrics(args) -> None:
    golden = pd.read_parquet(GOLDEN)
    preds = pd.read_parquet(PRED_PATH)

    summary = []
    for system, grp in preds.groupby("system"):
        merged = golden.merge(
            grp, on="golden_id", suffixes=("_true", "_pred")
        ).sort_values("golden_id")
        res = evaluate_system(
            merged.rename(columns={"intent_true": "intent", "route_true": "route"}),
            merged["intent_pred"].tolist(),
            merged["route_pred"].tolist(),
            name=str(system),
        )
        summary.append(res)

        rep = intent_report(
            merged["intent_true"].tolist(), merged["intent_pred"].tolist()
        )
        rep.to_csv(RESULTS / f"intent_report_{system}.csv", index=False)

    (RESULTS / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    table = pd.DataFrame(
        [
            {
                "system": s["system"],
                "intent_macro_f1": round(s["intent_macro_f1"]["point"], 3),
                "f1_ci": f"[{s['intent_macro_f1']['ci_lo']:.2f},{s['intent_macro_f1']['ci_hi']:.2f}]",
                "intent_acc": round(s["intent_accuracy_raw"]["point"], 3),
                "auto_rate": round(s["headline_safe_auto"]["auto_handled_rate"]["point"], 3),
                "harmful_auto": round(s["headline_safe_auto"]["harmful_auto_rate"]["point"], 3),
                "within_budget": s["headline_safe_auto"]["within_budget"],
                "safe_auto_rate": round(s["headline_safe_auto"]["safe_auto_rate"], 3),
            }
            for s in summary
        ]
    ).sort_values("safe_auto_rate", ascending=False)

    table.to_csv(RESULTS / "summary_table.csv", index=False)
    print(table.to_string(index=False))
    print(f"\n[metrics] wrote {RESULTS/'metrics.json'} and summary_table.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["predict", "judge", "metrics", "all"])
    ap.add_argument("--offline", action="store_true",
                    help="serve only from the committed cache; fail on any miss")
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--primary-only", action="store_true")
    ap.add_argument("--with-ceiling", action="store_true")
    ap.add_argument("--ignore-roster", action="store_true")
    ap.add_argument("--judge-sample", type=int, default=0,
                    help="limit judging to N golden rows (0 = all); all systems "
                         "are scored on the same rows so they stay comparable")
    ap.add_argument("--secondary-sample", type=int, default=150,
                    help="cap on rows scored by non-primary judges")
    ap.add_argument("--agreement-ids", default=None,
                    help="file of golden_ids the secondary judges should score")
    args = ap.parse_args()

    if args.stage in ("predict", "all"):
        stage_predict(args)
    if args.stage in ("judge", "all"):
        stage_judge(args)
    if args.stage in ("metrics", "all"):
        stage_metrics(args)


if __name__ == "__main__":
    main()
