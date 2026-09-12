"""LLM-as-judge for reply quality.

The judge is the load-bearing component of this evaluation, so the things that
make judges untrustworthy are addressed explicitly rather than hoped away:

1. POSITION AND IDENTITY BIAS. The judge never learns which system wrote a
   reply. Replies arrive anonymised, and the harness randomises order.

2. SELF-PREFERENCE. A model judging its own family's output scores it
   generously. The same replies are therefore scored by a cross-family judge
   (Groq gpt-oss-120b vs Gemini drafts) and the delta is reported as a result.

3. THE REFERENCE REPLY IS EVIDENCE, NOT AN ANSWER KEY. The judge sees what the
   brand actually said, but is told explicitly that the historical reply is not
   necessarily good. 47% of this brand's real replies are channel handoffs; a judge treating them as the target would reward imitation of a
   non-answer. This instruction is the difference between measuring quality and
   measuring mimicry.

4. SCORES ARE USELESS WITHOUT AGREEMENT EVIDENCE. Nothing here is trustworthy
   until src/eval/agreement.py has compared it to human labels; the judge
   reports its own numbers, and agreement.py decides whether to believe them.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.text import normalise  # noqa: E402
from llm.client import LLMClient  # noqa: E402

RUBRIC = ROOT / "config" / "rubric.yaml"


@dataclass
class Verdict:
    scores: dict[str, int] = field(default_factory=dict)
    send_as_is: str = "no"
    rationale: str = ""
    parse_failed: bool = False
    raw: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def mean_score(self) -> float:
        return sum(self.scores.values()) / len(self.scores) if self.scores else 0.0


def render_rubric(rubric: dict) -> str:
    parts = []
    for d in rubric["dimensions"]:
        q = " ".join(d["question"].split())
        parts.append(f"{d['id']} ({d['name']}): {q}")
        for score, desc in sorted(d["scale"].items()):
            parts.append(f"    {score} = {desc}")
    return "\n".join(parts)


JUDGE_PROMPT = """You are evaluating a draft customer-support reply for \
{brand} on Twitter. Be a strict, fair reviewer.

=== THE CUSTOMER WROTE ===
{message}

=== HOW THIS BRAND HANDLED SIMILAR ISSUES BEFORE ===
{exemplars}

=== WHAT THE BRAND ACTUALLY REPLIED TO THIS MESSAGE ===
{reference}

IMPORTANT: the actual reply above is what happened, NOT a model answer. This \
brand answers with a channel handoff ("Send us a DM and we'll take a closer \
look") in 47% of cases, which resolves nothing publicly. Do not treat matching \
it as automatically good, and do not treat differing from it as automatically bad.

=== THE DRAFT REPLY UNDER REVIEW ===
{candidate}

=== RUBRIC ===
{rubric}

=== CALIBRATION ===
{calibration}

=== THE DECISION THAT MATTERS ===
{binary_question}

Score each dimension 1-5, then answer the send/no-send question.

Reply with JSON only:
{{"grounded": <1-5>, "actionable": <1-5>, "safe": <1-5>, "tone": <1-5>, \
"send_as_is": "yes"|"no", "rationale": "<one sentence, max 25 words>"}}"""


class ReplyJudge:
    def __init__(
        self,
        client: LLMClient,
        model_key: str | None = None,
        brand: str = "Apple",
        rubric_path: Path = RUBRIC,
    ):
        self.client = client
        self.model_key = model_key or client.config["roles"]["judge_primary"]
        self.brand = brand
        self.rubric = yaml.safe_load(rubric_path.read_text(encoding="utf-8"))
        self.rendered = render_rubric(self.rubric)
        self.calibration = "\n".join(
            f"- {c}" for c in self.rubric.get("calibration_notes", [])
        )
        self.dimensions = [d["id"] for d in self.rubric["dimensions"]]
        self.binary_question = " ".join(
            self.rubric["binary"]["question"].split()
        )

    def build_prompt(
        self,
        message: str,
        candidate: str,
        reference: str,
        exemplars: list | None = None,
    ) -> str:
        if exemplars:
            ex_text = "\n\n".join(
                f"Customer: {e.customer_text[:200]}\nBrand: {e.agent_text[:200]}"
                for e in exemplars[:3]
            )
        else:
            ex_text = "(no similar past exchange was retrieved)"

        return JUDGE_PROMPT.format(
            brand=self.brand,
            message=normalise(message)[:500],
            exemplars=ex_text,
            reference=normalise(reference)[:400],
            candidate=candidate[:500] or "(empty reply)",
            rubric=self.rendered,
            calibration=self.calibration,
            binary_question=self.binary_question,
        )

    def judge(
        self,
        message: str,
        candidate: str,
        reference: str,
        exemplars: list | None = None,
    ) -> Verdict:
        # An empty draft is a failure, not something to spend a judge call on.
        if not candidate.strip():
            return Verdict(
                scores={d: 1 for d in self.dimensions},
                send_as_is="no",
                rationale="Empty reply - drafting failed.",
            )

        prompt = self.build_prompt(message, candidate, reference, exemplars)
        try:
            data, resp = self.client.complete_json(
                self.model_key, prompt, max_tokens=250,
                required_keys=("send_as_is",),
            )
        except ValueError as e:
            return Verdict(parse_failed=True, raw=f"PARSE_FAIL: {e}")

        scores = {}
        for d in self.dimensions:
            try:
                scores[d] = max(1, min(5, int(round(float(data.get(d, 3))))))
            except (TypeError, ValueError):
                scores[d] = 3

        send = str(data.get("send_as_is", "no")).strip().lower()
        if send not in ("yes", "no"):
            send = "yes" if send.startswith("y") else "no"

        return Verdict(
            scores=scores,
            send_as_is=send,
            rationale=str(data.get("rationale", ""))[:200],
            raw=resp.text,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
        )
