"""Grounded reply drafting.

The reply is written from retrieved historical resolutions, not from the
model's general knowledge of music streaming. That distinction is the whole
claim of the system, so the prompt is built to make ungrounded output visibly
wrong rather than merely discouraged:

  * Retrieved exemplars are presented as the ONLY sanctioned source of
    product-specific claims.
  * When retrieval returns nothing, the prompt says so explicitly and asks for
    a safe holding reply. A model that invents troubleshooting steps with no
    precedent is failing, and we want that failure visible in the judge scores
    rather than smoothed over by a generic fallback.
  * The exemplar ids are returned alongside the draft, so the judge and a human
    reader can both check the draft against the evidence that produced it.

The prompt also encodes what Spotify's own replies look like - short, one
concrete next step, no invented policy - derived from reading the corpus, not
from generic "be helpful" instructions.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agent.retrieve import Exemplar, ResolutionIndex  # noqa: E402
from data.text import normalise  # noqa: E402
from llm.client import LLMClient  # noqa: E402


@dataclass
class Draft:
    reply: str
    exemplar_ids: tuple[str, ...] = ()
    grounded: bool = True          # False when no precedent was retrieved
    raw: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


GROUNDED_PROMPT = """You are a customer support agent for {brand} replying \
publicly on Twitter.

Below are real past exchanges where this brand handled a similar issue. They \
are your ONLY source for product-specific claims.

{exemplars}

Write a reply to this new customer message:
\"\"\"{message}\"\"\"

The customer's issue has been classified as: {intent}

Requirements:
- Under 280 characters. Twitter is the medium.
- Give ONE concrete next step, drawn from how the past exchanges above handled \
this. Do not stack multiple instructions.
- Only state product facts that appear in the examples above. If you do not \
know something, ask for it rather than guessing.
- Never invent policy, timelines, refunds, or compensation.
- Never ask for a password, payment details, or full account numbers in public.
- Match the brand's voice in the examples: warm, brief, no corporate padding.

Reply with JSON only:
{{"reply": "<your reply text>"}}"""


UNGROUNDED_PROMPT = """You are a customer support agent for {brand} replying \
publicly on Twitter.

NO SIMILAR PAST EXCHANGE WAS FOUND for this message. You therefore have no \
precedent for how this brand handles it.

Customer message:
\"\"\"{message}\"\"\"

Classified as: {intent}

Because you have no precedent, write a SAFE HOLDING REPLY that:
- Acknowledges the specific issue the customer raised.
- Asks one clarifying question OR says a human will follow up.
- Invents NOTHING: no troubleshooting steps, no policy, no timelines.
- Stays under 280 characters.

Reply with JSON only:
{{"reply": "<your reply text>"}}"""


def render_exemplars(exemplars: list[Exemplar]) -> str:
    parts = []
    for i, e in enumerate(exemplars, 1):
        parts.append(
            f"[Example {i}]\n"
            f"Customer: {e.customer_text[:280]}\n"
            f"{'Brand'}: {e.agent_text[:280]}"
        )
    return "\n\n".join(parts)


class ReplyDrafter:
    def __init__(
        self,
        client: LLMClient,
        index: ResolutionIndex,
        model_key: str | None = None,
        brand: str = "Spotify",
        k: int = 4,
    ):
        self.client = client
        self.index = index
        self.model_key = model_key or client.config["roles"]["drafter"]
        self.brand = brand
        self.k = k

    def build_prompt(
        self, message: str, intent: str, exemplars: list[Exemplar]
    ) -> str:
        msg = normalise(message)[:600]
        if exemplars:
            return GROUNDED_PROMPT.format(
                brand=self.brand,
                exemplars=render_exemplars(exemplars),
                message=msg,
                intent=intent,
            )
        return UNGROUNDED_PROMPT.format(
            brand=self.brand, message=msg, intent=intent
        )

    def draft(
        self,
        message: str,
        intent: str,
        before: pd.Timestamp | None = None,
    ) -> Draft:
        exemplars = self.index.search(message, k=self.k, before=before)
        prompt = self.build_prompt(message, intent, exemplars)
        ids = tuple(e.pair_id for e in exemplars)

        try:
            data, resp = self.client.complete_json(
                self.model_key, prompt, max_tokens=300, required_keys=("reply",)
            )
        except ValueError as e:
            return Draft(
                reply="",
                exemplar_ids=ids,
                grounded=bool(exemplars),
                raw=f"PARSE_FAIL: {e}",
            )

        reply = str(data.get("reply", "")).strip()
        return Draft(
            reply=reply,
            exemplar_ids=ids,
            grounded=bool(exemplars),
            raw=resp.text,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
        )
