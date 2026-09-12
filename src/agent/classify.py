"""Intent classification.

The prompt is built from config/taxonomy.yaml rather than hand-written, so the
taxonomy has exactly one definition and the classifier cannot drift from the
document the human annotator used. Change the YAML and every model in the arena
sees the change on the next run.

TOKEN BUDGET IS A DESIGN CONSTRAINT, NOT AN AFTERTHOUGHT. Groq's free tier
allows 200K tokens/day per model. A prompt carrying every positive and
negative example for ten intents runs ~1,220 tokens; the compact rendering
below runs ~580, which is what lets a 210-row run fit alongside the judge. `verbose=True` restores the full examples for the ablation that measures
what that compression costs.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.text import normalise  # noqa: E402
from llm.client import LLMClient  # noqa: E402

TAXONOMY = ROOT / "config" / "taxonomy.yaml"


@dataclass
class Classification:
    intent: str
    confidence: float
    raw: str = ""
    # Which model actually answered. May differ from the requested model when
    # the client fails over on quota exhaustion, so results are never
    # attributed to a model that did not produce them.
    served_by: str = ""


def load_taxonomy(path: Path = TAXONOMY) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def render_taxonomy(tax: dict, verbose: bool = False) -> str:
    """Render intents for the prompt.

    Compact mode gives id + a one-line definition. Verbose adds the worked
    examples. Both come from the same YAML.
    """
    lines = []
    for i in tax["intents"]:
        definition = " ".join(i["definition"].split())
        if not verbose:
            # First sentence only - enough to separate the classes.
            definition = definition.split(". ")[0].rstrip(".") + "."
        lines.append(f"- {i['id']}: {definition}")
        if verbose:
            for ex in i.get("positive_examples", [])[:2]:
                lines.append(f"    YES: {ex}")
            for ex in i.get("negative_examples", [])[:1]:
                lines.append(f"    NO:  {ex}")
    return "\n".join(lines)


PROMPT = """You are triaging inbound customer messages for {brand} \
({brand_description}) on Twitter.

Classify the message into exactly one intent:

{taxonomy}

Rules:
- Choose the intent describing what the customer WANTS, not their tone.
- A message can only have one intent. If it raises several, pick the one with \
the greatest consequence for the customer.
- Use `other` only when no class fits. Do not use it for messages written in \
languages other than English - classify those by their actual intent.
- confidence is your probability that a careful human annotator would agree, \
from 0.0 to 1.0. Be honest: low confidence is useful signal, not failure.

Customer message:
\"\"\"{message}\"\"\"

Reply with JSON only:
{{"intent": "<id>", "confidence": <0.0-1.0>}}"""


class IntentClassifier:
    def __init__(self, client: LLMClient, model_key: str | None = None,
                 verbose_taxonomy: bool = False):
        self.client = client
        self.model_key = model_key or client.config["roles"]["classifier"]
        self.tax = load_taxonomy()
        self.valid = {i["id"] for i in self.tax["intents"]}
        self.rendered = render_taxonomy(self.tax, verbose=verbose_taxonomy)
        self.fallback = "other"

    def build_prompt(self, message: str) -> str:
        return PROMPT.format(
            brand=self.tax["brand"],
            # Comes from the taxonomy so the brand and its description can
            # never drift apart. An earlier version hardcoded "a music
            # streaming service", which survived the switch to AppleSupport and
            # would have told the model the wrong thing on every message.
            brand_description=self.tax.get(
                "brand_description", "a consumer technology company"
            ),
            taxonomy=self.rendered,
            message=normalise(message)[:600],
        )

    def classify(self, message: str) -> Classification:
        prompt = self.build_prompt(message)
        try:
            data, resp = self.client.complete_json(
                self.model_key, prompt, max_tokens=100,
                required_keys=("intent",),
            )
        except ValueError as e:
            # A model that will not emit parseable JSON is a real failure mode
            # and must show up in the results, not be silently retried away.
            return Classification(self.fallback, 0.0, raw=f"PARSE_FAIL: {e}")

        intent = str(data.get("intent", "")).strip()
        if intent not in self.valid:
            # Hallucinated label. Recorded as `other` with zero confidence so it
            # is visible in the confusion matrix rather than hidden.
            return Classification(
                self.fallback, 0.0, raw=f"INVALID_LABEL: {intent!r}"
            )

        try:
            conf = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        return Classification(
            intent, max(0.0, min(1.0, conf)), raw=resp.text, served_by=resp.model
        )


if __name__ == "__main__":
    client = LLMClient()
    clf = IntentClassifier(client)
    p = clf.build_prompt("my songs keep pausing on my iphone after the ios update")
    print(p)
    print()
    print(f"[classify] compact prompt is ~{len(p)//4} tokens")
    verbose = IntentClassifier(client, verbose_taxonomy=True)
    pv = verbose.build_prompt("test")
    print(f"[classify] verbose prompt is ~{len(pv)//4} tokens")
