"""Auto-handle vs escalate, with a stated reason.

DESIGN: RULES FIRST, MODEL SECOND, AND THE RULES ONLY EVER ESCALATE.

The cost here is asymmetric. Auto-handling a compromised account or a refund
dispute is a serious failure; escalating a question a bot could have answered
costs an agent two minutes. So the deterministic rules are one-directional -
they can force an escalation, never force an auto-handle. A model that is
overconfident about a security incident cannot override them.

The confidence floor exists for the same reason: an uncertain classification is
itself evidence that a human should look. If the classifier cannot tell what
the customer wants, the reply is being drafted for the wrong intent.

Every decision carries a `reason` string. An escalation a human cannot explain
is one they cannot act on, and "the model said so" is not triage.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.text import normalise  # noqa: E402

TAXONOMY = ROOT / "config" / "taxonomy.yaml"


@dataclass
class RoutingDecision:
    route: str          # "auto" | "escalate"
    reason: str         # human-readable justification
    rule_id: str = ""   # which rule fired, if any
    confidence: float = 0.0


class Router:
    def __init__(self, taxonomy_path: Path = TAXONOMY):
        tax = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8"))
        self.priors = {i["id"]: i["default_route"] for i in tax["intents"]}
        self.threshold = float(tax.get("low_confidence_threshold", 0.55))

        # Compile trigger phrases per rule. Word boundaries matter: without
        # them "sue" matches inside "issue" and "harm" inside "pharmacy",
        # which would fire the safety rule on ordinary complaints.
        self.rules = []
        for rule in tax.get("escalation_rules", []):
            pattern = "|".join(re.escape(t) for t in rule["triggers"])
            self.rules.append(
                {
                    "id": rule["id"],
                    "regex": re.compile(rf"\b({pattern})\b", re.IGNORECASE),
                    "reason": rule["reason"],
                }
            )

    def route(
        self, message: str, intent: str, confidence: float = 1.0
    ) -> RoutingDecision:
        body = normalise(message)

        # 1. Hard rules. One-directional: these can only escalate.
        for rule in self.rules:
            if rule["regex"].search(body):
                return RoutingDecision(
                    route="escalate",
                    reason=rule["reason"],
                    rule_id=rule["id"],
                    confidence=confidence,
                )

        # 2. Confidence floor. Uncertainty about intent is itself a reason to
        #    involve a human, because the reply is being written for a guess.
        if confidence < self.threshold:
            return RoutingDecision(
                route="escalate",
                reason=(
                    f"Intent confidence {confidence:.2f} is below the "
                    f"{self.threshold:.2f} threshold - classification is "
                    "unreliable, so any drafted reply may address the wrong issue."
                ),
                rule_id="low_confidence",
                confidence=confidence,
            )

        # 3. Taxonomy prior for the predicted intent.
        prior = self.priors.get(intent, "escalate")
        if prior == "escalate":
            return RoutingDecision(
                route="escalate",
                reason=(
                    f"Intent '{intent}' requires account-specific action or "
                    "human judgement by policy."
                ),
                rule_id="intent_prior",
                confidence=confidence,
            )

        return RoutingDecision(
            route="auto",
            reason=(
                f"Intent '{intent}' is handled by standard guidance and no "
                "escalation trigger was present."
            ),
            rule_id="intent_prior",
            confidence=confidence,
        )


if __name__ == "__main__":
    r = Router()
    cases = [
        ("my account was hacked and someone changed my email", "account_access", 0.95),
        ("can I get a refund for last month", "billing_charge", 0.9),
        ("how do I make a playlist collaborative", "playlist_library", 0.95),
        ("the app keeps crashing", "playback_failure", 0.92),
        ("something weird is going on", "other", 0.30),
        ("I have an issue with my playlist", "playlist_library", 0.9),
    ]
    for msg, intent, conf in cases:
        d = r.route(msg, intent, conf)
        print(f"{d.route:9} [{d.rule_id:15}] {msg[:46]:48} {d.reason[:52]}")
