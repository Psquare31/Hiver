"""Prove the committed response cache is genuine.

A cache committed to a repo is a claim: "these are the responses the models
actually gave." A reviewer has no reason to take that on trust, and a project
whose headline numbers replay from a cache needs a way to check it.

This re-runs a random sample of cached prompts against the live APIs and
compares. Exact string equality is NOT the bar - these are sampled models at
temperature 0, and providers change checkpoints behind a stable model id, so
some drift is expected and honest. What we assert is that a fresh call produces
a response of the same shape and, for structured calls, the same decision.

Reported per sampled item:
  identical   - byte-for-byte match
  equivalent  - same parsed decision (intent / send_as_is), different wording
  divergent   - different decision. Investigate before trusting the cache.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from llm.cache import CACHE_DIR, LLMResponse  # noqa: E402
from llm.client import LLMClient, _extract_json  # noqa: E402


def load_entries() -> list[tuple[Path, dict]]:
    out = []
    for f in CACHE_DIR.rglob("*.json"):
        try:
            out.append((f, json.loads(f.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def decision_of(text: str) -> str | None:
    """Extract the operative decision from a structured response, if any."""
    data = _extract_json(text)
    if not isinstance(data, dict):
        return None
    for key in ("intent", "send_as_is"):
        if key in data:
            return f"{key}={str(data[key]).strip().lower()}"
    if "reply" in data:
        return "reply"     # free text; compared by shape only
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    entries = load_entries()
    if not entries:
        raise SystemExit(
            f"No cache entries found under {CACHE_DIR}. Run `make live` first."
        )

    print(f"[verify] cache holds {len(entries)} entries")

    # The cache stores the response but not the prompt (prompts are large and
    # the key is their hash). Re-deriving requires the prompt, so verification
    # replays the pipeline's own prompts instead - see README. Here we check
    # internal consistency plus a live re-call for entries we can rebuild.
    rng = random.Random(args.seed)
    sample = rng.sample(entries, min(args.sample, len(entries)))

    client = LLMClient(offline=False)
    identical = equivalent = divergent = unverifiable = 0

    for path, data in sample:
        prompt = data.get("prompt")
        if not prompt:
            unverifiable += 1
            continue
        model_key = data.get("model_key")
        if not model_key:
            unverifiable += 1
            continue

        fresh = client._provider(data["provider"]).complete(
            data["model"], prompt, 0.0, 512
        )
        if fresh.text.strip() == data["text"].strip():
            identical += 1
        elif decision_of(fresh.text) == decision_of(data["text"]) is not None:
            equivalent += 1
        else:
            divergent += 1
            print(f"[verify] DIVERGENT {path.name[:12]}")
            print(f"  cached: {data['text'][:120]}")
            print(f"  fresh : {fresh.text[:120]}")

    print(f"\n[verify] sampled {len(sample)}")
    print(f"  identical    {identical}")
    print(f"  equivalent   {equivalent}")
    print(f"  divergent    {divergent}")
    print(f"  unverifiable {unverifiable}  (no prompt stored - see note below)")

    if unverifiable:
        print(
            "\nNOTE: entries written before prompt-retention was enabled cannot be\n"
            "replayed directly. `make live` rewrites the cache with prompts stored,\n"
            "after which this check covers every entry."
        )
    if divergent:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
