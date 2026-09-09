"""Build the golden evaluation set.

TWO DESIGN CHOICES THAT SHAPE EVERY NUMBER IN THE REPORT.

1. STRATIFICATION USES A MODEL-INDEPENDENT HEURISTIC.
   The obvious way to get a class-balanced golden set is to run the classifier
   over the corpus and sample per predicted intent. That quietly biases the
   evaluation: the set becomes "cases the classifier already sorts confidently",
   and rare or ambiguous messages - exactly where the system fails - are
   under-represented. So stratification uses keyword rules derived from the
   taxonomy, never the system under test.

2. THE SET IS DELIBERATELY NOT REPRESENTATIVE, AND CARRIES WEIGHTS TO FIX THAT.
   Two strata are drawn:
     * `random`   - uniform sample of real traffic. Representative, but a rare
                    intent may appear twice, which cannot support a per-class
                    metric.
     * `stratified` - balanced across intents plus deliberate oversampling of
                    hard cases, so per-class numbers are measurable at all.
   Metrics computed on the pooled set overstate difficulty (hard cases are
   over-represented). Every row therefore carries `weight`, letting the harness
   report both the raw number and one reweighted to the natural distribution.
   The gap between them is reported, not hidden.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.text import for_display, normalise  # noqa: E402

PAIRS = ROOT / "data" / "processed" / "pairs_SpotifyCares.parquet"
TAXONOMY = ROOT / "config" / "taxonomy.yaml"
OUT_DIR = ROOT / "golden"

N_RANDOM = 80
N_STRATIFIED = 130
SEED = 20260910

# Model-independent provisional labels for stratification only. These are NOT
# ground truth and never enter evaluation - they exist to spread the sample
# across the taxonomy so every class is measurable.
PROVISIONAL_RULES: dict[str, re.Pattern] = {
    "billing_charge": re.compile(
        r"\b(charged|charge|refund|billed|payment failed|double.?charg|"
        r"money back|invoice|overcharg)\b", re.I),
    "account_access": re.compile(
        r"\b(log ?in|login|log ?out|password|hacked|compromised|locked out|"
        r"can'?t access|sign ?in|username|my email|account.{0,15}(stolen|taken))\b", re.I),
    "subscription_manage": re.compile(
        r"\b(premium|family plan|student|subscription|cancel|upgrade|downgrade|"
        r"trial|promo|renew|hulu)\b", re.I),
    "content_missing": re.compile(
        r"\b(album|not available|unavailable|missing from|removed|region|"
        r"licens|why isn'?t .{0,25}on spotify|can'?t find .{0,20}(album|song|artist))\b", re.I),
    "device_compat": re.compile(
        r"\b(iphone|ipad|android|ios|windows|mac|macos|os x|smart ?tv|chromecast|"
        r"alexa|car ?play|web player|desktop|version \d)\b", re.I),
    "playlist_library": re.compile(
        r"\b(playlist|my library|saved songs|discover weekly|liked songs|"
        r"add.{0,15}songs?|remove.{0,15}songs?|sync)\b", re.I),
    "playback_failure": re.compile(
        r"\b(crash|freez|won'?t play|not playing|stops? playing|skipping|"
        r"buffering|keeps? pausing|no sound|cuts out|glitch)\b", re.I),
    "feature_request": re.compile(
        r"\b(please add|you should add|bring back|feature request|wish .{0,15}could|"
        r"would be (great|nice) if|why can'?t we)\b", re.I),
}

# Signals that a message is genuinely hard. Oversampled on purpose: a golden set
# of easy cases proves nothing about a system meant to decide autonomy.
HARD_SIGNALS: dict[str, re.Pattern] = {
    "escalation_adjacent": re.compile(
        r"\b(refund|hacked|compromised|lawyer|legal|sue|gdpr|third time|3rd time|"
        r"still waiting|weeks now|no one has (replied|responded)|fraud)\b", re.I),
    "multi_intent": re.compile(
        r"\b(also|and another thing|second (issue|problem)|two (issues|problems)|"
        r"plus,|on top of that)\b", re.I),
    "non_english": re.compile(
        r"[À-ɏЀ-ӿ؀-ۿ一-鿿]{3,}"),
    # Anger/sarcasm markers. NOTE: deliberately NOT re.I - the whole point of
    # the SHOUTING pattern is case. An earlier version applied re.I here, which
    # made [A-Z]{6,} match any six-letter word and flagged 72% of the corpus as
    # sarcastic. Case-sensitive patterns and case-insensitive ones cannot share
    # a compiled regex.
    "shouting": re.compile(r"\b[A-Z]{6,}\b|!{3,}"),
    "hostile_wording": re.compile(
        r"\b(worst|pathetic|useless|garbage|ridiculous|disgrace|scam|"
        r"thanks for nothing|absolute joke)\b", re.I),
}


def provisional_intent(text: str) -> str:
    """First matching rule wins, in a priority order that reflects routing cost.

    Billing and account come first because misrouting those is the expensive
    error; a message mentioning both "premium" and "charged twice" should land
    in billing.
    """
    for intent in (
        "billing_charge",
        "account_access",
        "playback_failure",
        "content_missing",
        "subscription_manage",
        "device_compat",
        "playlist_library",
        "feature_request",
    ):
        if PROVISIONAL_RULES[intent].search(text):
            return intent
    return "other"


def hard_flags(text: str) -> list[str]:
    return [name for name, rx in HARD_SIGNALS.items() if rx.search(text)]


def build(n_random: int = N_RANDOM, n_stratified: int = N_STRATIFIED,
          seed: int = SEED) -> pd.DataFrame:
    pairs = pd.read_parquet(PAIRS)
    taxonomy = yaml.safe_load(TAXONOMY.read_text(encoding="utf-8"))
    intent_ids = [i["id"] for i in taxonomy["intents"]]

    df = pairs.copy()
    df["customer_clean"] = df["customer_text"].map(normalise)
    # Drop messages too short to carry an intent - they would be labelled
    # `other` by any annotator and add noise, not signal.
    df = df[df["customer_clean"].str.split().str.len() >= 5].reset_index(drop=True)

    df["provisional_intent"] = df["customer_clean"].map(provisional_intent)
    df["hard_flags"] = df["customer_clean"].map(hard_flags)
    df["is_hard"] = df["hard_flags"].str.len() > 0

    natural = df["provisional_intent"].value_counts(normalize=True)

    rng_seed = seed
    # --- stratum A: uniform random (representative) ----------------------
    rand = df.sample(n=n_random, random_state=rng_seed).copy()
    rand["stratum"] = "random"

    # --- stratum B: balanced + hard-case oversample ----------------------
    remaining = df.drop(rand.index)
    per_class = max(1, round(n_stratified / (len(intent_ids) * 2)) + 2)
    picks = []
    for intent in intent_ids:
        pool = remaining[remaining["provisional_intent"] == intent]
        if pool.empty:
            continue
        hard_pool = pool[pool["is_hard"]]
        easy_pool = pool[~pool["is_hard"]]
        n_hard = min(len(hard_pool), per_class)
        n_easy = min(len(easy_pool), per_class)
        if n_hard:
            picks.append(hard_pool.sample(n=n_hard, random_state=rng_seed))
        if n_easy:
            picks.append(easy_pool.sample(n=n_easy, random_state=rng_seed))

    strat = pd.concat(picks) if picks else remaining.head(0)
    if len(strat) > n_stratified:
        strat = strat.sample(n=n_stratified, random_state=rng_seed)
    strat = strat.copy()
    strat["stratum"] = "stratified"

    golden = pd.concat([rand, strat], ignore_index=True)

    # --- sampling weights -------------------------------------------------
    # Weight = natural share / sample share, so a weighted metric estimates the
    # value on real traffic. Random-stratum rows are already representative.
    sample_share = golden["provisional_intent"].value_counts(normalize=True)
    golden["weight"] = golden.apply(
        lambda r: 1.0 if r["stratum"] == "random"
        else float(natural.get(r["provisional_intent"], 0.0))
        / float(sample_share.get(r["provisional_intent"], 1.0)),
        axis=1,
    )

    golden["customer_display"] = golden["customer_text"].map(for_display)
    golden["agent_display"] = golden["agent_text"].map(for_display)
    golden["golden_id"] = [f"g{i:03d}" for i in range(len(golden))]

    cols = [
        "golden_id", "stratum", "weight", "pair_id",
        "customer_tweet_id", "customer_text", "customer_display", "customer_clean",
        "agent_tweet_id", "agent_text", "agent_display",
        "customer_at", "response_lag_min",
        "provisional_intent", "hard_flags", "is_hard",
    ]
    return golden[cols].reset_index(drop=True)


if __name__ == "__main__":
    g = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "golden_pool.parquet"
    g.to_parquet(out, index=False)

    print(f"[golden] n = {len(g)}")
    print(f"[golden] strata:\n{g['stratum'].value_counts().to_string()}")
    print(f"\n[golden] provisional intent spread:")
    print(g.groupby(["provisional_intent", "stratum"]).size().unstack(fill_value=0).to_string())
    print(f"\n[golden] hard cases: {g['is_hard'].sum()} ({100*g['is_hard'].mean():.0f}%)")
    flat = [f for flags in g["hard_flags"] for f in flags]
    print(f"[golden] hard-signal breakdown: {pd.Series(flat).value_counts().to_dict()}")
    print(f"\n[golden] wrote {out}")
