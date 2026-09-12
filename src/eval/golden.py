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

import os
BRAND = os.environ.get("BRAND", "AppleSupport")
PAIRS = ROOT / "data" / "processed" / f"pairs_{BRAND}.parquet"
TAXONOMY = ROOT / "config" / "taxonomy.yaml"
OUT_DIR = ROOT / "golden"

N_RANDOM = 80
N_STRATIFIED = 130
SEED = 20260910

# Model-independent provisional labels for stratification only. These are NOT
# ground truth and never enter evaluation - they exist to spread the sample
# across the taxonomy so every class is measurable.
PROVISIONAL_RULES: dict[str, re.Pattern] = {
    "hardware_repair": re.compile(
        r"\b(crack|cracked|broken screen|repair|replacement|warranty|applecare|"
        r"genius bar|trade.?in|water damage|shattered|how much (would|does|is))\b", re.I),
    "account_appleid": re.compile(
        r"\b(apple ?id|icloud|itunes account|app store|two.?factor|verification code|"
        r"password|sign ?in|signed out|storage (full|plan)|subscription|"
        r"purchase|receipt|billed|charged)\b", re.I),
    "battery_power": re.compile(
        r"\b(battery|charging|charge|drain|draining|dies|died|shut ?(down|off)|"
        r"powers? off|percent)\b", re.I),
    "connectivity": re.compile(
        r"\b(wi.?fi|bluetooth|cellular|airplay|hotspot|airpods|carplay|"
        r"pairing|paired|signal|lte|network)\b", re.I),
    "app_software": re.compile(
        r"\b(imessage|messages app|mail app|safari|itunes|apple music|photos app|"
        r"keyboard|autocorrect|question mark|emoji|siri|facetime|"
        r"notes|calendar|pages|numbers)\b", re.I),
    "os_update": re.compile(
        r"\b(ios ?1?[0-9]|update|updated|updating|upgrade|high sierra|"
        r"downgrade|new version|latest version|software update)\b", re.I),
    "device_performance": re.compile(
        r"\b(freez|crash|slow|lag|unresponsive|restart|reboot|stuck|"
        r"not working|glitch)\b", re.I),
    "feedback_complaint": re.compile(
        r"\b(fix (this|your|it)|garbage|trash|rubbish|worst|terrible|"
        r"ridiculous|disappointed|sucks)\b", re.I),
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

    Repair and Apple ID come first because misrouting those is the expensive
    error: a message mentioning both "cracked screen" and "iOS 11" belongs in
    hardware_repair, and one mentioning both "charged" and "update" belongs in
    account_appleid. Cheap-to-misroute classes are matched last.
    """
    for intent in (
        "hardware_repair",
        "account_appleid",
        "battery_power",
        "connectivity",
        "app_software",
        "device_performance",
        "os_update",
        "feedback_complaint",
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
    out = OUT_DIR / f"golden_pool_{BRAND}.parquet"
    g.to_parquet(out, index=False)

    print(f"[golden] n = {len(g)}")
    print(f"[golden] strata:\n{g['stratum'].value_counts().to_string()}")
    print(f"\n[golden] provisional intent spread:")
    print(g.groupby(["provisional_intent", "stratum"]).size().unstack(fill_value=0).to_string())
    print(f"\n[golden] hard cases: {g['is_hard'].sum()} ({100*g['is_hard'].mean():.0f}%)")
    flat = [f for flags in g["hard_flags"] for f in flags]
    print(f"[golden] hard-signal breakdown: {pd.Series(flat).value_counts().to_dict()}")
    print(f"\n[golden] wrote {out}")
