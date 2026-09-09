"""Brand selection, decided by measurement rather than by tweet volume.

The trap in this dataset is that volume and usability are inversely correlated.
The biggest support accounts reply almost entirely with handoffs - "share your
details here: <link>", "please DM us" - because the real resolution happens in
a private channel the dataset never captured. Grounding a reply generator on
that corpus means grounding it on non-answers, and a judge asking "does this
match how the brand historically resolved this" will happily reward a bot that
only ever says "please DM us".

TWO EARLIER VERSIONS OF THIS METRIC WERE WRONG, INSTRUCTIVELY SO:

1. Matching redirect phrases ("dm us", "click here") scored AmazonHelp at 0.6%
   handoff - apparently the most groundable brand in the corpus. Inspection
   showed the opposite: Amazon's handoffs read "Kindly share your details
   here: <url>", which no phrase list anticipated. Enumerating ways to say
   "go away" is unbounded; every brand invents its own.

2. Inverting to detect SUBSTANCE over-credited instead, because Spotify's
   canonical DM wall - "Can you DM us your account's email address?" - is
   phrased as a question about a product noun and scored as troubleshooting.

The version below requires a STRONG signal (an instruction or a probe for
technical state) and explicitly demotes credential requests. Tuned and pinned
against hand-labelled fixtures in tests/fixtures_replies.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "twcs.parquet"
OUT = ROOT / "results" / "brand_selection.csv"

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
SIGNATURE_RE = re.compile(r"[\^/~-]\s?[A-Z]{1,3}\s*$")

# STRONG substance means the reply does troubleshooting work that would still
# be useful to the *next* customer with the same problem: it names a step, or
# probes technical state. That is what a grounded draft can reuse.
INSTRUCTION_RE = re.compile(
    r"\b("
    r"try|restart|reinstall|re-install|uninstall|update|upgrade|"
    r"clear|toggle|enable|disable|turn(ed)? (on|off)|"
    r"log ?(out|in)|logging (out|in)|sign ?(out|in)|reset|refresh|reboot|"
    r"tap|go to|head to|select|press|swipe|"
    r"double-?check"
    r")\b",
    re.IGNORECASE,
)

# Probes for technical state. Deliberately excludes bare "app"/"platform",
# which fire on policy answers ("no change log for the Android app").
DIAGNOSTIC_RE = re.compile(
    r"\b("
    r"device|operating system|os x|\bos\b|"
    r"(app|software|spotify|ios|android) version|version (are|you)|"
    r"browser|incognito|private window|"
    r"error (message|code)|screenshot|"
    r"when did|how long|what happens|does (it|this) happen|"
    r"always been|started after|"
    r"wi-?fi|offline mode|settings"
    r")\b",
    re.IGNORECASE,
)

# Asking for identifiers in order to continue privately. This LOOKS like a
# question about a product noun, which is exactly why a naive matcher scores it
# as substance - the single biggest false-positive source on this corpus.
CREDENTIAL_RE = re.compile(
    r"\b("
    r"(account'?s? )?(email address|username|user name)|"
    r"order (number|id)|confirmation (number|code)|account number|"
    r"your details|postcode|zip code|phone number"
    r")\b",
    re.IGNORECASE,
)

# Pure channel-handoff: the whole message is "let's continue elsewhere".
HANDOFF_RE = re.compile(
    r"("
    r"\bdm us\b|\bdm'?d you\b|sent you a dm|sent a dm|dm your way|"
    r"send us a (dm|direct|private)|"
    r"\bdirect message\b|\bprivate message\b|carry on chatting|"
    r"share (your|the) details|send us your details|take a look backstage|"
    r"reach out to (our|the) team|contact (our|the) team|get in touch|"
    r"fill (out |in )?(this|the) form|complete (this|the) form|"
    r"click (the )?link|follow (the )?link|details here|here:"
    r")",
    re.IGNORECASE,
)


def clean_for_analysis(text: str) -> str:
    """Strip URLs, @mentions and agent initials so heuristics see only prose."""
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = SIGNATURE_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def has_substance(text: str) -> bool:
    """True if the reply contains troubleshooting content worth grounding on.

    Requires an instruction or a diagnostic probe. A bare credential request is
    excluded even though it is phrased as a question, because nothing in it
    generalises to the next customer with the same problem.
    """
    body = clean_for_analysis(text)
    if len(body.split()) < 4:
        return False

    # Judge substance on the text with credential asks removed, so that
    # "DM us your email" cannot masquerade as troubleshooting.
    residual = CREDENTIAL_RE.sub(" ", body)
    return bool(INSTRUCTION_RE.search(residual)) or bool(
        DIAGNOSTIC_RE.search(residual)
    )


def is_handoff(text: str) -> bool:
    """True if the reply's only content is moving the customer off-channel."""
    body = clean_for_analysis(text)
    had_url = bool(URL_RE.search(text))
    signals = (
        bool(HANDOFF_RE.search(body))
        or bool(CREDENTIAL_RE.search(body))
        or (had_url and len(body.split()) <= 25)
    )
    return signals and not has_substance(text)


def analyse(top_n: int = 15) -> pd.DataFrame:
    df = pd.read_parquet(RAW, columns=["author_id", "inbound", "text"])
    outbound = df[~df["inbound"].astype(bool)]
    counts = outbound["author_id"].value_counts().head(top_n)

    rows = []
    for brand, n in counts.items():
        replies = outbound.loc[outbound["author_id"] == brand, "text"].dropna().astype(str)
        subst = replies.map(has_substance)
        hand = replies.map(is_handoff)
        bodies = replies.map(clean_for_analysis)
        rows.append(
            {
                "brand": brand,
                "agent_tweets": int(n),
                "substantive_pct": round(100 * subst.mean(), 1),
                "handoff_pct": round(100 * hand.mean(), 1),
                "median_words": int(bodies.str.split().str.len().median()),
                "groundable_replies": int(subst.sum()),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("groundable_replies", ascending=False)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    res = analyse()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    pd.set_option("display.width", 140)
    print(res.to_string(index=False))
    print(f"\n[brand] wrote {OUT}")
