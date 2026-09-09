"""Text normalisation shared by every stage.

Kept in one place so the retrieval index, the classifier prompt, and the
evaluation all see identically-shaped text. Divergence here is a classic
source of silent train/serve skew.
"""

from __future__ import annotations

import html
import re

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
# Support agents sign off with initials: "/DR", "^GK", "- AM", "~JS"
SIGNATURE_RE = re.compile(r"\s*[\^/~]\s?[A-Za-z]{1,3}\s*$|\s+-\s?[A-Z]{2,3}\s*$")
WS_RE = re.compile(r"\s+")


def normalise(
    text: str,
    *,
    strip_urls: bool = True,
    strip_mentions: bool = True,
    strip_signature: bool = True,
) -> str:
    """Canonical cleanup: unescape entities, drop noise, collapse whitespace.

    HTML entities are real in this corpus (1,361 Spotify replies contain
    &gt;/&amp;), and a raw "&gt;" reads as garbage to both a classifier and a
    human annotator.
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    if strip_urls:
        text = URL_RE.sub(" ", text)
    if strip_mentions:
        text = MENTION_RE.sub(" ", text)
    if strip_signature:
        text = SIGNATURE_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


def for_display(text: str) -> str:
    """Normalisation for anything a human reads (labelling sheet, report).

    Keeps @mentions out but preserves the sentence as written, including
    URLs collapsed to a placeholder so annotators can see a link was present.
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = URL_RE.sub("[link]", text)
    text = MENTION_RE.sub("", text)
    text = SIGNATURE_RE.sub("", text)
    return WS_RE.sub(" ", text).strip()
