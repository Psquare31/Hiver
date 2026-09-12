"""Tests for the parts where a silent bug would corrupt the reported numbers.

Priority is not code coverage. It is the specific failures that would produce a
plausible-looking but wrong result: metric arithmetic, leakage guards, cache key
collisions, rate-limit accounting, and the regex boundaries that already bit
once during development.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "golden"))


# --------------------------------------------------------------------------
# Substance classifier - the metric that chose the brand
# --------------------------------------------------------------------------

def test_substance_classifier_matches_hand_labels():
    from data.brand import has_substance
    from fixtures_replies import LABELLED

    wrong = [(t, e, has_substance(t)) for t, e in LABELLED if has_substance(t) != e]
    assert not wrong, f"{len(wrong)} fixture(s) misclassified: {wrong[:2]}"


def test_credential_request_is_not_substance():
    """The single most consequential false positive on this corpus."""
    from data.brand import has_substance

    assert not has_substance(
        "Hey there! Can you DM us your account's email address or username? "
        "We'll take a look backstage"
    )


def test_diagnostic_probe_is_substance():
    from data.brand import has_substance

    assert has_substance(
        "Can you let us know what device, operating system, and Spotify "
        "version you're using?"
    )


# --------------------------------------------------------------------------
# Regex boundaries - these broke twice during development
# --------------------------------------------------------------------------

def test_escalation_triggers_respect_word_boundaries():
    """'sue' must not match inside 'issue'."""
    from agent.route import Router

    r = Router()
    d = r.route("I have an issue with my battery", "battery_power", 0.9)
    assert d.route == "auto", f"'issue' wrongly triggered {d.rule_id}"

    d = r.route("I will sue you over this", "other", 0.9)
    assert d.route == "escalate" and d.rule_id == "legal_or_regulatory"


def test_safety_hazard_rule_fires_and_outranks_intent_prior():
    """A swollen battery must never get an automated troubleshooting reply,
    even though battery_power's prior is auto."""
    from agent.route import Router

    r = Router()
    d = r.route("my battery is swollen and bulging", "battery_power", 0.99)
    assert d.route == "escalate" and d.rule_id == "safety_hazard"


def test_known_rule_gaps_are_still_gaps():
    """Documented in golden/labels_author.py and reported as a failure mode.

    These are phrasings a human escalates and the regexes miss. The test pins
    the gap so it cannot silently change without the report changing too.
    """
    from agent.route import Router

    r = Router()
    for phrasing in [
        "our laptop is practically on fire after 20 mins",   # list has "caught fire"
        "I got an electric shock from your headphones",      # absent entirely
        "it is not supposed to inflate like this",           # list has "swelling"
    ]:
        d = r.route(phrasing, "device_performance", 0.9)
        assert d.rule_id != "safety_hazard", (
            f"Gap closed for {phrasing!r} - update the report's failure analysis"
        )


def test_shouting_detector_is_case_sensitive():
    """[A-Z]{6,} under re.I matches any six-letter word - it flagged 72%
    of the corpus before this was caught."""
    from eval.golden import HARD_SIGNALS

    rx = HARD_SIGNALS["shouting"]
    assert rx.search("this is TERRIBLE")
    assert not rx.search("this is terrible")


def test_no_corrupt_backspace_in_source():
    """A \\b in a patched regex once became a literal 0x08 byte, silently
    disabling every word boundary in the file."""
    for f in (ROOT / "src").rglob("*.py"):
        assert chr(8) not in f.read_text(encoding="utf-8"), f"corrupt byte in {f}"


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def test_cache_key_is_order_independent_but_value_sensitive():
    from llm.cache import cache_key

    a = cache_key("p", "m", "prompt", {"temperature": 0.0, "max_tokens": 10})
    b = cache_key("p", "m", "prompt", {"max_tokens": 10, "temperature": 0.0})
    c = cache_key("p", "m", "prompt", {"temperature": 0.1, "max_tokens": 10})
    d = cache_key("p", "m", "prompt2", {"temperature": 0.0, "max_tokens": 10})
    assert a == b
    assert a != c and a != d


def test_cache_roundtrip_and_corruption_is_a_miss():
    from llm.cache import LLMResponse, ResponseCache, cache_key

    with tempfile.TemporaryDirectory() as td:
        cache = ResponseCache(Path(td))
        k = cache_key("p", "m", "hello", {})
        assert cache.get(k) is None

        cache.put(k, LLMResponse(text="hi", model="m", provider="p",
                                 prompt_tokens=5, completion_tokens=2))
        got = cache.get(k)
        assert got is not None and got.text == "hi" and got.cached is True
        assert got.total_tokens == 7

        # A truncated file must behave as a miss, not raise.
        cache._path(k).write_text("{not json", encoding="utf-8")
        assert cache.get(k) is None


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------

def test_daily_token_budget_raises_rather_than_hanging():
    from llm.ratelimit import Limits, RateLimiter

    with tempfile.TemporaryDirectory() as td:
        rl = RateLimiter("t", Limits(tpd=5000), state_dir=Path(td))
        rl.acquire(4000)
        with pytest.raises(RuntimeError, match="daily token budget"):
            rl.acquire(2000)


def test_daily_budget_survives_process_restart():
    """Without persistence a restart silently resets the counter and blows
    the real quota, which on Groq means a lockout until UTC midnight."""
    from llm.ratelimit import Limits, RateLimiter

    with tempfile.TemporaryDirectory() as td:
        a = RateLimiter("persist", Limits(tpd=10000), state_dir=Path(td))
        a.acquire(6000)
        b = RateLimiter("persist", Limits(tpd=10000), state_dir=Path(td))
        assert b.remaining_today()["tokens"] == 4000


def test_reconcile_corrects_estimate_drift():
    from llm.ratelimit import Limits, RateLimiter

    with tempfile.TemporaryDirectory() as td:
        rl = RateLimiter("t", Limits(tpd=10000), state_dir=Path(td))
        rl.acquire(1000)
        rl.reconcile(1000, 1500)          # actual cost was higher
        assert rl.remaining_today()["tokens"] == 8500


# --------------------------------------------------------------------------
# Metrics - checked against values computed by hand
# --------------------------------------------------------------------------

def test_routing_metrics_against_hand_computed_values():
    from eval.metrics import routing_metrics

    truth = ["escalate", "escalate", "auto", "auto", "auto"]
    pred = ["escalate", "auto", "auto", "escalate", "auto"]
    #   harmful auto (truth=esc, pred=auto)      = 1/5 = 0.2
    #   unnecessary escalation (truth=auto,pred=esc) = 1/5 = 0.2
    #   escalation recall  = 1 of 2 = 0.5
    #   escalation precision = 1 of 2 = 0.5
    r = routing_metrics(truth, pred)
    assert r.harmful_auto_rate.point == pytest.approx(0.2)
    assert r.unnecessary_escalation_rate.point == pytest.approx(0.2)
    assert r.escalation_recall.point == pytest.approx(0.5)
    assert r.escalation_precision.point == pytest.approx(0.5)


def test_safe_auto_rate_zeroes_out_when_over_harm_budget():
    from eval.metrics import safe_auto_rate

    # Auto-handles everything, including 2 that needed a human -> 40% harmful.
    truth = ["escalate", "escalate", "auto", "auto", "auto"]
    pred = ["auto"] * 5
    res = safe_auto_rate(truth, pred, harm_budget=0.05)
    assert res["auto_handled_rate"]["point"] == pytest.approx(1.0)
    assert res["within_budget"] is False
    assert res["safe_auto_rate"] == 0.0


def test_bootstrap_ci_brackets_the_point_estimate():
    from eval.metrics import bootstrap_ci

    vals = np.array([1.0] * 70 + [0.0] * 30)
    ci = bootstrap_ci(vals)
    assert ci.point == pytest.approx(0.7)
    assert ci.lo < 0.7 < ci.hi
    assert (ci.hi - ci.lo) < 0.25   # n=100 should not be wildly wide


# --------------------------------------------------------------------------
# Agreement
# --------------------------------------------------------------------------

def test_kappa_endpoints():
    from eval.agreement import cohens_kappa

    a = ["yes", "no"] * 20
    assert cohens_kappa(a, a) == pytest.approx(1.0)
    # Perfectly inverted -> strongly negative
    inv = ["no" if x == "yes" else "yes" for x in a]
    assert cohens_kappa(a, inv) < -0.9


def test_judge_vs_human_joins_on_system_not_just_id():
    """Two systems answer the same golden_id. Joining on id alone would
    compare a human's score for one system's reply against another's."""
    from eval.agreement import judge_vs_human

    judge = pd.DataFrame({
        "golden_id": ["g1", "g1"],
        "system": ["agent", "trivial"],
        "send_as_is": ["yes", "no"],
    })
    human = pd.DataFrame({
        "golden_id": ["g1", "g1"],
        "system": ["agent", "trivial"],
        "send_as_is": ["yes", "no"],
    })
    out = judge_vs_human(judge, human)
    assert out["n_compared"] == 2
    assert out["binary"]["raw_agreement"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Leakage - the guarantees the report depends on
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    not (ROOT / "data" / "processed" / "bm25_index_AppleSupport.pkl").exists(),
    reason="index not built",
)
def test_no_golden_thread_appears_in_retrieval_index():
    from agent.retrieve import ResolutionIndex

    idx = ResolutionIndex.load()
    golden = pd.read_parquet(ROOT / "golden" / "golden_labelled_AppleSupport.parquet")
    assert not set(idx.df["pair_id"]) & set(golden["pair_id"])


@pytest.mark.skipif(
    not (ROOT / "data" / "processed" / "bm25_index_AppleSupport.pkl").exists(),
    reason="index not built",
)
def test_retrieved_exemplars_always_predate_their_query():
    from agent.retrieve import ResolutionIndex

    idx = ResolutionIndex.load()
    golden = pd.read_parquet(ROOT / "golden" / "golden_labelled_AppleSupport.parquet")
    for _, row in golden.head(40).iterrows():
        for e in idx.search(row["customer_text"], k=4, before=row["customer_at"]):
            assert e.customer_at < row["customer_at"]


# --------------------------------------------------------------------------
# Golden set integrity
# --------------------------------------------------------------------------

def test_golden_labels_are_complete_and_valid():
    import yaml
    from labels_author import LABELS

    golden = pd.read_parquet(ROOT / "golden" / "golden_labelled_AppleSupport.parquet")
    tax = yaml.safe_load((ROOT / "config" / "taxonomy.yaml").read_text(encoding="utf-8"))
    valid = {i["id"] for i in tax["intents"]}

    assert set(LABELS) == set(golden["golden_id"])
    assert all(v[0] in valid for v in LABELS.values())
    assert all(v[1] in ("auto", "escalate") for v in LABELS.values())


def test_golden_set_size_is_within_brief():
    golden = pd.read_parquet(ROOT / "golden" / "golden_labelled_AppleSupport.parquet")
    assert 150 <= len(golden) <= 250


def test_other_class_stays_under_coverage_threshold():
    """The taxonomy file commits to <15% `other`. If this fails, the taxonomy
    is missing a class rather than the classifier being wrong."""
    golden = pd.read_parquet(ROOT / "golden" / "golden_labelled_AppleSupport.parquet")
    assert (golden["intent"] == "other").mean() < 0.15


# --------------------------------------------------------------------------
# Failover - load-bearing now that free-tier quota is unpredictable
# --------------------------------------------------------------------------

def _client_with_stub(monkeypatch, behaviour):
    """LLMClient whose providers are replaced by a scripted stub.

    `behaviour` maps model_id -> either an Exception to raise or a text to
    return, so a test can make one model fail and assert the chain moves on.
    """
    import tempfile
    from llm import client as client_mod
    from llm.cache import LLMResponse

    class StubProvider:
        name = "stub"

        @staticmethod
        def estimate_tokens(text):
            return max(1, len(text) // 4)

        def complete(self, model, prompt, temperature, max_tokens):
            outcome = behaviour[model]
            if isinstance(outcome, Exception):
                raise outcome
            return LLMResponse(text=outcome, model=model, provider="stub")

    c = client_mod.LLMClient(offline=False, cache_dir=Path(tempfile.mkdtemp()))
    stub = StubProvider()
    monkeypatch.setattr(c, "_provider", lambda name: stub)
    # No real waiting in tests.
    monkeypatch.setattr(c, "retry_max", 1)
    for lim in c._limiters.values():
        monkeypatch.setattr(lim, "acquire", lambda **kw: None)
        monkeypatch.setattr(lim, "reconcile", lambda *a, **k: None)
    return c


def test_failover_moves_on_after_quota_exhaustion(monkeypatch):
    from llm.providers import RetryableError

    c = _client_with_stub(monkeypatch, {
        "gemini-3.7-flash": RetryableError("gemini 429: RESOURCE_EXHAUSTED"),
        "gemini-3.6-flash": '{"ok": true}',
    })
    monkeypatch.setattr(
        c, "_limiter_for", lambda spec: list(c._limiters.values())[0]
    )
    resp = c.complete("gemini_flash", "hi", max_tokens=50)
    assert resp.model == "gemini-3.6-flash"


def test_failover_also_moves_on_503(monkeypatch):
    """A 503 is about this model's availability, not the request. An earlier
    version only failed over on 429 and a transient overload killed a full run."""
    from llm.providers import RetryableError

    c = _client_with_stub(monkeypatch, {
        "gemini-3.7-flash": RetryableError("gemini 503: UNAVAILABLE high demand"),
        "gemini-3.6-flash": '{"ok": true}',
    })
    monkeypatch.setattr(
        c, "_limiter_for", lambda spec: list(c._limiters.values())[0]
    )
    assert c.complete("gemini_flash", "hi", max_tokens=50).model == "gemini-3.6-flash"


def test_failover_does_not_mask_a_real_error(monkeypatch):
    """A bad request fails identically on every model, so it must propagate
    rather than burn the whole chain."""
    from llm.providers import ProviderError

    c = _client_with_stub(monkeypatch, {
        "gemini-3.7-flash": ProviderError("gemini 400: INVALID_ARGUMENT"),
        "gemini-3.6-flash": '{"ok": true}',
    })
    monkeypatch.setattr(
        c, "_limiter_for", lambda spec: list(c._limiters.values())[0]
    )
    with pytest.raises(ProviderError, match="400"):
        c.complete("gemini_flash", "hi", max_tokens=50)
