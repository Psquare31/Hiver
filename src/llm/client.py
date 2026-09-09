"""The single entry point every LLM call in this project goes through.

Responsibilities, in the order they apply to a request:

  cache lookup -> rate-limit reservation -> provider call -> retry on transient
  -> token reconciliation -> cache write

Centralising this matters for the evaluation more than for the engineering. If
the classifier and the judge reached providers by different paths, they would
accumulate different retry behaviour and different token accounting, and the
cost comparison in the report would be measuring the plumbing rather than the
models.

OFFLINE MODE is the default for `make reproduce`: with ANY cache miss the run
fails loudly rather than silently hitting the network, so a reviewer without
keys gets either the exact published numbers or a clear error - never a
half-live run that quietly differs from the report.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .cache import LLMResponse, ResponseCache, cache_key
from .ratelimit import Limits, RateLimiter
from .providers import Provider, ProviderError, RetryableError, get_provider

ROOT = Path(__file__).resolve().parents[2]
MODELS_YAML = ROOT / "config" / "models.yaml"

load_dotenv(ROOT / ".env")


class CacheMissError(RuntimeError):
    """Raised in offline mode when a needed response is not in the cache."""


@dataclass
class ModelSpec:
    """One row of the model arena."""

    key: str            # short name used in configs and result tables
    provider: str
    model_id: str
    role: str           # classify | draft | judge | ceiling
    input_price: float  # USD per 1M tokens, published paid rate
    output_price: float

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Equivalent cost at published paid rates.

        We run on free tiers, so actual spend is zero. Reporting zero would make
        the arena useless - the point is which model gives the most accuracy per
        token. Published rates make the comparison portable to anyone paying.
        """
        return (
            prompt_tokens * self.input_price
            + completion_tokens * self.output_price
        ) / 1_000_000


class LLMClient:
    def __init__(
        self,
        config_path: Path | None = None,
        offline: bool | None = None,
        cache_dir: Path | None = None,
    ):
        self.config = yaml.safe_load(
            (config_path or MODELS_YAML).read_text(encoding="utf-8")
        )
        self.models: dict[str, ModelSpec] = {}
        for key, spec in self.config["models"].items():
            self.models[key] = ModelSpec(
                key=key,
                provider=spec["provider"],
                model_id=spec["model_id"],
                role=spec.get("role", "classify"),
                input_price=float(spec.get("input_price_per_1m", 0.0)),
                output_price=float(spec.get("output_price_per_1m", 0.0)),
            )

        self.cache = ResponseCache(cache_dir)
        if offline is None:
            offline = os.environ.get("LLM_OFFLINE", "0") == "1"
        self.offline = offline

        self._providers: dict[str, Provider] = {}
        self._limiters: dict[str, RateLimiter] = {}
        # Limit scope differs by provider and getting it wrong is expensive in
        # both directions. Groq publishes limits PER MODEL, so a provider-wide
        # limiter would throttle the arena to roughly one model per day for no
        # reason. Gemini's daily request cap is account-wide, so per-model
        # limiters there would overshoot the real quota and earn 429s.
        for name, cfg in self.config["providers"].items():
            self._limiters[name] = RateLimiter(
                name,
                Limits(
                    rpm=cfg.get("rpm"),
                    tpm=cfg.get("tpm"),
                    rpd=cfg.get("rpd"),
                    tpd=cfg.get("tpd"),
                ),
            )
        self._limit_scope = {
            name: cfg.get("limit_scope", "provider")
            for name, cfg in self.config["providers"].items()
        }

        self.retry_max = int(self.config.get("retry", {}).get("max_attempts", 5))
        self.retry_base = float(self.config.get("retry", {}).get("base_delay_s", 2.0))

    def _limiter_for(self, spec: ModelSpec) -> RateLimiter:
        """Return the limiter governing this model, creating per-model ones lazily."""
        if self._limit_scope.get(spec.provider, "provider") != "model":
            return self._limiters[spec.provider]

        key = f"{spec.provider}:{spec.key}"
        if key not in self._limiters:
            cfg = self.config["providers"][spec.provider]
            self._limiters[key] = RateLimiter(
                key.replace("/", "_").replace(":", "_"),
                Limits(
                    rpm=cfg.get("rpm"),
                    tpm=cfg.get("tpm"),
                    rpd=cfg.get("rpd"),
                    tpd=cfg.get("tpd"),
                ),
            )
        return self._limiters[key]

    # -- provider access (lazy: never construct a client we do not use) ----

    def _provider(self, name: str) -> Provider:
        if name not in self._providers:
            self._providers[name] = get_provider(name)
        return self._providers[name]

    def validate_roster(self) -> dict[str, list[str]]:
        """Check every configured model exists at its provider.

        Model IDs drift between doc revisions. Failing here, with the live list
        in the message, is far cheaper than discovering it 300 calls into a run.
        """
        problems: dict[str, list[str]] = {}
        by_provider: dict[str, list[ModelSpec]] = {}
        for spec in self.models.values():
            by_provider.setdefault(spec.provider, []).append(spec)

        for provider_name, specs in by_provider.items():
            try:
                available = set(self._provider(provider_name).list_models())
            except ProviderError as e:
                problems[provider_name] = [f"cannot reach provider: {e}"]
                continue
            missing = [s.model_id for s in specs if s.model_id not in available]
            if missing:
                problems[provider_name] = [
                    f"configured but not available: {missing}",
                    f"available: {sorted(available)}",
                ]
        return problems

    # -- the call ----------------------------------------------------------

    def complete(
        self,
        model_key: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> LLMResponse:
        if model_key not in self.models:
            raise ValueError(
                f"Unknown model key {model_key!r}. Configured: {sorted(self.models)}"
            )
        spec = self.models[model_key]
        params = {"temperature": temperature, "max_tokens": max_tokens}
        key = cache_key(spec.provider, spec.model_id, prompt, params)

        hit = self.cache.get(key)
        if hit is not None:
            return hit

        if self.offline:
            raise CacheMissError(
                f"Cache miss for {model_key} in offline mode (key {key[:12]}...). "
                "Run `make live` with API keys to populate, or check that the "
                "prompt/params match the committed run exactly."
            )

        limiter = self._limiter_for(spec)
        provider = self._provider(spec.provider)
        est = provider.estimate_tokens(prompt) + max_tokens

        last_err: Exception | None = None
        for attempt in range(self.retry_max):
            limiter.acquire(est_tokens=est)
            try:
                resp = provider.complete(
                    spec.model_id, prompt, temperature, max_tokens
                )
            except RetryableError as e:
                last_err = e
                # Full jitter: synchronised retries across a batch are what turn
                # one 429 into a sustained thundering herd against a free tier.
                delay = min(60.0, self.retry_base * (2**attempt))
                time.sleep(random.uniform(0, delay))
                continue
            except ProviderError:
                raise

            limiter.reconcile(est, resp.total_tokens or est)
            # Retain the prompt so `make verify-cache` can re-issue this exact
            # call later; a cache nobody can audit cannot back a headline number.
            resp.prompt = prompt
            resp.model_key = model_key
            self.cache.put(key, resp)
            return resp

        raise RuntimeError(
            f"{model_key}: exhausted {self.retry_max} attempts; last error: {last_err}"
        )

    def complete_json(
        self,
        model_key: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        required_keys: tuple[str, ...] = (),
    ) -> tuple[dict[str, Any], LLMResponse]:
        """Completion parsed as JSON, tolerant of the usual model formatting.

        Small models wrap JSON in prose or fences far more often than large
        ones. Repairing that here rather than in each caller keeps the arena
        fair: we are comparing intent accuracy, not markdown discipline.
        """
        resp = self.complete(
            model_key, prompt, temperature=temperature, max_tokens=max_tokens
        )
        data = _extract_json(resp.text)
        if data is None:
            raise ValueError(
                f"{model_key} returned unparseable JSON: {resp.text[:200]!r}"
            )
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise ValueError(
                f"{model_key} JSON missing keys {missing}: {resp.text[:200]!r}"
            )
        return data, resp

    def budget_report(self) -> dict[str, Any]:
        return {
            "cache": self.cache.stats(),
            "remaining_today": {
                name: lim.remaining_today() for name, lim in self._limiters.items()
            },
        }


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction from a model response."""
    if not text:
        return None
    candidates = [text]
    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.insert(0, fenced.group(1))
    # Fall back to the outermost brace pair.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for c in candidates:
        try:
            parsed = json.loads(c.strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None
