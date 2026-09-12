"""Provider adapters for the two free tiers we use.

Kept deliberately thin: one `complete()` per provider that takes a prompt and
returns an LLMResponse with real token accounting. Everything shared - caching,
rate limiting, retries, JSON repair - lives in client.py so the two providers
cannot drift apart in behaviour.

Model IDs drift between doc revisions, so `validate_roster()` checks the
configured models against each provider's live list-models endpoint at startup
and fails with the available names rather than 404-ing mid-run.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod

from .cache import LLMResponse


class ProviderError(RuntimeError):
    """Non-retryable provider failure (bad key, unknown model, refusal)."""


class RetryableError(RuntimeError):
    """Transient failure - rate limit or 5xx. Safe to back off and retry."""


class Provider(ABC):
    name: str

    @abstractmethod
    def complete(
        self, model: str, prompt: str, temperature: float, max_tokens: int
    ) -> LLMResponse: ...

    @abstractmethod
    def list_models(self) -> list[str]: ...

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimate for rate-limit reservation.

        ~4 chars/token is the standard English approximation. We only need this
        to be close enough to reserve budget; client.py reconciles with the
        provider's real usage numbers afterwards.
        """
        return max(1, len(text) // 4)


class GeminiProvider(Provider):
    name = "gemini"

    # Headroom for thinking tokens that thinking_budget=0 does not suppress.
    # ~300 observed on gemini-3.7-flash, but gemini-3.6-flash blew through 812
    # on the drafting prompt, so a single fixed reserve is not enough. We start
    # here and escalate on truncation (see complete()). Unused output tokens
    # are not billed, so a generous ceiling costs nothing.
    THINKING_RESERVE = 1024
    MAX_ESCALATIONS = 3

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ProviderError(
                "GEMINI_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey and put it in .env"
            )
        from google import genai

        self._genai = genai
        self.client = genai.Client(api_key=key)
        # model_id -> whether it accepts thinking_config. Probed on first use.
        self._supports_thinking_cfg: dict[str, bool] = {}

    def list_models(self) -> list[str]:
        out = []
        for m in self.client.models.list():
            name = getattr(m, "name", "") or ""
            # API returns "models/gemini-3.8-flash"; callers use the bare id.
            out.append(name.split("/", 1)[-1] if "/" in name else name)
        return out

    @staticmethod
    def _unwrap(exc: BaseException) -> BaseException:
        """Dig the real API error out of the SDK's retry wrapper.

        google-genai retries internally via tenacity and then raises
        `RetryError[<Future ... raised ServerError>]`. That wrapper is NOT an
        APIError subclass, so an `except APIError` handler misses it entirely
        and a transient 5xx escapes as an uncaught exception mid-run. The
        status code we need to classify retryable-vs-fatal is only on the
        wrapped cause.
        """
        seen = set()
        cur: BaseException | None = exc
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            if hasattr(cur, "last_attempt"):          # tenacity.RetryError
                try:
                    nxt = cur.last_attempt.exception()
                except Exception:
                    nxt = None
            else:
                nxt = cur.__cause__ or cur.__context__
            if nxt is None:
                return cur
            cur = nxt
        return exc

    # Transient failures that carry no HTTP status code. A dropped TCP
    # connection has no `.code`, so a status-code-only classifier files it as
    # fatal and kills the run - which is exactly what a WinError 10054 did
    # partway through a 210-row prediction.
    _TRANSIENT_TYPES = (
        ConnectionError,        # covers ConnectionResetError/AbortedError
        TimeoutError,
        OSError,                # socket-level failures on Windows
    )
    _TRANSIENT_MARKERS = (
        "connection reset", "forcibly closed", "connection aborted",
        "timed out", "timeout", "ssl", "remote end closed",
        "temporarily unavailable",
    )

    def _classify(self, exc: BaseException) -> Exception:
        """Map a raw SDK exception to Retryable vs fatal, via the real cause."""
        root = self._unwrap(exc)
        code = getattr(root, "code", None) or getattr(root, "status_code", None)
        text = f"{type(root).__name__}: {root}"

        if code in (429, 500, 502, 503, 504) or "ServerError" in type(root).__name__:
            return RetryableError(f"gemini {code}: {text[:200]}")
        if isinstance(root, self._TRANSIENT_TYPES) or any(
            m in text.lower() for m in self._TRANSIENT_MARKERS
        ):
            return RetryableError(f"gemini transient: {text[:200]}")
        return ProviderError(f"gemini {code}: {text[:300]}")

    def complete(
        self, model: str, prompt: str, temperature: float, max_tokens: int
    ) -> LLMResponse:
        # GEMINI THINKING TOKENS: THREE SEPARATE TRAPS, ALL SILENT.
        #
        # (1) Thinking is drawn from the SAME max_output_tokens budget as the
        #     answer, and on these prompts it consumes nearly all of it:
        #       classifier prompt, max=100 -> 92 thinking / 4 answer
        #                                     text was the fragment: format?
        #       classifier prompt, max=300 -> 287 thinking / 9 answer
        #                                     text was a truncated JSON opener
        #     Both return HTTP 200 with finish_reason=MAX_TOKENS and truncated
        #     garbage rather than an error - so every classification would have
        #     fallen back to `other` at zero confidence, and the run would have
        #     looked like a model-quality problem, not a config bug.
        #
        # (2) thinking_budget=0 is not universally supported. gemini-3.5-flash-lite
        #     rejects thinking_config outright with a 400 (and does no thinking
        #     anyway), so support is probed per model and memoised rather than
        #     hardcoded.
        #
        # (3) Where it IS accepted, it may be IGNORED. gemini-3.7-flash with
        #     thinking_budget=0 still spent ~296 thinking tokens:
        #       drafter prompt, max=300 -> 296 thinking / 7 answer, TRUNCATED
        #       drafter prompt, max=600 -> 298 thinking / 40 answer, complete
        #
        # Since the budget cannot be relied on, the robust remedy is headroom:
        # a fixed reserve added on top of whatever the caller asked for, so
        # callers size max_tokens for the ANSWER without knowing any of this.
        reserve = self.THINKING_RESERVE
        t0 = time.perf_counter()

        # ESCALATE ON TRUNCATION RATHER THAN RETRY IT.
        # Truncation is deterministic: the same prompt at the same ceiling
        # truncates every time. Treating it as a transient error meant each
        # draft burned five identical retries with exponential backoff before
        # failing over - which is exactly why the first full run appeared to
        # hang while making no progress. Doubling the ceiling is the only retry
        # that can actually succeed.
        for attempt in range(self.MAX_ESCALATIONS):
            effective_max = max_tokens + reserve
            base = {"temperature": temperature, "max_output_tokens": effective_max}
            use_thinking_cfg = self._supports_thinking_cfg.get(model, True)
            config = dict(base)
            if use_thinking_cfg:
                config["thinking_config"] = {"thinking_budget": 0}

            try:
                resp = self.client.models.generate_content(
                    model=model, contents=prompt, config=config
                )
            except Exception as first:
                root = self._unwrap(first)
                code = getattr(root, "code", None) or getattr(root, "status_code", None)
                if use_thinking_cfg and code == 400:
                    # This model does not accept thinking_config. Remember, retry.
                    self._supports_thinking_cfg[model] = False
                    try:
                        resp = self.client.models.generate_content(
                            model=model, contents=prompt, config=base
                        )
                    except Exception as e:
                        raise self._classify(e) from e
                else:
                    raise self._classify(first) from first

            cand = resp.candidates[0] if getattr(resp, "candidates", None) else None
            finish = str(getattr(cand, "finish_reason", "") or "")
            if "MAX_TOKENS" not in finish:
                break
            reserve *= 4
        else:
            # Still truncated after escalating. Not retryable - the caller
            # should fail over to a different model instead of spinning.
            raise ProviderError(
                f"gemini {model}: still truncated at max_output_tokens="
                f"{max_tokens + reserve // 4} after {self.MAX_ESCALATIONS} "
                "escalations. This model spends too much of the budget on "
                "thinking for this prompt."
            )

        elapsed = time.perf_counter() - t0
        text = resp.text or ""
        usage = getattr(resp, "usage_metadata", None)

        # THINKING TOKENS ARE BILLED AS OUTPUT but are NOT included in
        # candidates_token_count. Counting only the answer under-reports real
        # output cost badly - which would have quietly made Gemini look far
        # cheaper than it is in the arena's cost table.
        answer_tokens = getattr(usage, "candidates_token_count", 0) or 0
        thinking_tokens = getattr(usage, "thoughts_token_count", 0) or 0

        return LLMResponse(
            text=text,
            model=model,
            provider=self.name,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            completion_tokens=answer_tokens + thinking_tokens,
            latency_s=round(elapsed, 3),
        )


class GroqProvider(Provider):
    name = "groq"

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ProviderError(
                "GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com and put it in .env"
            )
        import groq

        self._groq = groq
        self.client = groq.Groq(api_key=key)

    def list_models(self) -> list[str]:
        return [m.id for m in self.client.models.list().data]

    # gpt-oss models are REASONING models. They emit chain-of-thought into a
    # separate `reasoning` field and only fill `content` once that finishes.
    # At default effort a short answer burned 95 completion tokens and, with a
    # small max_tokens, returned finish_reason="length" and an EMPTY content
    # string - a silent failure that would have emptied every cross-family
    # judge verdict without raising anything.
    #
    # reasoning_effort="low" fixes both problems: content is populated, and
    # completion tokens drop ~3x (29 vs 95), which matters directly against
    # Groq's 200K tokens/day free-tier budget.
    REASONING_MODELS = ("gpt-oss", "qwen")
    REASONING_FLOOR = 1024
    MAX_ESCALATIONS = 3

    def _is_reasoning(self, model: str) -> bool:
        return any(m in model for m in self.REASONING_MODELS)

    def complete(
        self, model: str, prompt: str, temperature: float, max_tokens: int
    ) -> LLMResponse:
        """Complete, escalating the ceiling if reasoning eats the whole budget.

        Same failure shape as Gemini, different mechanism: gpt-oss writes its
        chain-of-thought into `message.reasoning` and only fills `content`
        afterwards. If the ceiling is reached first, the call returns HTTP 200
        with finish_reason="length" and an EMPTY content string.

        A fixed floor of 512 was not enough - gpt-oss-20b exceeded it on the
        classifier prompt - and because truncation is deterministic, treating it
        as retryable just burned five identical attempts. So the ceiling is
        escalated instead, exactly as on the Gemini side.
        """
        reasoning = self._is_reasoning(model)
        budget = max(max_tokens, self.REASONING_FLOOR) if reasoning else max_tokens

        t0 = time.perf_counter()
        for _ in range(self.MAX_ESCALATIONS):
            kwargs = {}
            if "gpt-oss" in model:
                # Qwen does not accept reasoning_effort; it emits inline
                # <think> blocks instead (stripped in client._extract_json).
                kwargs["reasoning_effort"] = "low"
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=budget,
                    **kwargs,
                )
            except self._groq.RateLimitError as e:
                raise RetryableError(f"groq 429: {e}") from e
            except (self._groq.APIConnectionError, self._groq.InternalServerError) as e:
                raise RetryableError(f"groq transient: {e}") from e
            except self._groq.APIStatusError as e:
                raise ProviderError(f"groq {e.status_code}: {e}") from e

            choice = resp.choices[0]
            text = choice.message.content or ""
            if text.strip() or choice.finish_reason != "length":
                break
            budget *= 4
        else:
            # Not retryable - the caller should fail over to another model
            # rather than spin on a deterministic truncation.
            raise ProviderError(
                f"groq {model}: still truncated mid-reasoning at "
                f"max_tokens={budget // 4} after {self.MAX_ESCALATIONS} "
                "escalations; content stayed empty."
            )

        elapsed = time.perf_counter() - t0
        usage = resp.usage
        return LLMResponse(
            text=text,
            model=model,
            provider=self.name,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            # Groq's completion_tokens already includes reasoning tokens, so
            # no separate adjustment is needed here (unlike Gemini).
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_s=round(elapsed, 3),
        )


_REGISTRY: dict[str, type[Provider]] = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
}


def get_provider(name: str, **kwargs) -> Provider:
    if name not in _REGISTRY:
        raise ProviderError(
            f"Unknown provider {name!r}. Known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)
