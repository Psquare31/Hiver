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

    def list_models(self) -> list[str]:
        out = []
        for m in self.client.models.list():
            name = getattr(m, "name", "") or ""
            # API returns "models/gemini-3.8-flash"; callers use the bare id.
            out.append(name.split("/", 1)[-1] if "/" in name else name)
        return out

    def complete(
        self, model: str, prompt: str, temperature: float, max_tokens: int
    ) -> LLMResponse:
        from google.genai import errors as genai_errors

        t0 = time.perf_counter()
        try:
            resp = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
        except genai_errors.APIError as e:
            code = getattr(e, "code", None)
            if code in (429, 500, 502, 503, 504):
                raise RetryableError(f"gemini {code}: {e}") from e
            raise ProviderError(f"gemini {code}: {e}") from e

        elapsed = time.perf_counter() - t0
        text = resp.text or ""
        usage = getattr(resp, "usage_metadata", None)
        return LLMResponse(
            text=text,
            model=model,
            provider=self.name,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
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

    def complete(
        self, model: str, prompt: str, temperature: float, max_tokens: int
    ) -> LLMResponse:
        t0 = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except self._groq.RateLimitError as e:
            raise RetryableError(f"groq 429: {e}") from e
        except (self._groq.APIConnectionError, self._groq.InternalServerError) as e:
            raise RetryableError(f"groq transient: {e}") from e
        except self._groq.APIStatusError as e:
            raise ProviderError(f"groq {e.status_code}: {e}") from e

        elapsed = time.perf_counter() - t0
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            model=model,
            provider=self.name,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
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
