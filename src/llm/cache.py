"""Content-addressed disk cache for LLM responses.

Why this exists: free-tier rate limits make a cold evaluation run ~100 minutes,
but the brief requires headline results to reproduce in under 15 minutes. The
cache is committed to the repo so `make reproduce` replays it offline. To keep
that honest rather than hand-wavy:

  * The key is SHA256 over every input that can change the output
    (provider, model, prompt, and all sampling params). Change any of them and
    you get a miss, not a stale hit.
  * Each entry records when it was created and how many tokens it cost, so the
    cost tables in the report are computed from real usage, not estimates.
  * `make verify-cache` re-runs a random sample live and diffs, which is what
    makes the committed cache auditable by a reviewer.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "cache" / "llm"


@dataclass
class LLMResponse:
    """One completion, plus the accounting the report needs."""

    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # The prompt is stored alongside the response even though the key is its
    # hash. Two reasons, both about trust rather than engineering:
    #   1. `make verify-cache` can only re-issue a call if it still has the
    #      prompt. Without this the committed cache is unauditable, and an
    #      unauditable cache behind the headline numbers is worthless.
    #   2. A reviewer can read exactly what each model was asked, which is
    #      usually the first thing anyone wants to check about an LLM result.
    # Costs roughly 4KB per entry; worth it.
    prompt: str = ""
    model_key: str = ""
    # Set when the response came from disk rather than the network. Never
    # serialised into the cache file itself - it is a property of the read.
    cached: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def cache_key(provider: str, model: str, prompt: str, params: dict[str, Any]) -> str:
    """Stable hash over everything that can change the output.

    params is sorted so that dict ordering never produces a spurious miss.
    """
    payload = json.dumps(
        {
            "provider": provider,
            "model": model,
            "prompt": prompt,
            "params": params,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    """Sharded file cache. Sharding by key prefix keeps directory sizes sane
    on Windows, where a single directory with 10k+ entries gets slow."""

    def __init__(self, root: Path | None = None):
        self.root = root or CACHE_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> LLMResponse | None:
        path = self._path(key)
        if not path.exists():
            with self._lock:
                self.misses += 1
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A truncated entry (interrupted write) should behave as a miss,
            # not crash a long run.
            with self._lock:
                self.misses += 1
            return None
        with self._lock:
            self.hits += 1
        data.pop("cached", None)
        return LLMResponse(**data, cached=True)

    def put(self, key: str, response: LLMResponse) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(response)
        data.pop("cached", None)
        # Atomic write: a crash mid-run must not leave a half-written entry
        # that later reads as corrupt.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(path)

    def stats(self) -> dict[str, int | float]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }

    def __len__(self) -> int:
        return sum(1 for _ in self.root.rglob("*.json"))
