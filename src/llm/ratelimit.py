"""Rate limiting for free-tier providers.

The non-obvious part: Groq's free tier is *token*-bound, not request-bound.
It allows 30 requests/minute but only 8K tokens/minute and 200K tokens/day.
A naive RPM-only limiter sails past the request check and then collects 429s
for the rest of the day. So we track four independent budgets - requests/min,
tokens/min, requests/day, tokens/day - and block on whichever binds first.

Daily budgets persist to disk. Without that, restarting the process would
silently reset the day counter and blow the real quota, which on Groq means
a hard lockout until midnight UTC rather than a soft backoff.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = ROOT / "cache" / "ratelimit"


@dataclass
class Limits:
    """A provider's free-tier ceiling. None means 'not enforced'."""

    rpm: int | None = None
    tpm: int | None = None
    rpd: int | None = None
    tpd: int | None = None


class RateLimiter:
    def __init__(self, name: str, limits: Limits, state_dir: Path | None = None):
        self.name = name
        self.limits = limits
        self._lock = threading.Lock()
        # (timestamp, tokens) within the trailing 60s window
        self._minute: deque[tuple[float, int]] = deque()
        self._state_dir = state_dir or STATE_DIR
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self._state_dir / f"{name}.json"
        self._day_key = ""
        self._day_requests = 0
        self._day_tokens = 0
        self._load_day()

    # ---- daily budget persistence -------------------------------------

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _load_day(self) -> None:
        today = self._today()
        if self._state_path.exists():
            try:
                data = json.loads(self._state_path.read_text(encoding="utf-8"))
                if data.get("day") == today:
                    self._day_key = today
                    self._day_requests = int(data.get("requests", 0))
                    self._day_tokens = int(data.get("tokens", 0))
                    return
            except (json.JSONDecodeError, OSError, ValueError):
                pass
        self._day_key = today
        self._day_requests = 0
        self._day_tokens = 0

    def _save_day(self) -> None:
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "day": self._day_key,
                    "requests": self._day_requests,
                    "tokens": self._day_tokens,
                }
            ),
            encoding="utf-8",
        )
        tmp.replace(self._state_path)

    def _roll_day_if_needed(self) -> None:
        today = self._today()
        if today != self._day_key:
            self._day_key = today
            self._day_requests = 0
            self._day_tokens = 0

    # ---- window maintenance -------------------------------------------

    def _prune(self, now: float) -> None:
        cutoff = now - 60.0
        while self._minute and self._minute[0][0] < cutoff:
            self._minute.popleft()

    def _minute_usage(self) -> tuple[int, int]:
        return len(self._minute), sum(t for _, t in self._minute)

    # ---- public API ----------------------------------------------------

    def remaining_today(self) -> dict[str, int | None]:
        with self._lock:
            self._roll_day_if_needed()
            return {
                "requests": (
                    None if self.limits.rpd is None
                    else max(0, self.limits.rpd - self._day_requests)
                ),
                "tokens": (
                    None if self.limits.tpd is None
                    else max(0, self.limits.tpd - self._day_tokens)
                ),
            }

    def acquire(self, est_tokens: int = 1000, timeout: float = 900.0) -> None:
        """Block until a request costing ~est_tokens can proceed.

        Raises RuntimeError if the *daily* budget is exhausted - unlike a
        per-minute stall, waiting that out could mean hours, so the caller
        needs to know rather than hang.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                self._roll_day_if_needed()
                now = time.monotonic()
                self._prune(now)

                if self.limits.rpd is not None and self._day_requests >= self.limits.rpd:
                    raise RuntimeError(
                        f"[{self.name}] daily request budget exhausted "
                        f"({self._day_requests}/{self.limits.rpd}). Resets 00:00 UTC."
                    )
                if (
                    self.limits.tpd is not None
                    and self._day_tokens + est_tokens > self.limits.tpd
                ):
                    raise RuntimeError(
                        f"[{self.name}] daily token budget exhausted "
                        f"({self._day_tokens:,}/{self.limits.tpd:,}, "
                        f"need {est_tokens:,}). Resets 00:00 UTC."
                    )

                n_req, n_tok = self._minute_usage()
                req_ok = self.limits.rpm is None or n_req < self.limits.rpm
                tok_ok = self.limits.tpm is None or n_tok + est_tokens <= self.limits.tpm

                if req_ok and tok_ok:
                    self._minute.append((now, est_tokens))
                    self._day_requests += 1
                    self._day_tokens += est_tokens
                    self._save_day()
                    return

                # Sleep only until the oldest entry ages out of the window.
                wait = 60.0 - (now - self._minute[0][0]) if self._minute else 1.0
                wait = max(0.05, min(wait, 5.0))

            if time.monotonic() > deadline:
                raise TimeoutError(f"[{self.name}] rate limit wait exceeded {timeout}s")
            time.sleep(wait)

    def reconcile(self, est_tokens: int, actual_tokens: int) -> None:
        """Correct the budgets once real usage is known.

        We must reserve *before* the call (we cannot know the true cost yet),
        so estimates drift. Folding the delta back in keeps the daily counter
        honest over a long run instead of accumulating error.
        """
        delta = actual_tokens - est_tokens
        if delta == 0:
            return
        with self._lock:
            self._day_tokens = max(0, self._day_tokens + delta)
            if self._minute:
                ts, tok = self._minute[-1]
                self._minute[-1] = (ts, max(0, tok + delta))
            self._save_day()
