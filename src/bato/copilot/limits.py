"""Request limits for a PUBLIC copilot (e.g. the Render deployment), so a script or a stuck page cannot run up the
LLM bill or exhaust a free-tier quota. Configured only through environment variables; the defaults suit a demo.

    BATO_COPILOT_PER_MINUTE   copilot POST requests per client per rolling minute       (default 6)
    BATO_COPILOT_PER_DAY      model-calling requests per server per UTC day, all clients (default 200)
    BATO_COPILOT_CONCURRENCY  model-calling requests in flight at once, all clients       (default 2)

"Model-calling" = questions and the Nepali SITREP. The English SITREP is the deterministic template (no AI): it is
only subject to the per-client minute limit, never to the daily cap, so it keeps working when the AI budget is spent.
State is in memory (one server process); a restart resets it.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone

ENV_PER_MINUTE, ENV_PER_DAY, ENV_CONCURRENCY = (
    "BATO_COPILOT_PER_MINUTE", "BATO_COPILOT_PER_DAY", "BATO_COPILOT_CONCURRENCY")
DEFAULTS = {ENV_PER_MINUTE: 6, ENV_PER_DAY: 200, ENV_CONCURRENCY: 2}
WINDOW_S = 60.0
MAX_TRACKED_CLIENTS = 10_000


@dataclass(frozen=True)
class Decision:
    allowed: bool
    message: str = ""
    retry_after_s: int = 0


class CopilotLimiter:
    def __init__(self, per_minute: int = 6, per_day: int = 200, concurrency: int = 2,
                 clock: Callable[[], float] = time.monotonic,
                 today: Callable[[], str] = lambda: datetime.now(timezone.utc).date().isoformat()):
        if min(per_minute, per_day, concurrency) < 1:
            raise ValueError("copilot limits must be at least 1")
        self.per_minute, self.per_day, self.concurrency = per_minute, per_day, concurrency
        self._clock, self._today = clock, today
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}
        self._day, self._used_today, self._in_flight = "", 0, 0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> CopilotLimiter:
        e = os.environ if env is None else env
        values = {}
        for key, default in DEFAULTS.items():
            raw = (e.get(key) or "").strip()
            try:
                values[key] = int(raw) if raw else default
            except ValueError:
                raise ValueError(f"{key} must be a whole number, got {raw!r}") from None
            if values[key] < 1:
                raise ValueError(f"{key} must be at least 1")
        return cls(values[ENV_PER_MINUTE], values[ENV_PER_DAY], values[ENV_CONCURRENCY])

    def acquire(self, client: str, uses_model: bool) -> Decision:
        """Admit or refuse one request. An admitted model call MUST be followed by ``release(True)``."""
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(client, deque())
            while hits and now - hits[0] >= WINDOW_S:
                hits.popleft()
            if len(hits) >= self.per_minute:
                wait = max(1, int(WINDOW_S - (now - hits[0])) + 1)
                return Decision(False, f"Too many copilot requests from this client: limit {self.per_minute} per "
                                       f"minute. Try again in {wait} s.", wait)
            if uses_model:
                day = self._today()
                if day != self._day:
                    self._day, self._used_today = day, 0
                if self._used_today >= self.per_day:
                    return Decision(False, f"The AI copilot's daily limit ({self.per_day} requests) is used up; it "
                                           "resets at 00:00 UTC. The deterministic English SITREP still works.", 3600)
                if self._in_flight >= self.concurrency:
                    return Decision(False, "The AI copilot is busy with other requests. Try again in a few seconds.",
                                    5)
                self._used_today += 1
                self._in_flight += 1
            hits.append(now)
            if len(self._hits) > MAX_TRACKED_CLIENTS:
                self._prune(now)
            return Decision(True)

    def release(self, uses_model: bool) -> None:
        if uses_model:
            with self._lock:
                self._in_flight = max(0, self._in_flight - 1)

    def _prune(self, now: float) -> None:
        for key in [k for k, d in self._hits.items() if not d or now - d[-1] >= WINDOW_S]:
            del self._hits[key]

    def public(self) -> dict[str, int]:
        return {"per_minute": self.per_minute, "per_day": self.per_day, "concurrency": self.concurrency}
