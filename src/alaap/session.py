"""CallSession — one live conversation.

The session owns the event bus and the clock. Runtimes push events; any number
of consumers (cost meter, analytics, transports, tests) subscribe. The clock is
injectable so tests and replays are deterministic.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Callable

from alaap.agent import AgentSpec
from alaap.events import SessionEvent


class CallSession:
    def __init__(
        self,
        agent: AgentSpec,
        session_id: str | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.agent = agent
        self.session_id = session_id or uuid.uuid4().hex
        self._clock = clock or time.monotonic
        self._t0 = self._clock()
        self._subscribers: list[asyncio.Queue[SessionEvent | None]] = []
        self._history: list[SessionEvent] = []
        self._closed = False

    @property
    def elapsed(self) -> float:
        return self._clock() - self._t0

    @property
    def history(self) -> list[SessionEvent]:
        return list(self._history)

    def emit(self, event: SessionEvent) -> None:
        if self._closed:
            raise RuntimeError("session is closed")
        self._history.append(event)
        for q in self._subscribers:
            q.put_nowait(event)

    async def subscribe(self) -> AsyncIterator[SessionEvent]:
        q: asyncio.Queue[SessionEvent | None] = asyncio.Queue()
        self._subscribers.append(q)
        try:
            while True:
                item = await q.get()
                if item is None:
                    return
                yield item
        finally:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def close(self) -> None:
        self._closed = True
        for q in self._subscribers:
            q.put_nowait(None)
