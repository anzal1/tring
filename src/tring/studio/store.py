"""Where Studio sessions go after they end.

A studio session is worth keeping. It is the record of what an agent actually
said, which tools it actually chose, and what the turn cost, and the whole
point of ``GET /api/sessions`` is being able to open yesterday's call and
scrub through it instead of trying to reproduce it.

The format is the simplest one that supports that:

.. code-block:: text

    ~/.tring/studio/sessions/
        index.json        summaries, newest first: {id, started, agent, turns, cost}
        <session id>.jsonl  one SessionEvent per line, in the order emitted

JSON Lines, not one JSON document, for one reason: a line is appended the
moment an event is emitted, so a studio that is killed mid-call still leaves a
readable, complete-up-to-the-crash session on disk. A single document would
have to be rewritten on every event or written only at the end, and "only at
the end" is exactly when a crashing process has nothing to write.

The summaries are derived, never supplied. ``turns``, ``cost`` and ``agent``
are counted off the same :data:`~tring.events.SessionEvent` stream every other
consumer in Tring reads, so a summary cannot disagree with the events it
summarizes. The one thing the stream does not carry is wall-clock time (event
``at`` is seconds since session start, by design, so replays are
deterministic), so the store stamps ``started`` itself on the first event.

Concurrency: appends are serialized through one lock and the write itself runs
in a worker thread, so a slow or networked home directory cannot stall the
event loop that is driving a live call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Default location. Under ``~/.tring`` rather than the working directory: the
#: studio is run from wherever an agent spec happens to live, and session
#: history belongs to the developer, not to that directory.
DEFAULT_SESSIONS_DIR = Path.home() / ".tring" / "studio" / "sessions"

INDEX_NAME = "index.json"

#: How many summaries ``index.json`` keeps. The ``.jsonl`` files are never
#: deleted (throwing away a developer's recordings is not this module's call);
#: the index is a listing, and a listing that grows without bound turns into a
#: file rewritten in full on every session end.
MAX_INDEX_ENTRIES = 500

#: Session ids come from ``uuid4().hex``, but the id in ``GET /api/sessions/{id}``
#: arrives from the network and is used to build a path. Anything outside this
#: alphabet is refused rather than sanitized, because a "cleaned up" path is
#: still a path someone chose.
_SAFE_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


class _Progress:
    """What has been written for one live session, and what it adds up to."""

    def __init__(self, started: str) -> None:
        self.started = started
        self.written = 0
        self.agent: str | None = None
        self.turns = 0
        self.cost = 0.0

    def observe(self, event: Mapping[str, Any]) -> None:
        kind = event.get("type")
        if kind == "session_started":
            agent = event.get("agent_name")
            self.agent = agent if isinstance(agent, str) else None
        elif kind == "user_transcript" and event.get("final", True):
            self.turns += 1
        elif kind == "cost_recorded":
            amount = event.get("amount")
            if isinstance(amount, int | float):
                self.cost += float(amount)


def summarize(
    events: Sequence[Mapping[str, Any]], started: str, session_id: str
) -> dict[str, Any]:
    """Fold an event list into the summary shape ``GET /api/sessions`` returns.

    Shared by the live path (folded incrementally as events arrive) and the
    re-read path (folded over a file), so a session read back off disk reports
    the same numbers it reported while it was running.
    """
    progress = _Progress(started)
    for event in events:
        progress.observe(event)
    return {
        "id": session_id,
        "started": started,
        "agent": progress.agent,
        "turns": progress.turns,
        "cost": round(progress.cost, 6),
    }


class SessionStore:
    """Append-only session history under one directory.

    Args:
        root: where sessions live. Defaults to :data:`DEFAULT_SESSIONS_DIR`.
            Tests point it at a temporary directory; nothing else in the studio
            knows the path.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root).expanduser() if root else DEFAULT_SESSIONS_DIR
        self._lock = asyncio.Lock()
        self._live: dict[str, _Progress] = {}

    # ------------------------------------------------------------- writing

    async def append(self, session_id: str, event: Mapping[str, Any]) -> None:
        """Append one event to a session's file, creating it on the first call."""
        await self._write(session_id, [event])

    async def sync(self, session_id: str, events: Sequence[Mapping[str, Any]]) -> None:
        """Append whatever tail of ``events`` has not been written yet.

        The teardown path's safety net. A session's event forwarder is
        cancelled before the runtime is stopped, so the last few events (the
        final cost lines, ``session_ended``) are emitted with nobody left to
        append them. Handing the session's whole history here writes exactly
        the events the file is missing: the store counts what it has written,
        so calling this with events already on disk is a no-op rather than a
        duplicate.
        """
        await self._write(session_id, events, tail=True)

    async def _write(
        self, session_id: str, events: Sequence[Mapping[str, Any]], tail: bool = False
    ) -> None:
        path = self.path_for(session_id)
        async with self._lock:
            progress = self._live.get(session_id)
            if progress is None:
                progress = _Progress(datetime.now(UTC).isoformat(timespec="seconds"))
                self._live[session_id] = progress
            if tail:
                # Sliced under the lock, so an append that landed between the
                # caller's decision to sync and this point is not rewritten.
                events = events[progress.written :]
            if not events:
                return
            for event in events:
                progress.observe(event)
            # Counted before the write, not after: ``asyncio.to_thread`` hands
            # the job to the executor as it is called, so the line lands even
            # if this coroutine is cancelled while awaiting it. Counting
            # afterwards would let a cancelled append write its line and then
            # let ``sync`` write it a second time.
            progress.written += len(events)
            lines = "".join(json.dumps(event) + "\n" for event in events)
            await asyncio.to_thread(self._append_lines, path, lines)

    @staticmethod
    def _append_lines(path: Path, lines: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(lines)

    async def finish(self, session_id: str) -> dict[str, Any] | None:
        """Record the session's summary in ``index.json`` and forget it.

        Returns the summary, or ``None`` if nothing was ever appended for this
        id (a browser that connected and closed without pressing Start).
        """
        async with self._lock:
            progress = self._live.pop(session_id, None)
            if progress is None or progress.written == 0:
                return None
            summary = {
                "id": session_id,
                "started": progress.started,
                "agent": progress.agent,
                "turns": progress.turns,
                "cost": round(progress.cost, 6),
            }
            await asyncio.to_thread(self._rewrite_index, summary)
            return summary

    def _rewrite_index(self, summary: Mapping[str, Any]) -> None:
        """Replace this session's entry and write the index atomically.

        ``os.replace`` so a studio killed mid-write leaves the previous index
        intact rather than a truncated one: the listing is a cache of what the
        ``.jsonl`` files already say, and a half-written cache is worse than a
        slightly stale one.

        The disk-truth rule is enforced here, incoming summary included. A
        session whose recording was deleted while its socket was still open
        arrives at ``finish`` with a summary and no file; indexing it would
        resurrect a session the developer already threw away, as a row whose
        replay can only 404. The same filter garbage-collects any stale
        entries an external prune left behind, so this rewrite merges with
        what is on disk rather than clobbering it with remembered state.
        """
        entries = [entry for entry in self._read_index() if entry.get("id") != summary["id"]]
        entries.insert(0, dict(summary))
        entries = self._existing_only(entries)
        del entries[MAX_INDEX_ENTRIES:]
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / INDEX_NAME
        temporary = self.root / f"{INDEX_NAME}.tmp"
        temporary.write_text(json.dumps({"sessions": entries}, indent=2), encoding="utf-8")
        os.replace(temporary, target)

    # ------------------------------------------------------------- reading

    def summaries(self) -> list[dict[str, Any]]:
        """Every indexed session, newest first. The body of ``GET /api/sessions``.

        Filtered against the files actually on disk, because the index is a
        cache and the ``.jsonl`` recordings are the truth: a developer who
        deletes a recording between index rewrites must not be shown a row
        whose replay 404s. Read-only on purpose; the stale entries themselves
        are garbage-collected by the next index rewrite, not by a GET.
        """
        return self._existing_only(self._read_index())

    def _existing_only(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop entries whose recording no longer exists (or never could).

        An id that fails :data:`_SAFE_ID` is dropped too: it cannot name a
        file this store would have written, so whatever put it in the index,
        it has no recording to stand on.
        """
        kept: list[dict[str, Any]] = []
        for entry in entries:
            session_id = entry.get("id")
            if (
                isinstance(session_id, str)
                and _SAFE_ID.match(session_id)
                and (self.root / f"{session_id}.jsonl").exists()
            ):
                kept.append(entry)
        return kept

    def read(self, session_id: str) -> dict[str, Any] | None:
        """One session: its summary fields plus every event, in order.

        ``None`` for an id that is unsafe, unknown, or unreadable. The events
        are re-read from the file rather than served from the index, so a
        session that is still running (or that was interrupted) reads back with
        everything it has so far.
        """
        if not _SAFE_ID.match(session_id):
            return None
        path = self.root / f"{session_id}.jsonl"
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return None
        events = [json.loads(line) for line in raw.splitlines() if line.strip()]
        indexed = next(
            (entry for entry in self._read_index() if entry.get("id") == session_id), None
        )
        started = str(indexed["started"]) if indexed else self._file_time(path)
        return {**summarize(events, started, session_id), "events": events}

    def path_for(self, session_id: str) -> Path:
        """The file one session writes to. Raises ``ValueError`` on a bad id."""
        if not _SAFE_ID.match(session_id):
            raise ValueError(f"unusable session id {session_id!r}")
        return self.root / f"{session_id}.jsonl"

    def _read_index(self) -> list[dict[str, Any]]:
        try:
            raw = (self.root / INDEX_NAME).read_text(encoding="utf-8")
        except OSError:
            return []
        try:
            document = json.loads(raw)
        except json.JSONDecodeError:
            # A corrupt index is a lost listing, not a lost session: the
            # ``.jsonl`` files are still there and still readable by id. The
            # next session end rewrites it.
            logger.warning("studio session index is not valid JSON; ignoring it")
            return []
        sessions = document.get("sessions") if isinstance(document, dict) else None
        if not isinstance(sessions, list):
            return []
        return [entry for entry in sessions if isinstance(entry, dict)]

    @staticmethod
    def _file_time(path: Path) -> str:
        """Fallback ``started`` for a session that never reached the index."""
        try:
            stamp = path.stat().st_mtime
        except OSError:  # pragma: no cover - the file was just read
            return datetime.now(UTC).isoformat(timespec="seconds")
        return datetime.fromtimestamp(stamp, UTC).isoformat(timespec="seconds")


__all__ = [
    "DEFAULT_SESSIONS_DIR",
    "INDEX_NAME",
    "MAX_INDEX_ENTRIES",
    "SessionStore",
    "summarize",
]
