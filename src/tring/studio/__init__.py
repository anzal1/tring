"""Tring Studio — a browser console for an agent spec.

Studio is the dev loop the console transport hints at, with a UI: edit the
:class:`~tring.agent.AgentSpec`, start a live session, type turns, and watch the
transcript, the tool timeline and the cost ledger build themselves out of the
same :data:`~tring.events.SessionEvent` stream every other consumer reads.

Run it with ``python -m tring.studio``. The wire contract between this server
and the page it serves is ``docs/STUDIO_PROTOCOL.md``.

Importing this package registers the ``studio_silent`` TTS provider, so it shows
up in the registry (and in the studio's own provider dropdowns) alongside the
real engines.
"""

from tring.studio.server import (
    DEFAULT_AGENT_PATH,
    DEFAULT_HOST,
    DEFAULT_PORT,
    StudioServer,
)
from tring.studio.silent_tts import SILENT_TTS_NAME, StudioSilentTTS

__all__ = [
    "DEFAULT_AGENT_PATH",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "SILENT_TTS_NAME",
    "StudioServer",
    "StudioSilentTTS",
]
