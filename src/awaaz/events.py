"""Unified session event model.

Every runtime — cascade, speech-to-speech, hybrid — emits this one event
vocabulary. Consumers (cost metering, analytics, transports, tests) subscribe
to events and never import a concrete runtime.

Transcript availability differs by architecture: cascade produces transcripts
mid-turn, S2S models often only after the call ends. Events therefore carry an
explicit ``availability`` marker instead of letting consumers assume timing.
"""

from __future__ import annotations

import enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class Role(str, enum.Enum):
    USER = "user"
    BOT = "bot"


class TranscriptAvailability(str, enum.Enum):
    LIVE = "live"  # produced during the turn (cascade STT)
    POST_CALL = "post_call"  # reconstructed after the call (typical S2S)


class CostComponent(str, enum.Enum):
    STT = "stt"
    LLM = "llm"
    TTS = "tts"
    S2S = "s2s"
    TELEPHONY = "telephony"
    EMBEDDING = "embedding"
    OTHER = "other"


class BaseEvent(BaseModel):
    """Common envelope for all session events.

    ``at`` is seconds since session start (monotonic), supplied by the
    session clock so replays and tests are deterministic.
    """

    session_id: str
    at: float


class SessionStarted(BaseEvent):
    type: Literal["session_started"] = "session_started"
    agent_name: str
    runtime_mode: str


class SessionEnded(BaseEvent):
    type: Literal["session_ended"] = "session_ended"
    reason: str = "completed"
    duration_seconds: float = 0.0


class UserTranscript(BaseEvent):
    type: Literal["user_transcript"] = "user_transcript"
    text: str
    final: bool = True
    language: str | None = None
    availability: TranscriptAvailability = TranscriptAvailability.LIVE


class BotUtterance(BaseEvent):
    """Text the bot has *generated* for speech (may not all be played)."""

    type: Literal["bot_utterance"] = "bot_utterance"
    text: str
    language: str | None = None


class BotSpeechPlayed(BaseEvent):
    """Text corresponding to audio that was *actually played* to the caller.

    TTS generates faster than audio plays; the gap between ``BotUtterance``
    and ``BotSpeechPlayed`` is what the interruption annotator reconciles.
    """

    type: Literal["bot_speech_played"] = "bot_speech_played"
    text: str


class Interruption(BaseEvent):
    """A genuine barge-in: the caller spoke while the bot was mid-utterance."""

    type: Literal["interruption"] = "interruption"
    heard_text: str
    unheard_text: str


class ToolCallStarted(BaseEvent):
    type: Literal["tool_call_started"] = "tool_call_started"
    tool_name: str
    call_id: str
    arguments: dict = Field(default_factory=dict)
    waiting_message: str | None = None
    spoken_mode: str | None = None


class ToolCallCompleted(BaseEvent):
    type: Literal["tool_call_completed"] = "tool_call_completed"
    tool_name: str
    call_id: str
    ok: bool = True
    result_summary: str | None = None
    latency_seconds: float | None = None


class CostRecorded(BaseEvent):
    """One metered cost line.

    ``estimated`` is honest accounting: True whenever units were inferred
    (e.g. chars/4 token estimates) rather than reported by the vendor.
    """

    type: Literal["cost_recorded"] = "cost_recorded"
    component: CostComponent
    provider: str
    units: float
    unit_name: str  # "tokens_in", "tokens_out", "tts_chars", "audio_seconds", ...
    amount: float
    currency: str = "USD"
    estimated: bool = False
    model: str | None = None
    cached_units: float | None = None  # e.g. cached prompt tokens, when reported


class SessionError(BaseEvent):
    type: Literal["error"] = "error"
    message: str
    recoverable: bool = True


SessionEvent = Annotated[
    Union[
        SessionStarted,
        SessionEnded,
        UserTranscript,
        BotUtterance,
        BotSpeechPlayed,
        Interruption,
        ToolCallStarted,
        ToolCallCompleted,
        CostRecorded,
        SessionError,
    ],
    Field(discriminator="type"),
]
