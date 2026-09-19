"""``studio_silent`` — a TTS provider that meters speech without making any.

Why this exists
---------------

Tring Studio is a *text* console for a *voice* agent. The browser types a turn,
the cascade runtime runs it end to end, and the UI renders the transcript, the
tool timeline and the cost panel from the session's event stream. What the UI
deliberately does **not** do is play audio: ``docs/STUDIO_PROTOCOL.md`` shipping
PCM to the browser would buy nothing and cost a lot of bandwidth, so bot audio
is counted rather than sent.

That leaves the TTS slot in an awkward spot. A cascade session needs *a* TTS
provider to exist — ``CascadeRuntime`` refuses to start without one — but on a
laptop with no ML stack and no vendor keys there may not be a working one, and
even where there is, synthesizing audio nobody will hear is pure waste.

The tempting shortcut is a provider that reports nothing at all. That would
quietly break the one promise the cost panel makes: the number of characters an
agent speaks is a real, billable quantity, and a persona rewrite that doubles it
is exactly the kind of regression Studio exists to surface. So this provider
emits no audio frames and still reports the **exact** character count it was
handed, ``estimated=False`` — the count is measured, not inferred.

What that means for the cost ledger
-----------------------------------

No ``Rate`` ships for ``studio_silent`` (there is no vendor and no price), so
:class:`~tring.cost.meter.CostMeter` records these lines at ``0.0`` with
``estimated=True`` and a missing-rate note. That is the correct reading and not
a gap: the characters are exact, the price is unknown *because nothing was paid
to speak them*. Point the spec at a real TTS provider and the same characters
acquire a real amount.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from tring.providers.base import TTSChunk, TTSProvider, Usage
from tring.providers.registry import register
from tring.runtimes.base import AudioFrame

#: Registry name, shared with the studio server so the two cannot drift.
SILENT_TTS_NAME = "studio_silent"

#: A zero-length frame in the canonical wire format. Frozen and shared: the
#: contract requires a frame per chunk, and "no audio" is honestly expressed as
#: zero bytes rather than as a buffer of silence, which would inflate the
#: runtime's playout estimate and the studio's ``audio_progress`` counter with
#: audio that was never produced.
SILENT_FRAME = AudioFrame(pcm=b"", sample_rate=16000, channels=1)


@register("tts", SILENT_TTS_NAME)
class StudioSilentTTS(TTSProvider):
    """Consume a text stream, produce no audio, report exact ``tts_chars``.

    One usage line per utterance rather than per flushed segment: a real engine
    flushes early because the caller is waiting to *hear* something, and there
    is nothing to wait for here. Counting once at the end is the smallest
    honest ledger for a stage that did no work.

    A barge-in cancels synthesis mid-utterance, which cancels this generator
    before the line is emitted — so aborted speech is not billed. That is
    accurate for a silent engine: nothing was synthesized.
    """

    name = SILENT_TTS_NAME

    def __init__(self, **_options: Any) -> None:
        # Accepts and ignores provider options so a spec written for a real
        # engine (voice, speed, model paths) can be run in the studio without
        # being edited first.
        pass

    async def synthesize(
        self, text: AsyncIterator[str], voice: str | None = None
    ) -> AsyncIterator[TTSChunk]:
        spoken = 0
        async for delta in text:
            spoken += len(delta)
        if not spoken:
            return
        yield TTSChunk(
            frame=SILENT_FRAME,
            usage=[
                Usage(
                    units=float(spoken),
                    unit_name="tts_chars",
                    # Measured: these are the characters this provider was
                    # actually handed, not an estimate derived from anything.
                    estimated=False,
                    model=None,
                )
            ],
        )


__all__ = ["SILENT_FRAME", "SILENT_TTS_NAME", "StudioSilentTTS"]
