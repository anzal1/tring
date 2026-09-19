"""Tests for FasterWhisperStreamingSTT (providers/local/stt_streaming.py).

No network, no GPU, no real model weights: faster-whisper is swapped out for
a scripted fake via the ``_model`` constructor seam, and VAD is swapped out
for a scripted fake via the ``_vad`` constructor seam. This file therefore
tests exactly one thing -- how FasterWhisperStreamingSTT reacts to a sequence
of VAD events -- and deliberately does not test VAD endpointing correctness
itself (real energy thresholds, real silence timing), which is
``tring.vad``'s own test suite's job per that module's docstring.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from tring.providers.base import AudioFrame
from tring.providers.local.stt_streaming import FasterWhisperStreamingSTT
from tring.vad import VADEventKind

# One frame = 320 bytes of 16-bit mono PCM at 16 kHz = 10 ms = 0.01 s, the
# same frame size the house tests use elsewhere (tests/test_providers_stt.py)
# so utterance durations come out to round numbers.
_FRAME_BYTES = 320


def _frame() -> AudioFrame:
    return AudioFrame(pcm=b"\x00\x00" * (_FRAME_BYTES // 2), sample_rate=16000, channels=1)


class _FakeVADEvent:
    def __init__(self, kind: VADEventKind, at: float = 0.0) -> None:
        self.kind = kind
        self.at = at


class ScriptedFakeVAD:
    """Emits a scripted list of events per ``feed()`` call, in order.

    Frame *content* is ignored entirely -- the script is keyed purely on call
    index, which is what makes the streaming class's reaction to VAD events
    testable independently of any real energy/silence logic.
    """

    def __init__(self, script: list[list[_FakeVADEvent]]) -> None:
        self._script = list(script)
        self._index = 0
        self.feed_count = 0

    def feed(self, frame: AudioFrame) -> list[_FakeVADEvent]:
        self.feed_count += 1
        events = self._script[self._index] if self._index < len(self._script) else []
        self._index += 1
        return events


class FakeWhisperModel:
    """Replays a scripted list of transcripts, one per ``transcribe()`` call.

    Records every call's keyword arguments so tests can assert on
    ``condition_on_previous_text`` and ``language`` without needing a real
    model.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def transcribe(
        self,
        wav: Any,
        language: str | None = None,
        beam_size: int = 1,
        condition_on_previous_text: bool = True,
    ) -> tuple[list[Any], Any]:
        self.calls.append(
            {
                "language": language,
                "beam_size": beam_size,
                "condition_on_previous_text": condition_on_previous_text,
            }
        )
        text = self._responses.pop(0) if self._responses else ""
        segment = SimpleNamespace(text=text)
        info = SimpleNamespace(language=language or "en")
        return [segment], info


def _make_stt(
    responses: list[str],
    script: list[list[_FakeVADEvent]],
    **kwargs: Any,
) -> tuple[FasterWhisperStreamingSTT, FakeWhisperModel, ScriptedFakeVAD]:
    model = FakeWhisperModel(responses)
    vad = ScriptedFakeVAD(script)
    stt = FasterWhisperStreamingSTT(
        _model=lambda: model,
        _vad=lambda: vad,
        **kwargs,
    )
    return stt, model, vad


async def _frames(n: int) -> AsyncIterator[AudioFrame]:
    for _ in range(n):
        yield _frame()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_registers_under_expected_name() -> None:
    import tring.providers.local  # noqa: F401 -- import is the point of the test
    from tring.providers import registry

    assert ("stt", "faster_whisper_streaming") in registry.available("stt")


# ---------------------------------------------------------------------------
# Finals fire per utterance, partials fire between
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finals_fire_per_utterance_with_partials_between() -> None:
    # Two four-frame utterances (0.04s of *included* speech each -- see the
    # module docstring: the speech_end frame itself is excluded from the
    # utterance it closes), separated by a plain frame with no events.
    #
    #   idx0: speech_start        (included: frame 0)
    #   idx1: -                   (included: frame 1)   -> 0.02s -> partial #1
    #   idx2: -                   (included: frame 2)
    #   idx3: -                   (included: frame 3)   -> 0.02s -> partial #2
    #   idx4: speech_end          (excluded; finalizes frames 0-3 = 0.04s)
    #   idx5: speech_start        (included: frame 5)
    #   idx6: -                   (included: frame 6)   -> 0.02s -> partial #3
    #   idx7: -                   (included: frame 7)
    #   idx8: -                   (included: frame 8)   -> 0.02s -> partial #4
    #   idx9: speech_end          (excluded; finalizes frames 5-8 = 0.04s)
    script = [
        [_FakeVADEvent(VADEventKind.SPEECH_START)],
        [],
        [],
        [],
        [_FakeVADEvent(VADEventKind.SPEECH_END)],
        [_FakeVADEvent(VADEventKind.SPEECH_START)],
        [],
        [],
        [],
        [_FakeVADEvent(VADEventKind.SPEECH_END)],
    ]
    stt, model, vad = _make_stt(
        responses=["par", "par tial", "final one", "par2", "par2 tial", "final two"],
        script=script,
        partial_interval_s=0.02,
    )

    results = [r async for r in stt.transcribe(_frames(10))]

    finals = [r for r in results if r.final]
    partials = [r for r in results if not r.final]

    assert len(finals) == 2
    assert len(partials) == 4
    assert vad.feed_count == 10

    # Order: partial, partial, final, partial, partial, final.
    assert [r.final for r in results] == [False, False, True, False, False, True]

    for final in finals:
        assert final.usage[0].unit_name == "audio_seconds"
        assert final.usage[0].units == pytest.approx(0.04)
        assert final.usage[0].estimated is False
        assert final.usage[0].model == "base"

    for partial in partials:
        assert partial.usage == []

    # condition_on_previous_text: False on every partial, True on every final.
    partial_calls = model.calls[:2] + model.calls[3:5]
    final_calls = [model.calls[2], model.calls[5]]
    assert all(c["condition_on_previous_text"] is False for c in partial_calls)
    assert all(c["condition_on_previous_text"] is True for c in final_calls)


@pytest.mark.asyncio
async def test_partial_interval_zero_disables_partials() -> None:
    script = [
        [_FakeVADEvent(VADEventKind.SPEECH_START)],
        [],
        [],
        [_FakeVADEvent(VADEventKind.SPEECH_END)],
    ]
    stt, model, _vad = _make_stt(
        responses=["hello there"],
        script=script,
        partial_interval_s=0,
    )

    results = [r async for r in stt.transcribe(_frames(4))]

    assert len(results) == 1
    assert results[0].final is True
    assert results[0].text == "hello there"
    # Only the one final transcription call was ever made.
    assert len(model.calls) == 1
    assert model.calls[0]["condition_on_previous_text"] is True


# ---------------------------------------------------------------------------
# Tail flush: stream ends mid-utterance with no speech_end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flushes_open_utterance_when_stream_ends_without_speech_end() -> None:
    script = [
        [_FakeVADEvent(VADEventKind.SPEECH_START)],
        [],
        [],
    ]
    stt, _model, _vad = _make_stt(
        responses=["cut off mid"],
        script=script,
        partial_interval_s=0,  # isolate the tail-flush behaviour
    )

    results = [r async for r in stt.transcribe(_frames(3))]

    assert len(results) == 1
    assert results[0].final is True
    assert results[0].text == "cut off mid"
    # All three frames (start + 2 more) were included since there was never a
    # speech_end frame to exclude.
    assert results[0].usage[0].units == pytest.approx(0.03)


@pytest.mark.asyncio
async def test_no_utterance_at_all_yields_nothing() -> None:
    stt, _model, _vad = _make_stt(responses=[], script=[[], [], []])

    results = [r async for r in stt.transcribe(_frames(3))]

    assert results == []


@pytest.mark.asyncio
async def test_empty_transcription_result_is_dropped_not_yielded() -> None:
    # Model returns "" for the utterance (e.g. pure noise) -- no STTResult
    # should be yielded for it, final or partial.
    script = [
        [_FakeVADEvent(VADEventKind.SPEECH_START)],
        [_FakeVADEvent(VADEventKind.SPEECH_END)],
    ]
    stt, _model, _vad = _make_stt(responses=[""], script=script)

    results = [r async for r in stt.transcribe(_frames(2))]

    assert results == []


# ---------------------------------------------------------------------------
# Language passthrough
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transcribe_call_language_overrides_constructor_default() -> None:
    script = [
        [_FakeVADEvent(VADEventKind.SPEECH_START)],
        [_FakeVADEvent(VADEventKind.SPEECH_END)],
    ]
    stt, model, _vad = _make_stt(responses=["hola"], script=script, language="en")

    results = [r async for r in stt.transcribe(_frames(2), language="es")]

    assert model.calls[0]["language"] == "es"
    assert results[0].language == "es"


@pytest.mark.asyncio
async def test_constructor_language_used_when_call_omits_it() -> None:
    script = [
        [_FakeVADEvent(VADEventKind.SPEECH_START)],
        [_FakeVADEvent(VADEventKind.SPEECH_END)],
    ]
    stt, model, _vad = _make_stt(responses=["bonjour"], script=script, language="fr")

    results = [r async for r in stt.transcribe(_frames(2))]

    assert model.calls[0]["language"] == "fr"
    assert results[0].language == "fr"


# ---------------------------------------------------------------------------
# model_size threading into Usage.model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_final_usage_model_reflects_model_size_option() -> None:
    script = [
        [_FakeVADEvent(VADEventKind.SPEECH_START)],
        [_FakeVADEvent(VADEventKind.SPEECH_END)],
    ]
    stt, _model, _vad = _make_stt(
        responses=["hi"], script=script, model_size="small.en"
    )

    results = [r async for r in stt.transcribe(_frames(2))]

    assert results[0].usage[0].model == "small.en"


# ---------------------------------------------------------------------------
# VAD engine selection
# ---------------------------------------------------------------------------


def test_unknown_vad_engine_raises_value_error() -> None:
    stt = FasterWhisperStreamingSTT(_model=lambda: FakeWhisperModel([]), vad="webrtc")
    with pytest.raises(ValueError, match="webrtc"):
        stt._load_vad()


def test_vad_factory_seam_is_cached_across_calls() -> None:
    vad = ScriptedFakeVAD([])
    calls = {"n": 0}

    def factory() -> ScriptedFakeVAD:
        calls["n"] += 1
        return vad

    stt = FasterWhisperStreamingSTT(_model=lambda: FakeWhisperModel([]), _vad=factory)
    first = stt._load_vad()
    second = stt._load_vad()

    assert first is vad
    assert second is vad
    assert calls["n"] == 1


def test_default_vad_engine_energy_resolves_a_real_energy_vad() -> None:
    """Exercises the real (non-seamed) default-VAD path: ``make_vad("energy")``."""
    from tring.vad import EnergyVAD

    stt = FasterWhisperStreamingSTT(_model=lambda: FakeWhisperModel([]))
    vad = stt._load_vad()

    assert isinstance(vad, EnergyVAD)
    # Cached, not rebuilt, on a second call.
    assert stt._load_vad() is vad


# ---------------------------------------------------------------------------
# faster-whisper missing -> ImportError names the tring[local] extra
# ---------------------------------------------------------------------------


def test_missing_faster_whisper_raises_error_naming_local_extra() -> None:
    # No `_model` seam supplied: exercises the real lazy import. faster-whisper
    # is not installed in this environment (core install has no ML stack),
    # so this raises for real rather than needing a monkeypatch.
    stt = FasterWhisperStreamingSTT(vad="energy", _vad=lambda: ScriptedFakeVAD([]))
    with pytest.raises(ImportError, match=r'tring\[local\]'):
        stt._load_model()


def test_negative_partial_interval_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="partial_interval_s"):
        FasterWhisperStreamingSTT(partial_interval_s=-1.0)
