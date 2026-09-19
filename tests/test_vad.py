"""Tests for tring.vad, tring.providers.local.vad_silero and TurnTaker.

Everything here runs on scripted PCM built with the stdlib ``array`` module:
deterministic sine bursts against deterministic pseudo-noise, framed at 20 ms
exactly as a transport would deliver them. No audio hardware, no model
downloads, no numpy.

The silero tests are the interesting ones. Its ONNX stack is genuinely absent
in a core install, which makes the ImportError path real rather than mocked;
the chunking tests then stub ``onnxruntime`` and ``numpy`` in ``sys.modules``
with fakes small enough to read in one sitting. Stubbing the modules rather
than injecting a session through a constructor parameter keeps the production
class free of a test-only seam, and it exercises the real session-construction
path (options, providers, model resolution) instead of skipping past it.
"""

from __future__ import annotations

import array
import math
import random
import sys
from types import ModuleType
from typing import Any

import pytest

from tring.agent import AgentSpec
from tring.events import Interruption
from tring.primitives.interruption import InterruptionVerdict, PlaybackLedger
from tring.providers.local.vad_silero import SileroVAD
from tring.runtimes.base import AudioFrame
from tring.runtimes.turn_taking import TurnTaker, concat_frames
from tring.session import CallSession
from tring.vad import (
    EnergyVAD,
    VADEvent,
    VADEventKind,
    VoiceActivityDetector,
    make_vad,
    pcm_dbfs,
)

SAMPLE_RATE = 16000
FRAME_MS = 20


# --------------------------------------------------------------------------
# Scripted audio
# --------------------------------------------------------------------------


def sine_pcm(ms: int, freq: float = 220.0, amplitude: float = 0.3) -> bytes:
    """A pure tone: stands in for speech energy, and is perfectly repeatable."""
    count = SAMPLE_RATE * ms // 1000
    samples = array.array(
        "h",
        (
            int(amplitude * 32767.0 * math.sin(2.0 * math.pi * freq * i / SAMPLE_RATE))
            for i in range(count)
        ),
    )
    return samples.tobytes()


def noise_pcm(ms: int, amplitude: float = 0.002, seed: int = 7) -> bytes:
    """Low-level room tone. Seeded, so every run sees the same noise floor."""
    rng = random.Random(seed)
    count = SAMPLE_RATE * ms // 1000
    peak = amplitude * 32767.0
    samples = array.array("h", (int(rng.uniform(-peak, peak)) for _ in range(count)))
    return samples.tobytes()


def framed(pcm: bytes, ms: int = FRAME_MS) -> list[AudioFrame]:
    """Slice a buffer the way a transport would: fixed-size frames, in order."""
    step = (SAMPLE_RATE * ms // 1000) * 2
    return [
        AudioFrame(pcm=pcm[i : i + step], sample_rate=SAMPLE_RATE, channels=1)
        for i in range(0, len(pcm), step)
    ]


def drive(vad: VoiceActivityDetector, pcm: bytes) -> list[VADEvent]:
    events: list[VADEvent] = []
    for frame in framed(pcm):
        events.extend(vad.feed(frame))
    return events


def kinds(events: list[VADEvent]) -> list[VADEventKind]:
    return [event.kind for event in events]


# --------------------------------------------------------------------------
# Level helpers
# --------------------------------------------------------------------------


def test_dbfs_scale_places_noise_far_below_speech() -> None:
    """The gate's entire premise: a burst clears seeded room tone by tens of dB."""
    assert pcm_dbfs(b"") == pytest.approx(-90.0)
    assert pcm_dbfs(noise_pcm(100)) < -50.0
    assert pcm_dbfs(sine_pcm(100)) > -20.0


# --------------------------------------------------------------------------
# EnergyVAD
# --------------------------------------------------------------------------


def test_energy_vad_detects_burst_against_noise() -> None:
    vad = EnergyVAD()
    pcm = noise_pcm(600) + sine_pcm(500) + noise_pcm(800)

    events = drive(vad, pcm)

    assert kinds(events) == [VADEventKind.SPEECH_START, VADEventKind.SPEECH_END]
    start, end = events
    # SPEECH_START is timestamped at the onset of the burst (0.6 s), not at the
    # frame that accumulated enough evidence to confirm it.
    assert start.at == pytest.approx(0.6, abs=FRAME_MS / 1000.0)
    # SPEECH_END is the last voiced moment (1.1 s), with the hangover excluded.
    assert end.at == pytest.approx(1.1, abs=FRAME_MS / 1000.0)
    assert vad.speaking is False
    assert vad.noise_floor_db is not None and vad.noise_floor_db < -50.0


def test_energy_vad_ignores_a_single_loud_frame() -> None:
    """A click is not a word: min_speech_ms is what makes that true."""
    vad = EnergyVAD(min_speech_ms=120)
    pcm = noise_pcm(400) + sine_pcm(FRAME_MS) + noise_pcm(400)

    assert drive(vad, pcm) == []
    assert vad.speaking is False


def test_hangover_merges_close_bursts_into_one_utterance() -> None:
    vad = EnergyVAD(hangover_ms=300)
    # 150 ms apart: well inside the hangover, so this is one utterance with a
    # pause in it, the way "four... five" is one answer.
    pcm = noise_pcm(400) + sine_pcm(300) + noise_pcm(150) + sine_pcm(300) + noise_pcm(600)

    events = drive(vad, pcm)

    assert kinds(events) == [VADEventKind.SPEECH_START, VADEventKind.SPEECH_END]
    assert events[0].at == pytest.approx(0.4, abs=FRAME_MS / 1000.0)
    # The end lands after the *second* burst (0.4+0.3+0.15+0.3 = 1.15 s).
    assert events[1].at == pytest.approx(1.15, abs=FRAME_MS / 1000.0)


def test_gap_longer_than_hangover_splits_into_two_utterances() -> None:
    """The companion to the merge test: past the hangover, they are two turns."""
    vad = EnergyVAD(hangover_ms=300)
    pcm = noise_pcm(400) + sine_pcm(300) + noise_pcm(600) + sine_pcm(300) + noise_pcm(600)

    events = drive(vad, pcm)

    assert kinds(events) == [
        VADEventKind.SPEECH_START,
        VADEventKind.SPEECH_END,
        VADEventKind.SPEECH_START,
        VADEventKind.SPEECH_END,
    ]


def test_noise_floor_climbs_to_meet_a_moderately_noisy_line() -> None:
    """Ordinary adaptation: a louder-than-seeded room stops looking like speech."""
    vad = EnergyVAD()
    room = noise_pcm(2000, amplitude=0.01, seed=11)

    assert drive(vad, room) == []
    assert vad.noise_floor_db is not None
    # The floor climbed from its -50 dB seed to sit near the room's own level
    # (about -45 dBFS), which is what keeps the gate closed on it.
    assert vad.noise_floor_db > -48.0


def test_a_latched_gate_is_closed_and_the_floor_recalibrated() -> None:
    """The documented worst case: a line loud enough to trip the seeded gate.

    An energy VAD has no way to know this is not speech, so it does open a
    turn. What it must not do is stay open for the rest of the call.
    """
    vad = EnergyVAD(max_speech_ms=400)
    loud_room = noise_pcm(1500, amplitude=0.05, seed=11)

    events = drive(vad, loud_room)

    assert kinds(events) == [VADEventKind.SPEECH_START, VADEventKind.SPEECH_END]
    assert vad.speaking is False
    # Re-seeded from the quietest frame of the spurious turn: roughly the
    # room's own level (about -31 dBFS), not the -50 dB seed it started from.
    assert vad.noise_floor_db is not None and vad.noise_floor_db > -40.0

    # Recovered: the same room no longer triggers, and real speech still does.
    assert drive(vad, noise_pcm(1000, amplitude=0.05, seed=11)) == []
    burst = drive(vad, sine_pcm(400) + noise_pcm(700))
    assert kinds(burst) == [VADEventKind.SPEECH_START, VADEventKind.SPEECH_END]


def test_call_opening_mid_speech_still_detects_the_turn() -> None:
    """The seed must be capped: calibrating "silence" from speech goes deaf."""
    vad = EnergyVAD()
    pcm = sine_pcm(500) + noise_pcm(800)

    events = drive(vad, pcm)

    assert kinds(events) == [VADEventKind.SPEECH_START, VADEventKind.SPEECH_END]
    assert events[0].at == pytest.approx(0.0, abs=FRAME_MS / 1000.0)


def test_reset_clears_speech_state_but_not_the_clock() -> None:
    vad = EnergyVAD()
    drive(vad, noise_pcm(400) + sine_pcm(300))
    assert vad.speaking is True

    vad.reset()

    assert vad.speaking is False
    assert vad.noise_floor_db is None
    # The timeline keeps running: a consumer's buffer offsets stay comparable.
    events = drive(vad, noise_pcm(400) + sine_pcm(400) + noise_pcm(600))
    assert kinds(events) == [VADEventKind.SPEECH_START, VADEventKind.SPEECH_END]
    assert events[0].at > 0.7


def test_empty_frames_are_ignored() -> None:
    vad = EnergyVAD()
    assert vad.feed(AudioFrame(pcm=b"", sample_rate=SAMPLE_RATE, channels=1)) == []


def test_make_vad_builds_each_tier_and_rejects_the_rest() -> None:
    energy = make_vad("energy", hangover_ms=99)
    assert isinstance(energy, EnergyVAD)
    assert energy.hangover_ms == 99
    # Options meant for another tier are absorbed, so one config block serves
    # both without the caller having to split it.
    assert isinstance(make_vad("energy", threshold=0.9), EnergyVAD)
    assert isinstance(make_vad("silero"), SileroVAD)

    with pytest.raises(ValueError, match="unknown vad kind"):
        make_vad("webrtc")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# SileroVAD
# --------------------------------------------------------------------------


def test_silero_without_onnxruntime_raises_a_helpful_import_error() -> None:
    """The real failure: a core install pointed at the neural tier."""
    assert "onnxruntime" not in sys.modules
    vad = SileroVAD()

    with pytest.raises(ImportError) as exc_info:
        vad.feed(framed(sine_pcm(100))[0])

    message = str(exc_info.value)
    assert "onnxruntime" in message
    assert 'pip install "tring[local]"' in message
    assert "make_vad('energy')" in message  # the way out that needs no install


def test_silero_rejects_frames_it_cannot_score() -> None:
    """Resampling belongs in the transport; guessing here would be silent error."""
    vad = SileroVAD(sample_rate=16000)
    with pytest.raises(ValueError, match="8000 Hz"):
        vad.feed(AudioFrame(pcm=sine_pcm(100), sample_rate=8000, channels=1))
    with pytest.raises(ValueError, match="2 channel"):
        vad.feed(AudioFrame(pcm=sine_pcm(100), sample_rate=16000, channels=2))
    with pytest.raises(ValueError, match="sample rates"):
        SileroVAD(sample_rate=44100)


class FakeOrtSession:
    """An ort-shaped session returning scripted probabilities.

    Records every feed dict so the tests can assert on what the chunker
    actually handed the model: window length, context prefix, and the state
    tensor threaded back in from the previous call.
    """

    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = list(probabilities)
        self.feeds: list[dict[str, Any]] = []
        self.calls = 0

    def run(self, output_names: Any, feed: dict[str, Any]) -> list[Any]:
        assert output_names is None
        self.feeds.append(feed)
        index = min(self.calls, len(self._probabilities) - 1)
        self.calls += 1
        # Silero returns [probability, new_state]; the probability is shaped
        # (batch, 1), hence the double nesting.
        return [[[self._probabilities[index]]], f"state-{self.calls}"]


class FakeSessionOptions:
    def __init__(self) -> None:
        self.inter_op_num_threads = 0
        self.intra_op_num_threads = 0


def install_fake_onnx(
    monkeypatch: pytest.MonkeyPatch, session: FakeOrtSession
) -> dict[str, Any]:
    """Stub ``onnxruntime`` and ``numpy`` for the duration of one test.

    The numpy stub is this small because the production code's entire numpy
    surface is three calls: wrap a nested list, wrap a scalar, and allocate the
    zeroed recurrent state. ``array`` returning its argument unchanged is what
    lets the tests read the real sample values back out of the feed dict.
    """
    built: dict[str, Any] = {}

    class FakeInferenceSession:
        def __init__(self, path: str, sess_options: Any, providers: Any) -> None:
            built["path"] = path
            built["options"] = sess_options
            built["providers"] = providers

        def run(self, output_names: Any, feed: dict[str, Any]) -> list[Any]:
            return session.run(output_names, feed)

    ort = ModuleType("onnxruntime")
    ort.SessionOptions = FakeSessionOptions  # type: ignore[attr-defined]
    ort.InferenceSession = FakeInferenceSession  # type: ignore[attr-defined]

    np = ModuleType("numpy")
    np.float32 = "float32"  # type: ignore[attr-defined]
    np.array = lambda obj, dtype=None: obj  # type: ignore[attr-defined]
    np.zeros = lambda shape, dtype=None: f"zeros{shape}"  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "numpy", np)
    return built


def test_silero_reframes_arbitrary_frames_into_exact_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    session = FakeOrtSession([0.0])
    built = install_fake_onnx(monkeypatch, session)
    model = tmp_path / "silero_vad.onnx"
    model.write_bytes(b"not-a-real-model")

    vad = SileroVAD(model_path=str(model))
    # 20 ms frames are 320 samples; the model takes 512. Five frames is 1600
    # samples, which is three windows (1536) with 64 samples left pending.
    for frame in framed(sine_pcm(100)):
        vad.feed(frame)

    assert session.calls == 3
    for feed in session.feeds:
        # Every input is context (64) + window (512), never padded short.
        assert len(feed["input"][0]) == 64 + 512
        assert feed["sr"] == 16000
    # The first window's context is the documented zero prefix; later windows
    # carry the tail of their predecessor, which is the point of keeping it.
    assert session.feeds[0]["input"][0][:64] == [0.0] * 64
    assert session.feeds[1]["input"][0][:64] == session.feeds[0]["input"][0][-64:]
    # The recurrent state is threaded forward, not re-zeroed every window.
    assert session.feeds[0]["state"] == "zeros(2, 1, 128)"
    assert session.feeds[1]["state"] == "state-1"
    assert session.feeds[2]["state"] == "state-2"

    # The session was built the way silero builds it: single-threaded, CPU.
    assert built["path"] == str(model)
    assert built["providers"] == ["CPUExecutionProvider"]
    assert built["options"].inter_op_num_threads == 1
    assert built["options"].intra_op_num_threads == 1


def test_silero_samples_are_normalised_to_unit_range(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    session = FakeOrtSession([0.0])
    install_fake_onnx(monkeypatch, session)
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")

    vad = SileroVAD(model_path=str(model))
    pcm = sine_pcm(50, amplitude=1.0)
    for frame in framed(pcm):
        vad.feed(frame)

    window = session.feeds[0]["input"][0][64:]
    assert max(window) <= 1.0
    assert min(window) >= -1.0
    # Full-scale audio really does reach close to the rails: a normalisation
    # bug that divided by the wrong constant would show up here.
    assert max(window) > 0.9


def test_silero_hysteresis_opens_and_closes_a_turn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    # One window is 32 ms. Scripted: 2 quiet, 4 speech, then quiet forever.
    session = FakeOrtSession([0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.05])
    install_fake_onnx(monkeypatch, session)
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")

    vad = SileroVAD(model_path=str(model), min_silence_ms=96, speech_pad_ms=0)
    events = drive(vad, sine_pcm(1000))

    assert kinds(events) == [VADEventKind.SPEECH_START, VADEventKind.SPEECH_END]
    # Speech began at window 2 => 64 ms; quiet began at window 6 => 192 ms.
    # Compared by hand rather than with pytest.approx: numpy is stubbed for
    # this test, and approx reaches into sys.modules["numpy"] when it is there.
    assert abs(events[0].at - 0.064) < 1e-6
    assert abs(events[1].at - 0.192) < 1e-6
    assert vad.speaking is False


def test_silero_ignores_probability_dips_inside_the_hysteresis_band(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """0.40 is below the 0.5 open threshold but above the 0.35 close threshold."""
    session = FakeOrtSession([0.9, 0.9, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.9])
    install_fake_onnx(monkeypatch, session)
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")

    vad = SileroVAD(model_path=str(model), min_silence_ms=64)
    events = drive(vad, sine_pcm(400))

    assert kinds(events) == [VADEventKind.SPEECH_START]
    assert vad.speaking is True


def test_silero_reset_drops_context_state_and_pending_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    session = FakeOrtSession([0.9])
    install_fake_onnx(monkeypatch, session)
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")

    vad = SileroVAD(model_path=str(model))
    drive(vad, sine_pcm(100))
    assert vad.speaking is True

    vad.reset()
    assert vad.speaking is False

    drive(vad, sine_pcm(40))
    # The first window after a reset starts from a zero context and the zeroed
    # state again: no audio from the previous path leaks across.
    after = session.feeds[-1]
    assert after["input"][0][:64] == [0.0] * 64
    assert after["state"] == "zeros(2, 1, 128)"


def test_silero_missing_model_file_names_both_ways_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    install_fake_onnx(monkeypatch, FakeOrtSession([0.0]))

    explicit = SileroVAD(model_path=str(tmp_path / "absent.onnx"))
    with pytest.raises(FileNotFoundError, match="model_path you supplied"):
        explicit.feed(framed(sine_pcm(100))[0])

    # With no explicit path, resolution falls to the silero-vad pip package,
    # which is not installed here.
    packaged = SileroVAD()
    with pytest.raises(FileNotFoundError) as exc_info:
        packaged.feed(framed(sine_pcm(100))[0])
    assert "pip install silero-vad" in str(exc_info.value)


# --------------------------------------------------------------------------
# TurnTaker
# --------------------------------------------------------------------------


def make_session() -> CallSession:
    agent = AgentSpec(name="test-agent", persona="You are a helpful test agent.")
    tick = iter(range(10_000))
    return CallSession(agent=agent, clock=lambda: float(next(tick)))


class Recorder:
    """Collects what a runtime would have been told to do."""

    def __init__(self) -> None:
        self.utterances: list[list[AudioFrame]] = []
        self.verdicts: list[InterruptionVerdict] = []

    async def utterance(self, frames: list[AudioFrame]) -> None:
        self.utterances.append(frames)

    async def interrupt(self, verdict: InterruptionVerdict) -> None:
        self.verdicts.append(verdict)


async def push_all(taker: TurnTaker, pcm: bytes) -> None:
    for frame in framed(pcm):
        await taker.push(frame)


async def test_turn_taker_emits_one_utterance_per_turn() -> None:
    recorder = Recorder()
    taker = TurnTaker(vad=EnergyVAD(), on_utterance=recorder.utterance)

    await push_all(
        taker,
        noise_pcm(400) + sine_pcm(400) + noise_pcm(700) + sine_pcm(400) + noise_pcm(700),
    )

    assert len(recorder.utterances) == 2
    assert taker.speaking is False
    for frames in recorder.utterances:
        joined = concat_frames(frames)
        assert joined is not None
        # Each utterance carries the 400 ms burst plus pre-roll and hangover,
        # and nothing like the whole 2.6 s stream.
        seconds = len(joined.pcm) / (SAMPLE_RATE * 2)
        assert 0.4 < seconds < 1.0


async def test_utterance_keeps_the_onset_the_vad_reported_late() -> None:
    """The pre-roll's reason to exist: no clipped first syllable."""
    recorder = Recorder()
    taker = TurnTaker(vad=EnergyVAD(min_speech_ms=200), on_utterance=recorder.utterance)

    await push_all(taker, noise_pcm(600) + sine_pcm(400) + noise_pcm(700))

    assert len(recorder.utterances) == 1
    frames = recorder.utterances[0]
    joined = concat_frames(frames)
    assert joined is not None
    # The whole tone survives: had buffering begun when SPEECH_START arrived,
    # the first 200 ms of evidence would already have been discarded.
    # A 400 ms sine spends about 80% of its samples above this level, so an
    # intact burst gives ~5100 of them and one clipped by the 200 ms detection
    # lag would give roughly half that.
    tone_samples = sum(1 for s in array.array("h", joined.pcm) if abs(s) > 3000)
    assert tone_samples > 4900


async def test_barge_in_fires_only_while_bot_audio_is_mid_playback() -> None:
    session = make_session()
    ledger = PlaybackLedger(session)
    recorder = Recorder()
    taker = TurnTaker(
        vad=EnergyVAD(),
        ledger=ledger,
        on_utterance=recorder.utterance,
        on_interrupt=recorder.interrupt,
    )

    # Turn one: the bot is mid-sentence when the caller starts. Genuine.
    text = "Your appointment is confirmed for Thursday the fourteenth at ten."
    ledger.utterance_started("u1", text)
    ledger.mark_played("u1", 10)
    await push_all(taker, noise_pcm(400) + sine_pcm(400) + noise_pcm(700))

    assert len(recorder.verdicts) == 1
    verdict = recorder.verdicts[0]
    assert verdict.genuine is True
    assert verdict.unheard_text is not None and "Thursday" in verdict.unheard_text
    assert verdict.context_annotation is not None
    assert len([e for e in session.history if isinstance(e, Interruption)]) == 1

    # Turn two: the bot finished first. Ordinary turn-taking, no interrupt.
    ledger.utterance_started("u2", "Anything else I can help with?")
    ledger.mark_played("u2", len("Anything else I can help with?"))
    ledger.utterance_finished("u2")
    await push_all(taker, sine_pcm(400) + noise_pcm(700))

    assert len(recorder.verdicts) == 1  # unchanged
    assert len(recorder.utterances) == 2
    assert len([e for e in session.history if isinstance(e, Interruption)]) == 1


async def test_grace_window_is_the_ledgers_call_not_the_turn_takers() -> None:
    """Playback essentially finished: jitter, not a barge-in."""
    session = make_session()
    ledger = PlaybackLedger(session, grace_chars=5)
    recorder = Recorder()
    taker = TurnTaker(
        vad=EnergyVAD(),
        ledger=ledger,
        on_utterance=recorder.utterance,
        on_interrupt=recorder.interrupt,
    )

    text = "All set, have a great day!"
    ledger.utterance_started("u1", text)
    ledger.mark_played("u1", len(text) - 3)

    await push_all(taker, noise_pcm(400) + sine_pcm(400) + noise_pcm(700))

    assert recorder.verdicts == []
    assert len(recorder.utterances) == 1


async def test_ledger_is_consulted_even_without_an_interrupt_callback() -> None:
    """The adjudication has side effects (the event, retiring the utterance)."""
    session = make_session()
    ledger = PlaybackLedger(session)
    taker = TurnTaker(vad=EnergyVAD(), ledger=ledger)

    ledger.utterance_started("u1", "A fairly long sentence the caller cuts off.")
    ledger.mark_played("u1", 4)
    await push_all(taker, noise_pcm(400) + sine_pcm(400) + noise_pcm(700))

    assert len([e for e in session.history if isinstance(e, Interruption)]) == 1
    # Retired: a second adjudication reports ordinary turn-taking.
    assert ledger.caller_started_speaking().genuine is False


async def test_no_ledger_means_pure_endpointing() -> None:
    recorder = Recorder()
    taker = TurnTaker(
        vad=EnergyVAD(), on_utterance=recorder.utterance, on_interrupt=recorder.interrupt
    )

    await push_all(taker, noise_pcm(400) + sine_pcm(400) + noise_pcm(700))

    assert len(recorder.utterances) == 1
    assert recorder.verdicts == []


async def test_flush_delivers_a_turn_cut_off_by_a_hang_up() -> None:
    recorder = Recorder()
    taker = TurnTaker(vad=EnergyVAD(), on_utterance=recorder.utterance)

    await push_all(taker, noise_pcm(400) + sine_pcm(400))  # line drops mid-word
    assert recorder.utterances == []

    await taker.flush()

    assert len(recorder.utterances) == 1
    assert taker.speaking is False
    # Flushing twice must not re-deliver the same audio.
    await taker.flush()
    assert len(recorder.utterances) == 1


async def test_idle_buffer_is_bounded_by_the_preroll() -> None:
    """Hours of hold music must not accumulate in memory."""
    taker = TurnTaker(vad=EnergyVAD(), preroll_ms=200)

    await push_all(taker, noise_pcm(4000))

    assert taker.speaking is False
    assert taker.elapsed == pytest.approx(4.0)
    # 200 ms of pre-roll plus the frame straddling the horizon, out of the
    # 4 s pushed: the buffer tracks recent audio, not the call.
    assert taker.buffered_seconds <= 0.25


async def test_overlong_speech_is_split_rather_than_buffered_forever() -> None:
    """A VAD held open by noise delivers in pieces instead of hoarding audio."""
    recorder = Recorder()
    taker = TurnTaker(
        vad=EnergyVAD(),
        on_utterance=recorder.utterance,
        max_utterance_seconds=1.0,
    )

    await push_all(taker, noise_pcm(400) + sine_pcm(3000) + noise_pcm(700))

    assert len(recorder.utterances) >= 3
    for frames in recorder.utterances:
        joined = concat_frames(frames)
        assert joined is not None
        assert len(joined.pcm) / (SAMPLE_RATE * 2) <= 1.5


def test_concat_frames_joins_or_returns_none() -> None:
    assert concat_frames([]) is None
    frames = framed(sine_pcm(100))
    joined = concat_frames(frames)
    assert joined is not None
    assert joined.pcm == b"".join(f.pcm for f in frames)
    assert joined.sample_rate == SAMPLE_RATE
    assert joined.channels == 1
