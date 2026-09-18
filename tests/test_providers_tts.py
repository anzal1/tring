"""Tests for the tts-pack providers: CartesiaTTS, OpenAITTS, SarvamTTS
(providers/cloud/tts_extra.py) and PiperTTS (providers/local/tts_extra.py).

No network, no real API keys, no real Piper model: HTTP providers are
driven through ``httpx.MockTransport`` (asserting the exact request that
would go over the wire), and PiperTTS is driven through a fake ``piper``
module injected into ``sys.modules`` (asserting the exact call it would
make against the real package's documented API).
"""

from __future__ import annotations

import base64
import io
import struct
import sys
import wave
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from tring.providers import registry
from tring.providers.base import TTSChunk
from tring.providers.cloud.tts_extra import (
    TTS_RATES as CLOUD_TTS_RATES,
)
from tring.providers.cloud.tts_extra import (
    CartesiaTTS,
    OpenAITTS,
    SarvamTTS,
    _buffer_sentences,
)


async def _iter(*chunks: str) -> AsyncIterator[str]:
    for chunk in chunks:
        yield chunk


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_cloud_tts_extra_registers_expected_names() -> None:
    import tring.providers.cloud  # noqa: F401 -- import is the point of the test

    available = registry.available("tts")
    assert ("tts", "cartesia") in available
    assert ("tts", "openai_tts") in available
    assert ("tts", "sarvam_tts") in available


def test_local_tts_extra_registers_piper() -> None:
    import tring.providers.local  # noqa: F401 -- import is the point of the test

    assert ("tts", "piper") in registry.available("tts")


# ---------------------------------------------------------------------------
# sentence buffering (shared by all three cloud providers)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_buffer_sentences_flushes_on_sentence_boundary() -> None:
    # Each delta's *end* is checked against the sentence-boundary regex, so
    # (as with real token-by-token streaming) a boundary lands exactly at a
    # delta edge rather than mid-delta.
    segments = [
        s
        async for s in _buffer_sentences(
            _iter("Hello there", ", how are you today?", " Fine, thanks."),
            min_chars=5,
            max_chars=200,
        )
    ]
    assert segments == ["Hello there, how are you today?", "Fine, thanks."]


@pytest.mark.asyncio
async def test_buffer_sentences_flushes_on_max_chars_at_word_boundary() -> None:
    # Word-by-word deltas (with trailing spaces), no sentence punctuation
    # anywhere: the max-chars/word-boundary rule is the only thing that can
    # flush before the stream ends.
    words = ["one ", "two ", "three ", "four ", "five ", "six ", "seven ", "eight "]
    segments = [
        s async for s in _buffer_sentences(_iter(*words), min_chars=5, max_chars=15)
    ]
    assert len(segments) >= 2
    assert "".join(f"{s} " for s in segments) == "".join(words)


@pytest.mark.asyncio
async def test_buffer_sentences_flushes_trailing_text_with_no_boundary() -> None:
    segments = [s async for s in _buffer_sentences(_iter("no punctuation here"), 5, 200)]
    assert segments == ["no punctuation here"]


# ---------------------------------------------------------------------------
# CartesiaTTS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cartesia_tts_builds_request_and_reports_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MY_CARTESIA_KEY", "sk_car_test")
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request.read()
        captured.append(
            {
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "json": __import__("json").loads(request.content),
            }
        )
        return httpx.Response(200, content=b"\x01\x02\x03\x04")

    tts = CartesiaTTS(
        api_key_env="MY_CARTESIA_KEY",
        model_id="sonic-3",
        voice_id="voice-abc",
        transport=httpx.MockTransport(handler),
    )

    chunks: list[TTSChunk] = [
        c async for c in tts.synthesize(_iter("Hello there.", " Bye now."))
    ]

    assert len(captured) == 2  # one POST per sentence-ish segment
    first = captured[0]
    assert first["method"] == "POST"
    assert first["url"] == "https://api.cartesia.ai/tts/bytes"
    assert first["headers"]["x-api-key"] == "sk_car_test"
    assert first["headers"]["cartesia-version"] == "2026-08-14"
    assert first["json"]["model_id"] == "sonic-3"
    assert first["json"]["transcript"] == "Hello there."
    assert first["json"]["voice"] == {"id": "voice-abc"}
    assert first["json"]["output_format"] == {
        "container": "raw",
        "encoding": "pcm_s16le",
        "sample_rate": 16000,
    }

    assert all(c.frame.sample_rate == 16000 and c.frame.channels == 1 for c in chunks)
    assert b"".join(c.frame.pcm for c in chunks) == b"\x01\x02\x03\x04" * 2

    usage_chunks = [c for c in chunks if c.usage]
    assert len(usage_chunks) == 2  # one usage-bearing chunk per segment
    first_usage = usage_chunks[0].usage[0]
    assert first_usage.unit_name == "tts_chars"
    assert first_usage.estimated is False
    assert first_usage.units == float(len("Hello there."))


@pytest.mark.asyncio
async def test_cartesia_tts_without_voice_id_raises_clear_error() -> None:
    tts = CartesiaTTS(api_key_env="X")
    with pytest.raises(ValueError, match="requires a voice id"):
        async for _ in tts.synthesize(_iter("hi")):
            pass


@pytest.mark.asyncio
async def test_cartesia_tts_missing_env_var_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SOME_MISSING_CARTESIA_KEY", raising=False)
    tts = CartesiaTTS(api_key_env="SOME_MISSING_CARTESIA_KEY", voice_id="v1")
    with pytest.raises(RuntimeError, match="SOME_MISSING_CARTESIA_KEY"):
        async for _ in tts.synthesize(_iter("hi")):
            pass


# ---------------------------------------------------------------------------
# OpenAITTS
# ---------------------------------------------------------------------------


def _sine_pcm16(num_samples: int) -> bytes:
    """A trivial, deterministic 16-bit PCM buffer (no numpy/audio math)."""
    return struct.pack(f"<{num_samples}h", *[(i % 100) * 100 for i in range(num_samples)])


@pytest.mark.asyncio
async def test_openai_tts_builds_request_and_resamples_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MY_OPENAI_KEY", "sk-test")
    captured: list[dict[str, Any]] = []
    fake_pcm_24k = _sine_pcm16(2400)  # 100ms at 24kHz

    def handler(request: httpx.Request) -> httpx.Response:
        request.read()
        captured.append(
            {
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "json": __import__("json").loads(request.content),
            }
        )
        return httpx.Response(200, content=fake_pcm_24k)

    tts = OpenAITTS(
        api_key_env="MY_OPENAI_KEY",
        model="tts-1",
        voice="alloy",
        transport=httpx.MockTransport(handler),
    )

    chunks = [c async for c in tts.synthesize(_iter("Hello there."))]

    assert len(captured) == 1
    req = captured[0]
    assert req["method"] == "POST"
    assert req["url"] == "https://api.openai.com/v1/audio/speech"
    assert req["headers"]["authorization"] == "Bearer sk-test"
    assert req["json"] == {
        "model": "tts-1",
        "input": "Hello there.",
        "voice": "alloy",
        "response_format": "pcm",
    }

    assert len(chunks) == 1
    frame = chunks[0].frame
    assert frame.sample_rate == 16000
    assert frame.channels == 1
    # 2400 samples at 24kHz -> ~1600 samples at 16kHz (downsampled by 2/3).
    assert 1500 <= len(frame.pcm) // 2 <= 1700

    usage = chunks[0].usage[0]
    assert usage.unit_name == "tts_chars"
    assert usage.estimated is False
    assert usage.units == float(len("Hello there."))
    assert usage.model == "tts-1"


@pytest.mark.asyncio
async def test_openai_tts_voice_argument_overrides_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MY_OPENAI_KEY", "sk-test")
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        request.read()
        captured["json"] = __import__("json").loads(request.content)
        return httpx.Response(200, content=_sine_pcm16(100))

    tts = OpenAITTS(
        api_key_env="MY_OPENAI_KEY", voice="alloy", transport=httpx.MockTransport(handler)
    )
    async for _ in tts.synthesize(_iter("hi."), voice="nova"):
        pass

    assert captured["json"]["voice"] == "nova"


# ---------------------------------------------------------------------------
# SarvamTTS
# ---------------------------------------------------------------------------


def _wav_b64(pcm: bytes, sample_rate: int = 16000, channels: int = 1) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.mark.asyncio
async def test_sarvam_tts_builds_request_and_unwraps_wav_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MY_SARVAM_KEY", "sarvam-test")
    fake_pcm = struct.pack("<4h", 10, 20, 30, 40)
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request.read()
        captured.append(
            {
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "json": __import__("json").loads(request.content),
            }
        )
        body = {"audios": [_wav_b64(fake_pcm)], "request_id": "req-1"}
        return httpx.Response(200, json=body)

    tts = SarvamTTS(
        api_key_env="MY_SARVAM_KEY",
        model="bulbul:v3",
        speaker="shubh",
        language_code="en-IN",
        transport=httpx.MockTransport(handler),
    )

    chunks = [c async for c in tts.synthesize(_iter("Namaste."))]

    assert len(captured) == 1
    req = captured[0]
    assert req["method"] == "POST"
    assert req["url"] == "https://api.sarvam.ai/text-to-speech"
    assert req["headers"]["api-subscription-key"] == "sarvam-test"
    assert req["json"] == {
        "text": "Namaste.",
        "language_code": "en-IN",
        "model": "bulbul:v3",
        "speaker": "shubh",
        "speech_sample_rate": 16000,
    }

    assert len(chunks) == 1
    frame = chunks[0].frame
    assert frame.pcm == fake_pcm
    assert frame.sample_rate == 16000
    assert frame.channels == 1

    usage = chunks[0].usage[0]
    assert usage.unit_name == "tts_chars"
    assert usage.estimated is False
    assert usage.units == float(len("Namaste."))


@pytest.mark.asyncio
async def test_sarvam_tts_missing_env_var_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SOME_MISSING_SARVAM_KEY", raising=False)
    tts = SarvamTTS(api_key_env="SOME_MISSING_SARVAM_KEY")
    with pytest.raises(RuntimeError, match="SOME_MISSING_SARVAM_KEY"):
        async for _ in tts.synthesize(_iter("hi")):
            pass


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------


def test_cloud_tts_rates_cover_every_registered_provider() -> None:
    rated_providers = {r.provider for r in CLOUD_TTS_RATES}
    assert rated_providers == {"cartesia", "openai_tts", "sarvam_tts"}
    cartesia_rate = next(r for r in CLOUD_TTS_RATES if r.provider == "cartesia")
    # Cartesia's public pricing page does not expose a verifiable
    # per-character rate for Sonic TTS (see tts_extra.py's comment) -- 0.0
    # is the honest value, not a filled-in guess.
    assert cartesia_rate.price_per_unit == 0.0


def test_local_tts_rates_price_piper_at_zero() -> None:
    from tring.providers.local.tts_extra import TTS_RATES as LOCAL_TTS_RATES

    piper_rate = next(r for r in LOCAL_TTS_RATES if r.provider == "piper")
    assert piper_rate.price_per_unit == 0.0
    assert piper_rate.unit_name == "tts_chars"


# ---------------------------------------------------------------------------
# PiperTTS -- driven through a fake `piper` module, no real model/weights.
# ---------------------------------------------------------------------------


class _FakeChunk:
    def __init__(self, audio_int16_bytes: bytes, sample_rate: int) -> None:
        self.audio_int16_bytes = audio_int16_bytes
        self.sample_rate = sample_rate
        self.sample_width = 2
        self.sample_channels = 1


class _FakeVoice:
    def __init__(self, calls: list[tuple[str, Any]], pcm: bytes, sample_rate: int) -> None:
        self._calls = calls
        self._pcm = pcm
        self._sample_rate = sample_rate

    def synthesize(self, text: str, syn_config: Any = None) -> Any:
        self._calls.append((text, syn_config))
        yield _FakeChunk(self._pcm, self._sample_rate)


class _FakeSynthesisConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeNdArray:
    """Just enough of numpy's ndarray surface for the identity resample path
    (native rate == wire rate) exercised by the test below."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def astype(self, _dtype: Any) -> _FakeNdArray:
        return self

    def tobytes(self) -> bytes:
        return self._data


def _install_fake_piper_and_numpy(
    monkeypatch: pytest.MonkeyPatch, pcm: bytes, sample_rate: int
) -> list[tuple[str, Any]]:
    calls: list[tuple[str, Any]] = []

    def _load(path: str, use_cuda: bool = False) -> _FakeVoice:
        return _FakeVoice(calls, pcm, sample_rate)

    fake_piper = type(sys)("piper")
    fake_piper.PiperVoice = type(  # type: ignore[attr-defined]
        "PiperVoice", (), {"load": staticmethod(_load)}
    )
    fake_piper.SynthesisConfig = _FakeSynthesisConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "piper", fake_piper)

    fake_numpy = type(sys)("numpy")
    fake_numpy.frombuffer = lambda buf, dtype=None: _FakeNdArray(bytes(buf))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)

    return calls


@pytest.mark.asyncio
async def test_piper_tts_synthesizes_via_documented_api_and_reports_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tring.providers.local.tts_extra import PiperTTS

    fake_pcm = struct.pack("<4h", 1, 2, 3, 4)
    calls = _install_fake_piper_and_numpy(monkeypatch, fake_pcm, sample_rate=16000)

    tts = PiperTTS(model_path="/models/en_US-lessac-medium.onnx")
    chunks = [c async for c in tts.synthesize(_iter("Hello there."))]

    assert len(calls) == 1
    called_text, syn_config = calls[0]
    assert called_text == "Hello there."
    assert isinstance(syn_config, _FakeSynthesisConfig)

    assert b"".join(c.frame.pcm for c in chunks) == fake_pcm
    assert all(c.frame.sample_rate == 16000 and c.frame.channels == 1 for c in chunks)

    usage_chunks = [c for c in chunks if c.usage]
    assert len(usage_chunks) == 1  # usage rides only the first frame of the segment
    usage = usage_chunks[0].usage[0]
    assert usage.unit_name == "tts_chars"
    assert usage.estimated is False
    assert usage.units == float(len("Hello there."))


def test_piper_tts_missing_package_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from tring.providers.local.tts_extra import PiperTTS

    monkeypatch.setitem(sys.modules, "piper", None)  # forces ModuleNotFoundError on import
    tts = PiperTTS(model_path="/models/whatever.onnx")
    with pytest.raises(ImportError, match="pip install piper-tts"):
        tts._load_voice()
