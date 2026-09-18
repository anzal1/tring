"""Tests for the STT adapters in providers/cloud/stt_extra.py.

No network: the WebSocket provider (AssemblyAI) is exercised through its pure
URL/header-building seams, and the REST providers (OpenAI, Sarvam) are
exercised with ``httpx.MockTransport``. Nothing here opens a real socket or
touches the network, per docs/EXTENDING.md's house testing pattern.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from tring.providers.base import AudioFrame
from tring.providers.cloud.stt_extra import (
    AssemblyAISTT,
    OpenAISTT,
    SarvamSTT,
    _endpoint_utterances,
)

# ---------------------------------------------------------------------------
# Registry registration
# ---------------------------------------------------------------------------


def test_stt_extra_registers_expected_provider_names() -> None:
    import tring.providers.cloud  # noqa: F401 -- import is the point of the test
    from tring.providers import registry

    available = registry.available("stt")
    assert ("stt", "assemblyai") in available
    assert ("stt", "openai_stt") in available
    assert ("stt", "sarvam") in available
    # Google Cloud Speech-to-Text v2 was skipped: its recognize endpoint only
    # accepts OAuth2/service-account auth, not a simple api_key_env pattern.
    assert ("stt", "google_stt") not in available


# ---------------------------------------------------------------------------
# _endpoint_utterances -- the shared batching seam for OpenAISTT/SarvamSTT
# ---------------------------------------------------------------------------


def _silence_frame(n_bytes: int = 320) -> AudioFrame:
    return AudioFrame(pcm=b"\x00\x00" * (n_bytes // 2), sample_rate=16000, channels=1)


def _loud_frame(n_bytes: int = 320) -> AudioFrame:
    # 16-bit little-endian samples near full scale: max amplitude, easily
    # above any silence_threshold used below.
    return AudioFrame(pcm=b"\xff\x7f" * (n_bytes // 2), sample_rate=16000, channels=1)


@pytest.mark.asyncio
async def test_endpoint_utterances_splits_on_trailing_silence() -> None:
    async def frames() -> AsyncIterator[AudioFrame]:
        for _ in range(5):
            yield _loud_frame()
        for _ in range(50):  # well past silence_seconds at 20ms/frame
            yield _silence_frame()
        for _ in range(5):
            yield _loud_frame()

    utterances = [
        pcm
        async for pcm, _sr, _ch in _endpoint_utterances(
            frames(),
            silence_threshold=0.01,
            silence_seconds=0.3,
            min_utterance_seconds=0.01,
            max_utterance_seconds=30.0,
        )
    ]
    # Two speech bursts separated by enough silence => two utterances. Each
    # utterance's buffer includes the loud frames it was built around (plus
    # whatever trailing silence padded it before the gate fired), so check
    # containment rather than exact byte equality.
    assert len(utterances) == 2
    assert _loud_frame().pcm in utterances[0]
    assert _loud_frame().pcm in utterances[1]


@pytest.mark.asyncio
async def test_endpoint_utterances_flushes_tail_when_stream_ends() -> None:
    async def frames() -> AsyncIterator[AudioFrame]:
        for _ in range(5):
            yield _loud_frame()
        # Stream ends mid-utterance (caller hung up) -- no trailing silence.

    utterances = [
        pcm
        async for pcm, _sr, _ch in _endpoint_utterances(
            frames(),
            silence_threshold=0.01,
            silence_seconds=0.3,
            min_utterance_seconds=0.01,
            max_utterance_seconds=30.0,
        )
    ]
    assert len(utterances) == 1
    assert utterances[0] == _loud_frame().pcm * 5


@pytest.mark.asyncio
async def test_endpoint_utterances_drops_utterance_shorter_than_minimum() -> None:
    async def frames() -> AsyncIterator[AudioFrame]:
        yield _loud_frame()  # 320 bytes = 10ms at 16kHz mono 16-bit
        for _ in range(50):
            yield _silence_frame()

    utterances = [
        pcm
        async for pcm, _sr, _ch in _endpoint_utterances(
            frames(),
            silence_threshold=0.01,
            silence_seconds=0.3,
            min_utterance_seconds=1.0,  # 10ms of speech never reaches this
            max_utterance_seconds=30.0,
        )
    ]
    assert utterances == []


# ---------------------------------------------------------------------------
# AssemblyAISTT -- pure URL/header seam (no socket opened)
# ---------------------------------------------------------------------------


def test_assemblyai_ws_url_carries_documented_query_params() -> None:
    stt = AssemblyAISTT(speech_model="universal-3-5-pro", sample_rate=16000)
    url = stt._ws_url()
    assert url.startswith("wss://streaming.assemblyai.com/v3/ws?")
    assert "sample_rate=16000" in url
    assert "encoding=pcm_s16le" in url
    assert "speech_model=universal-3-5-pro" in url
    assert "format_turns=true" in url


def test_assemblyai_ws_headers_carry_raw_key_without_bearer_prefix() -> None:
    stt = AssemblyAISTT()
    headers = stt._ws_headers("aai-test-key")
    assert headers == {"Authorization": "aai-test-key"}


def test_assemblyai_format_turns_false_is_reflected_in_url() -> None:
    stt = AssemblyAISTT(format_turns=False)
    assert "format_turns=false" in stt._ws_url()


# ---------------------------------------------------------------------------
# OpenAISTT -- httpx.MockTransport
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_stt_builds_request_and_parses_token_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-456")
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["content_type"] = request.headers.get("content-type", "")
        return httpx.Response(
            200,
            json={
                "text": "book a table for two",
                "usage": {"type": "tokens", "input_tokens": 14, "output_tokens": 5},
            },
        )

    stt = OpenAISTT(
        model="gpt-4o-transcribe",
        api_key_env="OPENAI_API_KEY",
        transport=httpx.MockTransport(handler),
        silence_seconds=0.3,
        min_utterance_seconds=0.01,
    )

    async def frames() -> AsyncIterator[AudioFrame]:
        for _ in range(5):
            yield _loud_frame()
        for _ in range(50):
            yield _silence_frame()

    results = [r async for r in stt.transcribe(frames())]

    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.openai.com/v1/audio/transcriptions"
    assert captured["headers"]["authorization"] == "Bearer sk-test-456"
    assert "multipart/form-data" in captured["content_type"]

    assert len(results) == 1
    assert results[0].text == "book a table for two"
    assert results[0].final is True
    usage_by_name = {u.unit_name: u for u in results[0].usage}
    assert usage_by_name["tokens_in"].units == 14
    assert usage_by_name["tokens_in"].estimated is False
    assert usage_by_name["tokens_out"].units == 5


@pytest.mark.asyncio
async def test_openai_stt_falls_back_to_exact_audio_seconds_without_usage_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-456")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "hello there"})

    stt = OpenAISTT(
        api_key_env="OPENAI_API_KEY",
        transport=httpx.MockTransport(handler),
        silence_seconds=0.3,
        min_utterance_seconds=0.01,
    )

    async def frames() -> AsyncIterator[AudioFrame]:
        for _ in range(5):
            yield _loud_frame()
        for _ in range(50):
            yield _silence_frame()

    results = [r async for r in stt.transcribe(frames())]
    assert len(results) == 1
    usage = results[0].usage
    assert len(usage) == 1
    assert usage[0].unit_name == "audio_seconds"
    assert usage[0].estimated is False
    # 5 loud frames (0.05s) plus the 30 trailing-silence frames (0.30s) that
    # accumulate in the same buffer before the silence gate fires = 0.35s.
    assert usage[0].units == pytest.approx(0.35)


@pytest.mark.asyncio
async def test_openai_stt_uses_base_url_override_for_groq_style_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-789")
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"text": "hi"})

    stt = OpenAISTT(
        model="whisper-large-v3",
        base_url="https://api.groq.com/openai",
        api_key_env="GROQ_API_KEY",
        transport=httpx.MockTransport(handler),
        silence_seconds=0.3,
        min_utterance_seconds=0.01,
    )

    async def frames() -> AsyncIterator[AudioFrame]:
        for _ in range(5):
            yield _loud_frame()
        for _ in range(50):
            yield _silence_frame()

    async for _ in stt.transcribe(frames()):
        pass

    assert captured["url"] == "https://api.groq.com/openai/v1/audio/transcriptions"


def test_openai_stt_missing_env_var_raises_clear_error() -> None:
    stt = OpenAISTT(api_key_env="SOME_MISSING_OPENAI_KEY")
    with pytest.raises(RuntimeError, match="SOME_MISSING_OPENAI_KEY"):
        import tring.providers.cloud.stt_extra as mod

        mod._api_key(stt._api_key_env, "OpenAISTT")


# ---------------------------------------------------------------------------
# SarvamSTT -- httpx.MockTransport
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sarvam_stt_builds_request_with_subscription_key_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SARVAM_API_KEY", "sarvam-test-key")
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={"transcript": "namaste", "language_code": "hi-IN"},
        )

    stt = SarvamSTT(
        api_key_env="SARVAM_API_KEY",
        language_code="hi-IN",
        transport=httpx.MockTransport(handler),
        silence_seconds=0.3,
        min_utterance_seconds=0.01,
    )

    async def frames() -> AsyncIterator[AudioFrame]:
        for _ in range(5):
            yield _loud_frame()
        for _ in range(50):
            yield _silence_frame()

    results = [r async for r in stt.transcribe(frames())]

    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.sarvam.ai/speech-to-text"
    assert captured["headers"]["api-subscription-key"] == "sarvam-test-key"
    assert "authorization" not in captured["headers"]

    assert len(results) == 1
    assert results[0].text == "namaste"
    assert results[0].language == "hi-IN"
    assert results[0].usage[0].unit_name == "audio_seconds"
    assert results[0].usage[0].estimated is False
    # 5 loud frames (0.05s) plus the 30 trailing-silence frames (0.30s) that
    # accumulate in the same buffer before the silence gate fires = 0.35s.
    assert results[0].usage[0].units == pytest.approx(0.35)


def test_sarvam_stt_missing_env_var_raises_clear_error() -> None:
    stt = SarvamSTT(api_key_env="SOME_MISSING_SARVAM_KEY")
    with pytest.raises(RuntimeError, match="SOME_MISSING_SARVAM_KEY"):
        import tring.providers.cloud.stt_extra as mod

        mod._api_key(stt._api_key_env, "SarvamSTT")


# ---------------------------------------------------------------------------
# STT_RATES
# ---------------------------------------------------------------------------


def test_stt_rates_cover_every_implemented_provider() -> None:
    from tring.events import CostComponent
    from tring.providers.cloud.stt_extra import STT_RATES

    providers = {rate.provider for rate in STT_RATES}
    assert providers == {"assemblyai", "openai_stt", "sarvam"}
    assert all(rate.component is CostComponent.STT for rate in STT_RATES)
    assert all(rate.as_of == "2026-09" for rate in STT_RATES)
    assert all(rate.price_per_unit > 0 for rate in STT_RATES)
