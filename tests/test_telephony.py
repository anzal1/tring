"""Tests for ``tring.telephony``: the FreeSWITCH dialplan generator
(``freeswitch_gen.py``) and the Twilio ``transfer_call`` tool (``transfer.py``).

No network, no FreeSWITCH process, no Twilio account: ``freeswitch_gen`` is
pure text rendering checked against golden files, and ``transfer.py``'s HTTP
call is driven through an ``httpx.MockTransport`` seam -- the same house
pattern ``tests/test_providers_llm.py`` uses for ``providers/cloud/llm_extra.py``.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest
from pydantic import ValidationError

from tring.agent import AgentSpec
from tring.events import SessionEnded, SessionTransferred
from tring.primitives.choreography import augment_tool_schema
from tring.session import CallSession
from tring.telephony.freeswitch_gen import (
    MODULES_CONF_SNIPPET_PATH,
    DidRoute,
    FreeswitchDialplanConfig,
    generate_dialplan,
)
from tring.telephony.transfer import TRANSFER_TOOL, TransferError, make_transfer_handler

GOLDEN_ROOT = Path(__file__).parent / "golden" / "freeswitch"

# ---------------------------------------------------------------------------
# freeswitch_gen: golden-file tests
# ---------------------------------------------------------------------------

_SINGLE_DID_CONFIG: dict[str, Any] = {
    "context": "default",
    "dids": {
        "+15551234567": {"ws_url": "wss://voice.example.com/twilio"},
    },
}

_MULTI_DID_CONFIG: dict[str, Any] = {
    "context": "ivr",
    "dids": {
        "+15557654321": {
            "ws_url": "wss://voice.example.com/agent2",
            "sample_rate": 8000,
            "mix_type": "mixed",
            "metadata": "agent2",
        },
        "+442071234567": {
            "ws_url": "ws://10.0.0.5:8080/agent1",
        },
    },
}


def _read_golden_tree(fixture_name: str) -> dict[str, str]:
    """Every file under ``golden/freeswitch/<fixture_name>/``, keyed by the
    path relative to that directory -- exactly the shape ``generate_dialplan``
    returns, so the two can be compared directly."""
    root = GOLDEN_ROOT / fixture_name
    return {
        str(path.relative_to(root)): path.read_text(encoding="utf-8")
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize(
    ("fixture_name", "config"),
    [
        ("single_did", _SINGLE_DID_CONFIG),
        ("multi_did_custom_context", _MULTI_DID_CONFIG),
    ],
)
def test_generate_dialplan_matches_golden_files_exactly(
    fixture_name: str, config: dict[str, Any]
) -> None:
    expected = _read_golden_tree(fixture_name)
    actual = generate_dialplan(config)
    assert actual == expected


def test_generate_dialplan_accepts_an_already_validated_config() -> None:
    """The dict path and the pre-validated-model path must render identically --
    generate_dialplan is one renderer, not two, regardless of input shape."""
    validated = FreeswitchDialplanConfig.model_validate(_SINGLE_DID_CONFIG)
    assert generate_dialplan(validated) == generate_dialplan(_SINGLE_DID_CONFIG)


def test_generate_dialplan_output_keys_are_the_documented_paths() -> None:
    out = generate_dialplan(_SINGLE_DID_CONFIG)
    assert MODULES_CONF_SNIPPET_PATH in out
    assert "dialplan/default/tring-agents.xml" in out


# ---------------------------------------------------------------------------
# freeswitch_gen: config validation
# ---------------------------------------------------------------------------


def test_did_route_rejects_a_non_websocket_url() -> None:
    with pytest.raises(ValidationError, match="ws_url"):
        DidRoute(ws_url="https://voice.example.com/twilio")


@pytest.mark.parametrize("bad_rate", [4000, 22050, 44100, 0])
def test_did_route_rejects_an_undocumented_sample_rate(bad_rate: int) -> None:
    with pytest.raises(ValidationError, match="sample_rate"):
        DidRoute(ws_url="wss://voice.example.com/twilio", sample_rate=bad_rate)


def test_did_route_rejects_metadata_containing_a_single_quote() -> None:
    """A literal ' would truncate the single-quoted api_on_answer value the
    generator emits, silently, mid-deployment -- see _did_extension_xml."""
    with pytest.raises(ValidationError, match="metadata"):
        DidRoute(ws_url="wss://voice.example.com/twilio", metadata="o'brien")


def test_freeswitch_dialplan_config_rejects_an_empty_dids_map() -> None:
    with pytest.raises(ValidationError, match="dids"):
        FreeswitchDialplanConfig(dids={})


def test_freeswitch_dialplan_config_from_yaml_round_trips(tmp_path: Path) -> None:
    yaml_path = tmp_path / "freeswitch.yaml"
    yaml_path.write_text(
        "context: default\n"
        "dids:\n"
        '  "+15551234567":\n'
        "    ws_url: wss://voice.example.com/twilio\n",
        encoding="utf-8",
    )
    config = FreeswitchDialplanConfig.from_yaml(yaml_path)
    assert config.context == "default"
    assert config.dids["+15551234567"].ws_url == "wss://voice.example.com/twilio"
    assert generate_dialplan(config) == generate_dialplan(_SINGLE_DID_CONFIG)


def test_generate_dialplan_sanitizes_extension_names_but_keeps_the_regex_exact() -> None:
    out = generate_dialplan(_SINGLE_DID_CONFIG)
    xml = out["dialplan/default/tring-agents.xml"]
    assert 'name="tring_did_15551234567"' in xml
    # The regex match itself must stay byte-exact (escaped) even though the
    # human-readable extension name is sanitized -- these are not the same
    # string and must not be conflated.
    assert r'expression="^\+15551234567$"' in xml


# ---------------------------------------------------------------------------
# transfer.py: TRANSFER_TOOL contract
# ---------------------------------------------------------------------------


def test_transfer_tool_requires_destination_and_is_choreographed() -> None:
    assert TRANSFER_TOOL.name == "transfer_call"
    assert TRANSFER_TOOL.choreographed is True
    assert TRANSFER_TOOL.parameters["required"] == ["destination"]


def test_transfer_tool_schema_gains_choreography_fields() -> None:
    """TRANSFER_TOOL composes with the existing choreography primitive with
    no special-casing -- it is just a ToolDef like any other."""
    schema = augment_tool_schema(TRANSFER_TOOL)
    assert "waiting_message" in schema["properties"]
    assert "destination" in schema["properties"]
    assert set(schema["required"]) >= {
        "destination",
        "waiting_message",
        "spoken_mode",
        "post_tool_response",
    }


# ---------------------------------------------------------------------------
# transfer.py: make_transfer_handler
# ---------------------------------------------------------------------------


def _session() -> CallSession:
    return CallSession(AgentSpec(name="transfer-test", persona="You are a test agent."))


def _mock_client(status_code: int, body: str = "") -> tuple[httpx.AsyncClient, dict[str, Any]]:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        # Twilio's REST API takes application/x-www-form-urlencoded bodies
        # (verified live, see transfer.py's module docstring); parse_qs, not
        # httpx.QueryParams, matches that encoding's literal-'+'-means-space
        # rule, which matters here because a transfer destination is itself
        # an E.164 number starting with '+'.
        parsed = parse_qs(request.content.decode())
        captured["form"] = {k: v[0] for k, v in parsed.items()}
        return httpx.Response(status_code, text=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), captured


async def test_transfer_to_a_phone_number_builds_dial_twiml_and_emits_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC_test")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "secret_token")
    client, captured = _mock_client(200)
    session = _session()

    handler = make_transfer_handler(session, "CA_test123", client=client)
    result = await handler({"destination": "+14155551234"})

    assert captured["method"] == "POST"
    assert (
        captured["url"] == "https://api.twilio.com/2010-04-01/Accounts/AC_test/Calls/CA_test123.json"
    )
    # Basic auth: decode the header back to "AC_test:secret_token" rather
    # than hand-computing the expected base64 string, so the assertion reads
    # as "these are the credentials that went over the wire," not a second
    # encoding this test could get subtly wrong in the same way as the code
    # under test.
    scheme, _, encoded = captured["headers"]["authorization"].partition(" ")
    assert scheme == "Basic"
    assert base64.b64decode(encoded).decode() == "AC_test:secret_token"

    assert "Twiml" in captured["form"]
    assert "<Dial>+14155551234</Dial>" in captured["form"]["Twiml"]
    assert "Url" not in captured["form"]

    assert result == {"transferred_to": "+14155551234", "call_sid": "CA_test123"}

    transferred = [e for e in session.history if isinstance(e, SessionTransferred)]
    ended = [e for e in session.history if isinstance(e, SessionEnded)]
    assert transferred == [
        SessionTransferred(
            session_id=session.session_id,
            at=transferred[0].at,
            target="+14155551234",
        )
    ]
    assert ended == [
        SessionEnded(session_id=session.session_id, at=ended[0].at, reason="transferred")
    ]
    # Order matters: a consumer watching the event stream must see *why*
    # before it sees *that the session ended*.
    assert session.history.index(transferred[0]) < session.history.index(ended[0])


async def test_transfer_to_a_twiml_url_sends_url_not_inline_twiml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC_test")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "secret_token")
    client, captured = _mock_client(200)
    session = _session()

    handler = make_transfer_handler(session, "CA_test123", client=client)
    await handler({"destination": "https://example.com/transfer.xml"})

    assert captured["form"]["Url"] == "https://example.com/transfer.xml"
    assert "Twiml" not in captured["form"]


async def test_transfer_raises_and_emits_nothing_on_a_twilio_error_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC_test")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "secret_token")
    client, _ = _mock_client(404, body=json.dumps({"message": "call not found"}))
    session = _session()

    handler = make_transfer_handler(session, "CA_missing", client=client)
    with pytest.raises(TransferError, match="404"):
        await handler({"destination": "+14155551234"})

    assert session.history == []


async def test_transfer_rejects_an_empty_destination_before_making_any_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC_test")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "secret_token")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    session = _session()
    transfer_handler = make_transfer_handler(session, "CA_test123", client=client)

    with pytest.raises(ValueError, match="destination"):
        await transfer_handler({"destination": "   "})

    assert calls == 0
    assert session.history == []


async def test_transfer_requires_the_account_sid_and_auth_token_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
    monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    session = _session()
    transfer_handler = make_transfer_handler(session, "CA_test123", client=client)

    with pytest.raises(RuntimeError, match="TWILIO_ACCOUNT_SID"):
        await transfer_handler({"destination": "+14155551234"})

    assert calls == 0


async def test_transfer_handler_honors_custom_env_var_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MY_SID", "AC_custom")
    monkeypatch.setenv("MY_TOKEN", "custom_token")
    client, captured = _mock_client(200)
    session = _session()

    handler = make_transfer_handler(
        session,
        "CA_test123",
        account_sid_env="MY_SID",
        auth_token_env="MY_TOKEN",
        client=client,
    )
    await handler({"destination": "+14155551234"})

    assert "AC_custom" in captured["url"]
