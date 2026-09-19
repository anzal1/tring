"""Generate a FreeSWITCH dialplan + ``mod_audio_stream`` wiring for a set of
DIDs, from a plain YAML/dict description.

``docs/TELEPHONY.md``'s "Generic SIP" section documents the bridge pattern
this module renders as text (SIP/RTP leg -> ``mod_audio_stream`` ->
``tring.transports.websocket``) and is explicit about what this repo does
*not* ship: "A FreeSWITCH dialplan or ``mod_audio_stream`` configuration
generator. Wiring a SIP trunk, a dialplan extension, and
``mod_audio_stream``'s ``uuid_audio_stream`` command together for a given
deployment is left to the operator." This module is that generator: pure
text rendering (no FreeSWITCH process, no XML parsing library, no network),
so it is fully unit-testable against golden files without a FreeSWITCH
install anywhere near CI.

``mod_audio_stream`` is a third-party module
(https://github.com/amigniter/mod_audio_stream); this repo does not vendor,
build, or load it. Two things below were verified live against that
project's own source, not written from memory:

- The ``uuid_audio_stream`` API command's argument order and allowed
  values, from its usage string in ``mod_audio_stream.c``\\'s
  ``SWITCH_ADD_API`` registration (fetched 2026-09-19 from
  https://raw.githubusercontent.com/amigniter/mod_audio_stream/main/mod_audio_stream.c):
  ``"<uuid> [start | stop | send_text | pause | resume | graceful-shutdown]
  [wss-url | path] [mono | mixed | stereo] [8000 | 16000] [metadata]"``.
  The sampling-rate argument is the literal integer ``8000`` or ``16000``
  (some third-party blog posts show a ``"16k"`` shorthand; the module's own
  usage string documents the numeric form, and it is the numeric form
  ``docs/TELEPHONY.md`` already tells operators to request, so that is what
  this generator emits).
- The module registers only an *API* command (``SWITCH_ADD_API``), no
  dialplan *application* (no ``SWITCH_ADD_APP``) -- so a dialplan cannot
  invoke it with a plain ``<action application="uuid_audio_stream" .../>``.
  The standard way to run an arbitrary API command exactly once a call is
  answered is FreeSWITCH core's own ``api_on_answer`` channel variable
  (https://developer.signalwire.com/freeswitch/Channel-Variables-Catalog/api_on_answer_16352805,
  fetched 2026-09-19: "executes an API (not an application) when the called
  party answers"), set via the ubiquitous ``mod_dptools`` ``set``
  application -- the same idiom used in ``mod_audio_stream``'s own
  ecosystem examples (e.g. the walkthrough at
  https://www.cyberpunk.tools/jekyll/update/2025/11/18/add-ai-voice-agent-to-freeswitch.html).
  ``park`` after ``answer`` is what keeps the channel alive so the media
  bug the API command attaches can stream in both directions for the rest
  of the call, instead of the dialplan falling through and hanging up.

Loading ``mod_audio_stream`` itself is a separate, one-time step
(``conf/autoload_configs/modules.conf.xml``) from wiring a specific DID to
a specific agent (``conf/dialplan/<context>/*.xml``); :func:`generate_dialplan`
therefore returns two independent files rather than trying to merge into
either of the operator's existing config trees.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, field_validator

MixType = Literal["mono", "mixed", "stereo"]

# The two sampling-rate values mod_audio_stream's own usage string documents
# -- see the module docstring for the exact citation. Anything else is
# rejected at config-validation time rather than silently passed through to
# a FreeSWITCH API command that would reject it live, mid-deployment.
_ALLOWED_SAMPLE_RATES: tuple[int, ...] = (8000, 16000)

#: Output file paths generate_dialplan writes to, relative to the operator's
#: FreeSWITCH ``conf/`` directory. Named module-level constants (not
#: reconstructed inline) so tests and callers reference the exact same
#: strings the generator does, rather than a second hand-typed copy that
#: could drift.
MODULES_CONF_SNIPPET_PATH = "autoload_configs/modules.conf.xml.append"


def _dialplan_path(context: str) -> str:
    """Where one context's generated extensions file lives.

    Matches stock FreeSWITCH layout: ``conf/dialplan/<context>/*.xml`` files
    are spliced into ``<context name="...">`` via that context's own
    ``<X-PRE-PROCESS cmd="include" data="<context>/*.xml"/>`` -- already
    present in a stock install, so this generator only ever adds a file
    under an existing context directory, never touches the include line.
    """
    return f"dialplan/{context}/tring-agents.xml"


class DidRoute(BaseModel):
    """One DID's route to a Tring agent over ``mod_audio_stream``.

    ``ws_url`` should point at :func:`tring.transports.websocket.serve`'s
    endpoint (the "Generic SIP" pattern in ``docs/TELEPHONY.md``): requesting
    16 kHz here makes ``mod_audio_stream``'s wire format identical to
    Tring's canonical ``AudioFrame`` format, so no resampling adapter is
    needed on this path at all (unlike the Twilio transport, which must
    resample 8 kHz mu-law both ways).
    """

    ws_url: str
    sample_rate: int = 16000
    mix_type: MixType = "mono"
    # Opaque string mod_audio_stream forwards to the websocket server
    # verbatim; defaults to the DID itself (see generate_dialplan) so a
    # multi-DID deployment can always tell which number an inbound
    # connection is for without parsing SIP headers on the agent side.
    metadata: str | None = None

    @field_validator("ws_url")
    @classmethod
    def _ws_url_scheme(cls, v: str) -> str:
        if not (v.startswith("ws://") or v.startswith("wss://")):
            raise ValueError(f"ws_url must start with ws:// or wss://, got {v!r}")
        return v

    @field_validator("sample_rate")
    @classmethod
    def _known_sample_rate(cls, v: int) -> int:
        if v not in _ALLOWED_SAMPLE_RATES:
            raise ValueError(
                f"sample_rate must be one of {_ALLOWED_SAMPLE_RATES} -- "
                "mod_audio_stream's own uuid_audio_stream usage string only "
                "documents these two (see this module's docstring)"
            )
        return v

    @field_validator("metadata")
    @classmethod
    def _metadata_has_no_quote(cls, v: str | None) -> str | None:
        # metadata is spliced, unescaped, into a single-quoted api_on_answer
        # value (see _did_extension_xml); a literal "'" would truncate the
        # api command FreeSWITCH actually runs, silently, mid-deployment.
        # Rejecting it here is cheaper than teaching every caller the wire
        # format's quoting rule.
        if v is not None and "'" in v:
            raise ValueError("metadata must not contain a single quote (\"'\")")
        return v


class FreeswitchDialplanConfig(BaseModel):
    """Top-level input to :func:`generate_dialplan`.

    ``dids`` keys are E.164 (or otherwise dialplan-matchable) destination
    numbers; ``context`` is the single FreeSWITCH dialplan context every DID
    in this config is routed under (a deployment routing DIDs across
    multiple contexts calls :func:`generate_dialplan` once per context and
    merges the resulting file dicts).
    """

    dids: dict[str, DidRoute]
    context: str = "default"

    @field_validator("dids")
    @classmethod
    def _at_least_one_did(cls, v: dict[str, DidRoute]) -> dict[str, DidRoute]:
        if not v:
            raise ValueError("dids must map at least one DID to a route")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> FreeswitchDialplanConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)


def _extension_name(did: str) -> str:
    """A FreeSWITCH-safe ``<extension name="...">`` value for one DID.

    Extension names are free-form XML attribute text, not a matched value
    (``destination_number`` regexes do the actual routing), but keeping them
    to a conventional identifier charset avoids surprising anyone grepping
    the generated XML or a FreeSWITCH ``xml_curl`` log with the raw DID.
    """
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", did.lstrip("+"))
    return f"tring_did_{sanitized}"


def _did_extension_xml(did: str, route: DidRoute) -> str:
    """Render one ``<extension>`` block wiring ``did`` to ``route``."""
    metadata = route.metadata if route.metadata is not None else did
    api_command = (
        f"uuid_audio_stream ${{uuid}} start {route.ws_url} "
        f"{route.mix_type} {route.sample_rate} {metadata}"
    )
    destination_expr = f"^{re.escape(did)}$"
    return (
        f'<extension name="{_extension_name(did)}">\n'
        f'  <condition field="destination_number" expression="{destination_expr}">\n'
        f'    <action application="set" data="api_on_answer=\'{api_command}\'"/>\n'
        f'    <action application="answer"/>\n'
        f'    <action application="park"/>\n'
        f"  </condition>\n"
        f"</extension>"
    )


def _dialplan_header(context: str) -> str:
    return (
        "<!--\n"
        "  Auto-generated by tring.telephony.freeswitch_gen.generate_dialplan.\n"
        "  Do not hand-edit: regenerate from the YAML/dict source instead, or\n"
        "  edits will be lost on the next run.\n"
        "\n"
        f"  Context: {context}. Drop this file at\n"
        f"  conf/{_dialplan_path(context)} in a stock FreeSWITCH install; the\n"
        "  context's existing <X-PRE-PROCESS cmd=\"include\" data=\"*.xml\"/>\n"
        "  already picks up new files here, no other config changes needed.\n"
        "\n"
        "  See tests/golden/freeswitch/README.md for what these fixtures\n"
        "  check, and docs/TELEPHONY.md's \"Generic SIP\" section for the\n"
        "  bridge pattern this file wires together end to end.\n"
        "-->"
    )


def _modules_conf_snippet() -> str:
    return (
        "<!--\n"
        "  Auto-generated by tring.telephony.freeswitch_gen.generate_dialplan.\n"
        '  Merge the <load module="mod_audio_stream"/> line below into your\n'
        "  existing conf/autoload_configs/modules.conf.xml (inside <modules>)\n"
        "  if it is not already present -- this file is a snippet to merge,\n"
        "  not a drop-in replacement for that file.\n"
        "\n"
        "  mod_audio_stream is a third-party module this repo does not vendor\n"
        "  or build; see https://github.com/amigniter/mod_audio_stream for\n"
        "  build/install instructions, and docs/TELEPHONY.md's \"What this\n"
        '  repo does not yet ship" section for exactly what "the FreeSWITCH\n'
        '  bridge pattern" does and does not mean here.\n'
        "\n"
        "  See tests/golden/freeswitch/README.md for what these fixtures\n"
        "  check.\n"
        "-->\n"
        '<load module="mod_audio_stream"/>'
    )


def generate_dialplan(config: FreeswitchDialplanConfig | dict[str, Any]) -> dict[str, str]:
    """Render a FreeSWITCH dialplan + ``mod_audio_stream`` load snippet.

    ``config`` is either an already-validated :class:`FreeswitchDialplanConfig`
    or a plain dict (typically ``yaml.safe_load`` of a file like::

        context: default
        dids:
          "+15551234567":
            ws_url: "wss://voice.example.com/twilio"
          "+15557654321":
            ws_url: "wss://voice.example.com/agent2"
            sample_rate: 8000
            mix_type: mixed
            metadata: "agent2"

    Returns a ``{relative_path: file_contents}`` dict, paths relative to a
    FreeSWITCH install's ``conf/`` directory (see :data:`MODULES_CONF_SNIPPET_PATH`
    and :func:`_dialplan_path`). This is pure text generation -- nothing
    here touches a filesystem or a FreeSWITCH process; callers decide how
    (or whether) to write these out.
    """
    validated = (
        config
        if isinstance(config, FreeswitchDialplanConfig)
        else FreeswitchDialplanConfig.model_validate(config)
    )

    extensions = "\n\n".join(
        _did_extension_xml(did, route) for did, route in validated.dids.items()
    )
    dialplan_xml = f"{_dialplan_header(validated.context)}\n\n{extensions}\n"

    return {
        _dialplan_path(validated.context): dialplan_xml,
        MODULES_CONF_SNIPPET_PATH: f"{_modules_conf_snippet()}\n",
    }


__all__ = [
    "MODULES_CONF_SNIPPET_PATH",
    "DidRoute",
    "FreeswitchDialplanConfig",
    "generate_dialplan",
]
