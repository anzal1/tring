# FreeSWITCH dialplan golden files

These fixtures are the exact expected output of
`tring.telephony.freeswitch_gen.generate_dialplan()` for two input
configurations, checked byte-for-byte by `tests/test_telephony.py`. Every
generated file's own header comment points back here so anyone who opens a
deployed copy of one of these files (not just this test) can find out what
produced it and why.

## Fixtures

- `single_did/` — one DID, all defaults (`context: default`,
  `sample_rate: 16000`, `mix_type: mono`, `metadata` defaulting to the DID).
- `multi_did_custom_context/` — two DIDs under a non-default context
  (`ivr`), exercising every non-default `DidRoute` field (`sample_rate: 8000`,
  `mix_type: mixed`, an explicit `metadata`) alongside a second DID left on
  every default, so both code paths through `_did_extension_xml` are covered
  in one fixture pair.

Each fixture directory mirrors the exact relative paths
`generate_dialplan()` returns as dict keys (e.g.
`dialplan/<context>/tring-agents.xml`,
`autoload_configs/modules.conf.xml.append`) — a stock FreeSWITCH install's
`conf/` directory layout, not a directory structure invented for this test.

## Regenerating after an intentional output change

If `freeswitch_gen.py`'s rendering changes on purpose, regenerate these
files rather than hand-editing them (the header comment inside each one
says the same thing to anyone who finds a deployed copy):

```bash
cd /Users/anzalhussainabidi/personal/tring
.venv/bin/python - <<'PY'
import pathlib
from tring.telephony.freeswitch_gen import generate_dialplan

single = {"context": "default", "dids": {
    "+15551234567": {"ws_url": "wss://voice.example.com/twilio"},
}}
multi = {"context": "ivr", "dids": {
    "+15557654321": {
        "ws_url": "wss://voice.example.com/agent2",
        "sample_rate": 8000,
        "mix_type": "mixed",
        "metadata": "agent2",
    },
    "+442071234567": {"ws_url": "ws://10.0.0.5:8080/agent1"},
}}

for name, cfg in [("single_did", single), ("multi_did_custom_context", multi)]:
    for relpath, contents in generate_dialplan(cfg).items():
        dest = pathlib.Path("tests/golden/freeswitch") / name / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(contents, encoding="utf-8")
PY
```

Then re-read the diff carefully: a golden-file test only catches
*unintentional* drift, so the regeneration step is exactly the moment a real
behavior change could slip past review unnoticed.
