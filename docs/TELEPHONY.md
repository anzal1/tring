# Telephony

This is the production guide for putting a real phone number in front of a
Tring agent. Two paths are covered: Twilio Media Streams, which
`tring.transports.twilio.serve_twilio` implements directly, and generic SIP
via a FreeSWITCH bridge, which today is a documented pattern rather than
shipped code (see "Generic SIP" below for exactly what that means).

Both paths end at the same place: a `RuntimeAdapter` that only ever sees the
stack's canonical wire format, 16 kHz mono 16-bit linear PCM
(`runtimes/base.py`'s `AudioFrame`). Everything in this document is about
getting telephone audio, which is never that format, converted to and from
it.

---

## Twilio Media Streams

### How the pieces fit together

```
caller  --PSTN-->  Twilio  --WebSocket, G.711 mu-law @ 8kHz-->  serve_twilio()
                                                                      |
                                                            decode + upsample
                                                                      |
                                                                      v
                                                              RuntimeAdapter
                                                            (push_audio / on_bot_audio)
                                                                      |
                                                          downsample + encode + chunk
                                                                      |
                                                                      v
caller  <--PSTN--  Twilio  <--WebSocket, G.711 mu-law @ 8kHz--  serve_twilio()
```

`tring.transports.twilio.serve_twilio(runtime_factory, host, port, path)`
opens a WebSocket server. Twilio connects to it once per call, sends a
`start` message describing the stream, then a `media` message for every
20ms of caller audio it has, and a `stop` message when the call ends. The
codec (`tring.transports.mulaw`) and the message handling
(`tring.transports.twilio`) were built and verified against Twilio's own
schema at <https://www.twilio.com/docs/voice/media-streams/websocket-messages>.

### 1. TwiML: point a call at the stream

A Twilio phone number needs a webhook that answers with TwiML containing a
`<Connect><Stream>` verb. `<Connect>` (not `<Start>`) is what makes the
stream bidirectional: Twilio will both send you the caller's audio and
accept audio back from you to play into the call. A minimal webhook
response looks like:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://your-host.example.com/twilio" />
  </Connect>
</Response>
```

The webhook itself is just an HTTP endpoint your application serves (any
web framework works; it does not need to be Tring code) that Twilio calls
when the phone number receives an incoming call, and that returns the XML
above. Configure that endpoint's URL under the phone number's "A call
comes in" setting in the Twilio console, or set it via the API when
provisioning a number programmatically.

### 2. Run the transport

```python
import asyncio

from tring.agent import AgentSpec
from tring.runtimes.cascade import CascadeRuntime
from tring.session import CallSession
from tring.transports.twilio import serve_twilio

agent = AgentSpec.from_yaml("agent.yaml")


def factory(_: CallSession) -> CascadeRuntime:
    # The placeholder session serve_twilio hands in is discarded here in
    # favor of a properly constructed one: the transport follows
    # `runtime.session` afterward, not the placeholder. See serve_twilio's
    # docstring for why that is the convention.
    return CascadeRuntime(CallSession(agent))


asyncio.run(serve_twilio(factory, host="0.0.0.0", port=8080, path="/twilio"))
```

Twilio requires `wss://` (TLS) in production; put a reverse proxy or load
balancer that terminates TLS in front of this server rather than
implementing TLS in the transport itself.

### 3. Local development: ngrok

Twilio cannot reach `localhost`, so local development needs a public URL
that tunnels back to your machine:

```bash
ngrok http 8080
```

ngrok prints a `https://<random>.ngrok-free.app` URL; use the `wss://`
equivalent of that host in the `<Stream url="...">` above (same host,
`wss://` instead of `https://`, same path). Point the phone number's
webhook at an HTTP endpoint that returns the TwiML from step 1 with that
URL baked in, either a second local server also tunneled through ngrok, or
a static TwiML Bin configured directly in the Twilio console for a fixed
URL that never changes between restarts.

### The mark trick: exact interruption reconciliation

Every runtime in this stack tracks "what has the caller actually heard"
through `primitives/interruption.py`'s `PlaybackLedger`, because that
question is what makes a barge-in correct: if the model's next turn
believes the caller heard the whole of its last sentence when they only
heard the first half, it will confidently reference words the caller never
heard. Left to itself, `CascadeRuntime` can only guess at the answer: it
assumes a frame handed to `on_bot_audio` is a frame already at the
caller's ear, and converts elapsed audio duration to a character position
at a fixed, documented speech rate (`SPOKEN_CHARS_PER_SECOND` in
`runtimes/cascade.py`). That assumption is false on a phone call: there is
a real jitter buffer between "the transport sent this media frame" and
"the caller's ear moved", and the error is worst exactly when it is
checked most, at the instant of a barge-in.

Twilio's `mark` message removes the guess. A `mark` sent right after a
batch of `media` frames is acknowledged by Twilio, mark for mark, only
once the audio in front of it has genuinely finished playing (see
[the mark message docs](https://www.twilio.com/docs/voice/media-streams/websocket-messages#mark-message)).
That is a real playout clock from the carrier, not an estimate from the
runtime, and `serve_twilio` uses it: every outbound audio chunk gets a
mark, every mark ack converts "bytes Twilio has confirmed played" into
"characters of the current utterance the caller has confirmed heard," and
that number is handed straight to `runtime.ledger.mark_played`, the exact
integration point `docs/EXTENDING.md` documents for any transport with a
real playout clock. This is the single biggest reason to prefer this
transport's interruption behavior over a browser or LAN WebSocket call:
telephony is the one path in this stack where "what did the caller hear"
is a fact the wire protocol will actually tell you, instead of a number
the runtime has to estimate.

One honest limitation, documented in `transports/twilio.py`'s code
comments rather than hidden: `PlaybackLedger` does not yet expose *which*
utterance it currently considers "in flight" through public API, only
through a private field. The transport reaches into that field
defensively (every lookup falls back to `None` rather than raising), so a
future change to `interruption.py`'s internal shape disables this upgrade
gracefully instead of breaking a live call. The clean fix, a public
`PlaybackLedger.current_utterance_id` property, is a roadmap item, not
something this document pretends is already done.

On a genuine barge-in (an `Interruption` event on the session), the
transport also sends Twilio a `clear` message, which empties whatever
audio Twilio has already buffered for playback. Without it, the caller
would keep hearing the tail of a sentence the runtime has already decided
to abandon, even though `push_audio`/`on_bot_audio` have both moved on.

---

## Generic SIP

Twilio Media Streams is one specific WebSocket protocol over G.711 mu-law.
A SIP trunk (a carrier, a PBX, an on-prem phone system) speaks SIP and RTP,
neither of which Tring parses today. The standard way to bridge the two
is FreeSWITCH:

```
caller --SIP/RTP--> FreeSWITCH --mod_audio_stream (WebSocket)--> tring.transports.websocket
```

[mod_audio_stream](https://github.com/amigniter/mod_audio_stream) is a
FreeSWITCH module that bridges one leg of a SIP call to a plain WebSocket,
streaming call audio out as binary frames and (from v1.1.0 onward)
accepting binary frames back to play into the call. It streams linear PCM
(`L16`), and its *default* is FreeSWITCH's own internal 8 kHz channel
rate, not 16 kHz: the sample rate is a parameter of the module's own
`uuid_audio_stream <uuid> start <wss-url> <mix-type> <sampling-rate>
<metadata>` API command, so the operator configuring the dialplan needs to
request `16000` explicitly. Do that, and the module's wire format becomes
*exactly* Tring's canonical `AudioFrame` format: 16 kHz mono 16-bit linear
PCM, no mu-law, no resampling step needed at all. That means the existing
[`tring.transports.websocket`](../src/tring/transports/websocket.py)
transport, built for browser and LAN audio, is also the correct transport
for a FreeSWITCH bridge configured this way: point `mod_audio_stream` at
`serve()`'s WebSocket endpoint, request 16 kHz in the `uuid_audio_stream`
call, and the two speak the same format natively.

### What this repo does not yet ship

Be precise about what "FreeSWITCH bridge pattern" means here: it is a
documented integration pattern using an existing FreeSWITCH module and
this repo's existing WebSocket transport, not a Tring-authored FreeSWITCH
integration. Specifically, Tring does **not** yet ship:

- A FreeSWITCH dialplan or `mod_audio_stream` configuration generator.
  Wiring a SIP trunk, a dialplan extension, and `mod_audio_stream`'s
  `uuid_audio_stream` command together for a given deployment is left to
  the operator, following `mod_audio_stream`'s own documentation.
- A codec path for a bridge that hands Tring 8 kHz G.711 (a-law or
  mu-law) instead of 16 kHz linear PCM. `transports/mulaw.py`'s codec and
  resample helpers are generic (nothing in them is Twilio-specific), so an
  8 kHz SIP bridge is a small adapter away, structurally the same shape as
  `transports/twilio.py`, but that adapter does not exist yet.
- Mark-equivalent exact playout tracking for SIP. RTP has no `mark`
  message; a SIP bridge's interruption reconciliation stays at
  `CascadeRuntime`'s own estimate (`SPOKEN_CHARS_PER_SECOND`) unless and
  until a specific bridge exposes a real playout signal `serve()` can be
  taught to consume the same way `serve_twilio` consumes Twilio's marks.

These are tracked as follow-up work, not silently assumed to work.
