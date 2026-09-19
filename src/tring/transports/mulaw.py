"""G.711 mu-law codec and linear resampling, pure stdlib.

Telephony transports (Twilio Media Streams, most SIP/RTP bridges) speak
G.711 mu-law at 8 kHz, not the stack's canonical 16 kHz linear PCM (see
``runtimes/base.py``'s ``AudioFrame`` docstring). Historically Python code
reached for the stdlib ``audioop`` module to do both the codec and the
resampling; ``audioop`` is deprecated as of 3.12 and removed outright in
3.13 (https://docs.python.org/3/whatsnew/3.13.html -- PEP 594), so a
telephony transport that still imports it stops working on the next Python
release. This module is the replacement: a small, dependency-free
reimplementation of exactly the two primitives telephony needs.

**mu-law codec.** ITU-T Recommendation G.711 (1988), section on the
"micro-law" compressor, defines the 8-bit logarithmic encoding as a
piecewise-linear approximation of the ITU compression curve: 8 "chords"
(segments) of doubling width, each split into 16 uniform steps, encoded as
sign (1 bit) + chord (3 bits) + step (4 bits), then bitwise-complemented.
:func:`linear_to_ulaw_sample` / :func:`ulaw_to_linear_sample` implement the
segment search with ``int.bit_length()`` instead of a lookup table --
"table-free" per this module's design brief -- which is possible because
the eight chord boundaries are exactly the powers of two the encoder is
already looking for. Both were checked bit-for-bit against CPython's
``audioop.lin2ulaw``/``audioop.ulaw2lin`` (still present through 3.12; this
was a *development-time* cross-check only -- nothing in this module imports
``audioop``) across the full 16-bit domain, all 65536 values in both
directions, exact match. The reference algorithm and constants (``BIAS``,
``CLIP``, the pre-encode ``>>2``) follow the widely-deployed public-domain
G.711 reference implementation (Sun Microsystems' ``g711.c``, the same
lineage CPython's own ``audioop`` and most open-source codecs implement).

**Resampling.** :func:`resample_linear16` is deliberately the same
linear-interpolation algorithm as
``providers/cloud/tts_extra.py``'s ``_resample_pcm16`` -- not imported from
there (that module resamples a *TTS vendor's* fixed rate down to the wire
format; this one resamples the *wire format* to and from 8 kHz telephony,
a different caller with a different reason to exist, and the two are meant
to be editable independently, matching that module's own comment about
why it doesn't share code with ``providers/local``'s copy of the same
idea). It is the same deliberately cheap choice for the same reason: no
numpy, no new dependency, a faint aliasing artifact above the destination
Nyquist frequency that is inaudible on a telephony-bandlimited path. A
future ``tring[hifi]`` extra should use a real polyphase resampler for
every one of these call sites at once.
"""

from __future__ import annotations

import array

#: ITU-T G.711 mu-law constants (see module docstring for the reference
#: implementation this follows). BIAS is added to the linear magnitude
#: before segment search so that even silence (0) lands in a valid segment.
_BIAS = 0x84  # 132
#: Clip ceiling *after* the pre-encode `>>2`, i.e. against a 13-bit
#: magnitude -- this is what keeps the maximum code word in range.
_CLIP = 8159
_QUANT_BIAS = _BIAS >> 2  # 33, i.e. BIAS scaled into the same >>2 domain


def linear_to_ulaw_sample(sample: int) -> int:
    """Encode one 16-bit signed linear PCM sample to an 8-bit mu-law byte.

    ``sample`` is truncated to the 13-bit magnitude domain by an initial
    ``>>2`` *before* the sign is extracted -- an arithmetic (floor) shift,
    which is what Python's ``>>`` already does on negative integers, exactly
    matching the two's-complement right-shift the reference C implementation
    relies on. This is not an approximation added here: discarding the low
    2 bits before quantizing is part of the G.711 reference algorithm
    itself, and skipping it (e.g. by taking the absolute value first) would
    silently shift every code-word boundary and stop matching what a real
    G.711 decoder on the other end of the call expects.
    """
    sample >>= 2
    if sample < 0:
        sample = -sample
        sign_mask = 0x7F
    else:
        sign_mask = 0xFF
    sample = min(sample, _CLIP)
    sample += _QUANT_BIAS

    # Segment (chord) search, table-free: chord boundaries sit exactly at
    # sample == 2**(6+chord) - 1, so the chord is `bit_length(sample) - 6`
    # clamped to [0, 8). `sample <= _CLIP + _QUANT_BIAS == 8192 == 2**13`
    # always, so `chord` can reach 8 (out of the valid 0..7 range) only at
    # that single saturating value -- handled explicitly below rather than
    # folded into the clamp, so it stays visibly a distinct "maximum
    # magnitude" case rather than silently aliasing onto chord 7.
    chord = max(0, sample.bit_length() - 6)
    if chord >= 8:
        return (0x7F ^ sign_mask) & 0xFF
    step = (sample >> (chord + 1)) & 0x0F
    return (((chord << 4) | step) ^ sign_mask) & 0xFF


def ulaw_to_linear_sample(byte: int) -> int:
    """Decode one 8-bit mu-law byte to a 16-bit signed linear PCM sample.

    Inverts :func:`linear_to_ulaw_sample`'s bit layout directly (complement,
    then pull sign/chord/step back apart) rather than a table -- the classic
    "table-free" G.711 decode, identical in shape to the reference
    ``ulaw2linear`` this module was checked against.
    """
    byte = (~byte) & 0xFF
    sign = byte & 0x80
    chord = (byte >> 4) & 0x07
    step = byte & 0x0F
    magnitude = (((step << 3) + _BIAS) << chord) - _BIAS
    return -magnitude if sign else magnitude


def linear16_to_mulaw(pcm: bytes) -> bytes:
    """Encode a buffer of little-endian 16-bit signed PCM to mu-law bytes.

    A trailing odd byte (a truncated final sample) is dropped rather than
    raising: audio frames get sliced by transports on arbitrary boundaries,
    and a transport should never crash a live call over half a sample at a
    chunk edge.
    """
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    return bytes(linear_to_ulaw_sample(s) for s in samples)


def mulaw_to_linear16(data: bytes) -> bytes:
    """Decode a buffer of mu-law bytes to little-endian 16-bit signed PCM."""
    out = array.array("h", bytes(2 * len(data)))
    for i, byte in enumerate(data):
        out[i] = ulaw_to_linear_sample(byte)
    return out.tobytes()


def resample_linear16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolation resample of little-endian 16-bit PCM.

    See the module docstring for why this duplicates (rather than imports)
    ``providers/cloud/tts_extra.py``'s ``_resample_pcm16``: same algorithm,
    same "no numpy" constraint, different call site that should stay free to
    change independently. Works in either direction -- upsampling 8 kHz
    telephony audio to the stack's 16 kHz wire format, or downsampling bot
    audio back down to 8 kHz for the call -- since linear interpolation
    needs no special-casing for the direction of rate change.
    """
    if src_rate == dst_rate or not pcm:
        return pcm
    src = array.array("h")
    src.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    src_n = len(src)
    if src_n < 2:
        return b""
    dst_n = max(int(src_n * dst_rate / src_rate), 0)
    if dst_n <= 0:
        return b""
    last_index = src_n - 1
    step = last_index / max(dst_n - 1, 1)
    out = array.array("h", bytes(2 * dst_n))
    for i in range(dst_n):
        pos = i * step
        lo = int(pos)
        hi = min(lo + 1, last_index)
        frac = pos - lo
        value = src[lo] * (1.0 - frac) + src[hi] * frac
        out[i] = max(-32768, min(32767, int(value)))
    return out.tobytes()


__all__ = [
    "linear16_to_mulaw",
    "linear_to_ulaw_sample",
    "mulaw_to_linear16",
    "resample_linear16",
    "ulaw_to_linear_sample",
]
