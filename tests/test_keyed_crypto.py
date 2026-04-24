"""chroma-ai keyed crypto tests: determinism, nonce uniqueness, key isolation."""

import os
import numpy as np
import pytest

from src.keyed_crypto import (
    GOP_SIZE,
    derive_gop_subkey,
    derive_block_nonce,
    get_keystream,
    block_keystream,
    zero_buffer,
    _cached_subkey,
)


K1 = b"\x01" * 32
K2 = b"\x02" * 32


def test_determinism():
    """Same (key, frame, by, bx) yields identical keystream across repeats."""
    ref = bytes(block_keystream(K1, 0, 0, 0, 16))
    for _ in range(10):
        assert bytes(block_keystream(K1, 0, 0, 0, 16)) == ref
    assert bytes(block_keystream(K1, 17, 3, 5, 32)) == \
        bytes(block_keystream(K1, 17, 3, 5, 32))


def test_nonce_and_keystream_uniqueness():
    """10k distinct (frame, by, bx) triples give 10k distinct nonces and keystreams."""
    nonces = set()
    streams = set()
    for frame in range(4):
        for by in range(0, 480, 8):          # 60 rows
            for bx in range(0, 424, 8):      # 53 cols  => 60*53=3180/frame, *4 frames=12720
                fig = frame % GOP_SIZE
                nonces.add(derive_block_nonce(fig, by, bx))
                streams.add(bytes(block_keystream(K1, frame, by, bx, 16)))
    assert len(nonces) >= 10000, f"only {len(nonces)} unique nonces"
    assert len(streams) >= 10000, f"only {len(streams)} unique keystreams"


def test_key_isolation():
    """Different master keys yield keystreams that differ in >=40% of bits (statistical)."""
    rng = np.random.RandomState(1)
    diffs = []
    for _ in range(16):
        ka = bytes(rng.randint(0, 256, 32).astype(np.uint8).tobytes())
        kb = bytes(rng.randint(0, 256, 32).astype(np.uint8).tobytes())
        _cached_subkey.cache_clear()
        sa = np.frombuffer(bytes(block_keystream(ka, 0, 0, 0, 128)), dtype=np.uint8)
        sb = np.frombuffer(bytes(block_keystream(kb, 0, 0, 0, 128)), dtype=np.uint8)
        bit_diff = np.unpackbits(sa ^ sb).mean()
        diffs.append(bit_diff)
    assert np.mean(diffs) >= 0.40, f"mean bit-diff {np.mean(diffs):.3f} below 0.40"


def test_no_state_leak_across_interleaved_calls():
    """Interleaved calls across different keys match sequential calls."""
    _cached_subkey.cache_clear()
    a1 = bytes(block_keystream(K1, 0, 0, 0, 16))
    b1 = bytes(block_keystream(K2, 0, 0, 0, 16))
    a2 = bytes(block_keystream(K1, 0, 0, 0, 16))
    b2 = bytes(block_keystream(K2, 0, 0, 0, 16))
    assert a1 == a2
    assert b1 == b2
    assert a1 != b1

    _cached_subkey.cache_clear()
    fresh_a = bytes(block_keystream(K1, 0, 0, 0, 16))
    _cached_subkey.cache_clear()
    fresh_b = bytes(block_keystream(K2, 0, 0, 0, 16))
    assert fresh_a == a1
    assert fresh_b == b1


def test_zero_buffer_wipes():
    buf = bytearray(b"\xaa" * 16)
    zero_buffer(buf)
    assert buf == bytearray(16)


def test_reject_bad_lengths():
    with pytest.raises(ValueError):
        derive_gop_subkey(b"short", 0)
    sk = derive_gop_subkey(K1, 0)
    with pytest.raises(ValueError):
        get_keystream(sk, b"\x00" * 8, 16)
    with pytest.raises(ValueError):
        get_keystream(b"\x00" * 16, derive_block_nonce(0, 0, 0), 16)
