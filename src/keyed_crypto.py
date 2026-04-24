"""chroma-ai keyed crypto: HKDF-SHA256 subkeys + ChaCha20 keystream per block."""

import struct
from functools import lru_cache

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

GOP_SIZE = 30
_HKDF_SALT = b"chroma-ai-v1"
_SUBKEY_LEN = 32


def derive_gop_subkey(master_key: bytes, gop_idx: int) -> bytes:
    """Derive a 32-byte ChaCha20 subkey for one GOP via HKDF-SHA256."""
    if not isinstance(master_key, (bytes, bytearray)) or len(master_key) < 16:
        raise ValueError("master_key must be >=16 bytes")
    return HKDF(
        algorithm=hashes.SHA256(),
        length=_SUBKEY_LEN,
        salt=_HKDF_SALT,
        info=f"GOP:{gop_idx}".encode("ascii"),
    ).derive(bytes(master_key))


def derive_block_nonce(frame_idx_in_gop: int, by: int, bx: int) -> bytes:
    """Pack (frame_in_gop, block_y, block_x) into a 12-byte ChaCha20 nonce."""
    return struct.pack(">III", frame_idx_in_gop & 0xFFFFFFFF,
                       by & 0xFFFFFFFF, bx & 0xFFFFFFFF)


def get_keystream(subkey: bytes, nonce: bytes, n_bytes: int) -> bytearray:
    """ChaCha20 keystream of length n_bytes as a mutable bytearray.

    Caller should zero the returned buffer after use (see core.py).
    """
    if len(subkey) != _SUBKEY_LEN:
        raise ValueError(f"subkey must be {_SUBKEY_LEN} bytes")
    if len(nonce) != 12:
        raise ValueError("nonce must be 12 bytes")
    # cryptography's raw ChaCha20 takes a 16-byte (counter || nonce) input;
    # prepend a 4-byte zero counter (IETF convention) so our public nonce
    # stays the 12-byte (frame_in_gop, by, bx) struct.
    full_nonce = b"\x00\x00\x00\x00" + bytes(nonce)
    enc = Cipher(algorithms.ChaCha20(bytes(subkey), full_nonce), mode=None).encryptor()
    return bytearray(enc.update(b"\x00" * n_bytes))


@lru_cache(maxsize=2)
def _cached_subkey(master_key: bytes, gop_idx: int) -> bytes:
    return derive_gop_subkey(master_key, gop_idx)


def block_keystream(master_key: bytes, frame_idx: int, by: int, bx: int,
                    n_bytes: int = 16) -> bytearray:
    """Convenience: (master_key, frame_idx, by, bx) → keystream bytearray.

    Derives per-GOP subkey (cached) and ChaCha20 keystream for this block's
    nonce. The returned buffer is mutable so the caller can zero it after use.
    """
    gop_idx = frame_idx // GOP_SIZE
    fig = frame_idx % GOP_SIZE
    subkey = _cached_subkey(bytes(master_key), gop_idx)
    return get_keystream(subkey, derive_block_nonce(fig, by, bx), n_bytes)


def zero_buffer(buf: bytearray) -> None:
    """Overwrite a mutable keystream buffer with zeros before it is dropped."""
    for i in range(len(buf)):
        buf[i] = 0
