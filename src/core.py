"""chroma-ai core DCT watermark engine."""

import numpy as np
from scipy.fft import dctn, idctn
from dataclasses import dataclass
from typing import Optional

ZIGZAG_8x8 = [
    (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
    (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
    (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
    (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
    (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
    (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
    (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
    (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7)
]

BAND_A_POSITIONS = [ZIGZAG_8x8[i] for i in range(14, 22)]
BAND_B_POSITION = (0, 2)

# ── Keyed security layer (used only when key is not None) ────────────────────
# Candidate Band B positions rotated per-block via keystream.
BAND_B_POSITIONS_KEYED = [(0, 2), (0, 3), (1, 2), (2, 1)]
# Number of Band A coefficients used per block in the keyed path (down from 8,
# offsetting the entropy cost of randomising positions via Fisher-Yates).
BAND_A_KEYED_TAKE = 4
# Keystream byte threshold for block gating: embed iff ks[0] < 192 → P=0.75.
_KEYED_GATE_THRESHOLD = 192


def _fisher_yates(items, rnd_bytes):
    """Keystream-seeded Fisher-Yates shuffle; returns a new list."""
    out = list(items)
    n = len(out)
    for i in range(n - 1, 0, -1):
        j = rnd_bytes[i] % (i + 1)
        out[i], out[j] = out[j], out[i]
    return out


@dataclass
class EmbedConfig:
    """Embedding configuration."""
    band_b_alpha: float = 37.0  # Task 2: absolute baseline (adaptive lookup overrides)
    band_a_alpha: float = 15.0
    adaptive: bool = True
    texture_gate: float = 50.0
    payload_size: int = 256
    seed: int = 42


@dataclass
class EmbedResult:
    """Result of embedding operation."""
    frame: np.ndarray
    psnr: float
    blocks_embedded: int
    blocks_skipped: int
    quality_accepted: bool
    mean_texture: float


@dataclass
class VerifyResult:
    """Result of verification."""
    detected: bool
    band_b_brr: float
    band_a_brr: Optional[float]
    confidence: str  # "high", "medium", "low", "none"
    payload_match: bool


def compute_texture_energy(block: np.ndarray) -> float:
    """AC energy of an 8x8 block."""
    d = dctn(block, type=2, norm='ortho')
    return float(np.sum(d ** 2) - d[0, 0] ** 2)


def frame_texture_stats(frame: np.ndarray) -> tuple[float, float]:
    """Compute mean and median texture energy for a frame."""
    h, w = frame.shape
    energies = []
    for by in range(0, h - 7, 8):
        for bx in range(0, w - 7, 8):
            energies.append(compute_texture_energy(frame[by:by+8, bx:bx+8]))
    return float(np.mean(energies)), float(np.median(energies))


def generate_payload(size: int, seed: int = 42) -> np.ndarray:
    """Generate deterministic binary payload."""
    return np.random.RandomState(seed).randint(0, 2, size=size).astype(np.uint8)


def compute_psnr(original: np.ndarray, modified: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in dB."""
    mse = np.mean((original.astype(np.float64) - modified.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return float(10 * np.log10(255 ** 2 / mse))


def embed_band_b(frame: np.ndarray, payload: np.ndarray,
                 config: EmbedConfig,
                 key: Optional[bytes] = None,
                 frame_idx: int = 0) -> np.ndarray:
    """
    Content-adaptive Band B embedding at DCT position (0,2).
    Flat blocks receive higher alpha to compensate for low coefficient magnitude.

    When ``key`` is None: original behavior (fixed position, no scrambling).
    When ``key`` is provided: per-block position rotation, sign scrambling,
    and block gating driven by a ChaCha20 keystream (see src/keyed_crypto.py).
    """
    h, w = frame.shape
    emb = frame.copy()
    bi = 0

    if key is not None:
        from .keyed_crypto import block_keystream, zero_buffer

    for by in range(0, h - 7, 8):
        for bx in range(0, w - 7, 8):
            if bi >= len(payload):
                return emb

            if key is not None:
                ks = block_keystream(key, frame_idx, by, bx, 16)
                if ks[0] >= _KEYED_GATE_THRESHOLD:
                    zero_buffer(ks)
                    continue
                pos = BAND_B_POSITIONS_KEYED[ks[1] % 4]
                xor_bit = ks[2] & 1
                zero_buffer(ks)
            else:
                pos = BAND_B_POSITION
                xor_bit = 0

            blk = emb[by:by+8, bx:bx+8].copy()

            if config.adaptive:
                # Task 2 refined lookup — absolute alpha, no multiplier.
                # Minimum alpha for ≥95% survival through H.265 CRF28 (+2 safety margin).
                tex = compute_texture_energy(blk)
                if tex < 100:
                    local_alpha = 37    # flat (tex<10) + low (10≤tex<100)
                else:
                    local_alpha = 32    # medium (100≤tex<1000) + high (tex≥1000)
            else:
                local_alpha = config.band_b_alpha

            d = dctn(blk, type=2, norm='ortho')
            r, c = pos
            bit = int(payload[bi]) ^ xor_bit
            d[r, c] = (abs(d[r, c]) + local_alpha) if bit == 1 \
                else -(abs(d[r, c]) + local_alpha)
            emb[by:by+8, bx:bx+8] = idctn(d, type=2, norm='ortho')
            bi += 1

    return emb


def embed_band_a(frame: np.ndarray, payload: np.ndarray,
                 config: EmbedConfig,
                 key: Optional[bytes] = None,
                 frame_idx: int = 0) -> tuple[np.ndarray, int, int]:
    """
    Texture-gated Band A embedding at zigzag positions 14-21.
    Returns (embedded_frame, blocks_used, blocks_skipped).

    When ``key`` is provided: positions are a keystream-seeded Fisher-Yates
    shuffle of zigzag[14:22] truncated to BAND_A_KEYED_TAKE=4 coefficients,
    sign is XOR-scrambled, and blocks are additionally gated at P=0.75.
    Texture gating is preserved on both paths.
    """
    h, w = frame.shape
    emb = frame.copy()
    bi = 0
    used = 0
    skipped = 0

    if key is not None:
        from .keyed_crypto import block_keystream, zero_buffer

    for by in range(0, h - 7, 8):
        for bx in range(0, w - 7, 8):
            if bi >= len(payload):
                return emb, used, skipped

            if key is not None:
                ks = block_keystream(key, frame_idx, by, bx, 16)
                if ks[0] >= _KEYED_GATE_THRESHOLD:
                    zero_buffer(ks)
                    skipped += 1
                    continue
                xor_bit = ks[2] & 1
                shuffled = _fisher_yates(BAND_A_POSITIONS, bytes(ks[3:11]))
                positions = shuffled[:BAND_A_KEYED_TAKE]
                zero_buffer(ks)
            else:
                xor_bit = 0
                positions = BAND_A_POSITIONS

            blk = emb[by:by+8, bx:bx+8].copy()
            tex = compute_texture_energy(blk)

            if tex < config.texture_gate:
                skipped += 1
                bi += 1
                continue

            local_alpha = config.band_a_alpha * max(0.5, min(2.0, 200.0 / (tex + 1)))
            d = dctn(blk, type=2, norm='ortho')
            bit = int(payload[bi]) ^ xor_bit

            for r, c in positions:
                d[r, c] = (abs(d[r, c]) + local_alpha) if bit == 1 \
                    else -(abs(d[r, c]) + local_alpha)

            emb[by:by+8, bx:bx+8] = idctn(d, type=2, norm='ortho')
            bi += 1
            used += 1

    return emb, used, skipped


def extract_band_b(frame: np.ndarray, n: int,
                   key: Optional[bytes] = None,
                   frame_idx: int = 0) -> np.ndarray:
    """Extract n bits from Band B. Keyed extract mirrors the keyed embed:
    same keystream-driven block gating, position rotation, and sign unscramble.
    """
    h, w = frame.shape
    bits = []

    if key is not None:
        from .keyed_crypto import block_keystream, zero_buffer

    for by in range(0, h - 7, 8):
        for bx in range(0, w - 7, 8):
            if len(bits) >= n:
                return np.array(bits, dtype=np.uint8)

            if key is not None:
                ks = block_keystream(key, frame_idx, by, bx, 16)
                if ks[0] >= _KEYED_GATE_THRESHOLD:
                    zero_buffer(ks)
                    continue
                pos = BAND_B_POSITIONS_KEYED[ks[1] % 4]
                xor_bit = ks[2] & 1
                zero_buffer(ks)
            else:
                pos = BAND_B_POSITION
                xor_bit = 0

            d = dctn(frame[by:by+8, bx:bx+8], type=2, norm='ortho')
            sign_bit = 1 if d[pos] > 0 else 0
            bits.append(sign_bit ^ xor_bit)
    return np.array(bits, dtype=np.uint8)


def extract_band_a(frame: np.ndarray, n: int,
                   key: Optional[bytes] = None,
                   frame_idx: int = 0) -> np.ndarray:
    """Extract n bits from Band A with majority voting.

    Unkeyed: votes across the fixed zigzag[14:22] positions.
    Keyed: re-derives the per-block shuffle + gating from the keystream and
    votes across the shuffled first BAND_A_KEYED_TAKE positions, unscrambling
    the sign.
    """
    h, w = frame.shape
    bits = []

    if key is not None:
        from .keyed_crypto import block_keystream, zero_buffer

    for by in range(0, h - 7, 8):
        for bx in range(0, w - 7, 8):
            if len(bits) >= n:
                return np.array(bits, dtype=np.uint8)

            if key is not None:
                ks = block_keystream(key, frame_idx, by, bx, 16)
                if ks[0] >= _KEYED_GATE_THRESHOLD:
                    zero_buffer(ks)
                    continue
                xor_bit = ks[2] & 1
                shuffled = _fisher_yates(BAND_A_POSITIONS, bytes(ks[3:11]))
                positions = shuffled[:BAND_A_KEYED_TAKE]
                zero_buffer(ks)
            else:
                xor_bit = 0
                positions = BAND_A_POSITIONS

            d = dctn(frame[by:by+8, bx:bx+8], type=2, norm='ortho')
            votes = sum(1 if d[r, c] > 0 else -1 for r, c in positions)
            sign_bit = 1 if votes > 0 else 0
            bits.append(sign_bit ^ xor_bit)
    return np.array(bits, dtype=np.uint8)


def compute_brr(original: np.ndarray, extracted: np.ndarray) -> float:
    """Bit Recovery Rate (%)."""
    n = min(len(original), len(extracted))
    if n == 0:
        return 0.0
    return float(np.mean(original[:n] == extracted[:n]) * 100)
