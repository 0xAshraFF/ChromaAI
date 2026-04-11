"""ChromaCascade core DCT watermark engine."""

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
                 config: EmbedConfig) -> np.ndarray:
    """
    Content-adaptive Band B embedding at DCT position (0,2).
    Flat blocks receive higher alpha to compensate for low coefficient magnitude.
    """
    h, w = frame.shape
    emb = frame.copy()
    bi = 0

    for by in range(0, h - 7, 8):
        for bx in range(0, w - 7, 8):
            if bi >= len(payload):
                return emb

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
            r, c = BAND_B_POSITION
            d[r, c] = (abs(d[r, c]) + local_alpha) if payload[bi] == 1 \
                else -(abs(d[r, c]) + local_alpha)
            emb[by:by+8, bx:bx+8] = idctn(d, type=2, norm='ortho')
            bi += 1

    return emb


def embed_band_a(frame: np.ndarray, payload: np.ndarray,
                 config: EmbedConfig) -> tuple[np.ndarray, int, int]:
    """
    Texture-gated Band A embedding at zigzag positions 14-21.
    Returns (embedded_frame, blocks_used, blocks_skipped).
    """
    h, w = frame.shape
    emb = frame.copy()
    bi = 0
    used = 0
    skipped = 0

    for by in range(0, h - 7, 8):
        for bx in range(0, w - 7, 8):
            if bi >= len(payload):
                return emb, used, skipped

            blk = emb[by:by+8, bx:bx+8].copy()
            tex = compute_texture_energy(blk)

            if tex < config.texture_gate:
                skipped += 1
                bi += 1
                continue

            local_alpha = config.band_a_alpha * max(0.5, min(2.0, 200.0 / (tex + 1)))
            d = dctn(blk, type=2, norm='ortho')
            bit = payload[bi]

            for r, c in BAND_A_POSITIONS:
                d[r, c] = (abs(d[r, c]) + local_alpha) if bit == 1 \
                    else -(abs(d[r, c]) + local_alpha)

            emb[by:by+8, bx:bx+8] = idctn(d, type=2, norm='ortho')
            bi += 1
            used += 1

    return emb, used, skipped


def extract_band_b(frame: np.ndarray, n: int) -> np.ndarray:
    """Extract n bits from Band B position (0,2)."""
    h, w = frame.shape
    bits = []
    for by in range(0, h - 7, 8):
        for bx in range(0, w - 7, 8):
            if len(bits) >= n:
                return np.array(bits, dtype=np.uint8)
            d = dctn(frame[by:by+8, bx:bx+8], type=2, norm='ortho')
            bits.append(1 if d[BAND_B_POSITION] > 0 else 0)
    return np.array(bits, dtype=np.uint8)


def extract_band_a(frame: np.ndarray, n: int) -> np.ndarray:
    """Extract n bits from Band A with majority voting across ZZ 14-21."""
    h, w = frame.shape
    bits = []
    for by in range(0, h - 7, 8):
        for bx in range(0, w - 7, 8):
            if len(bits) >= n:
                return np.array(bits, dtype=np.uint8)
            d = dctn(frame[by:by+8, bx:bx+8], type=2, norm='ortho')
            votes = sum(1 if d[r, c] > 0 else -1 for r, c in BAND_A_POSITIONS)
            bits.append(1 if votes > 0 else 0)
    return np.array(bits, dtype=np.uint8)


def compute_brr(original: np.ndarray, extracted: np.ndarray) -> float:
    """Bit Recovery Rate (%)."""
    n = min(len(original), len(extracted))
    if n == 0:
        return 0.0
    return float(np.mean(original[:n] == extracted[:n]) * 100)
