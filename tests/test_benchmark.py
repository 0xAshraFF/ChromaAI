"""chroma-ai test suite: Validates core claims with synthetic content."""

import numpy as np
import cv2
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core import (
    EmbedConfig, embed_band_b, extract_band_b, embed_band_a,
    extract_band_a, compute_brr, generate_payload, compute_psnr,
    frame_texture_stats
)
from src.embed import embed_frame
from src.verify import verify_frame
from src.texture_gating import assess_frame_quality


def make_noise_frame(w=854, h=480, seed=0):
    """High-texture synthetic frame."""
    return np.random.RandomState(seed).randint(0, 256, (h, w)).astype(np.float64)


def make_gradient_frame(w=854, h=480):
    """Low-texture gradient."""
    return np.tile(np.linspace(50, 200, w), (h, 1)).astype(np.float64)


def make_mixed_frame(w=854, h=480, seed=0):
    """Half noise, half flat."""
    frame = np.full((h, w), 128.0)
    frame[:, :w//2] = np.random.RandomState(seed).randint(0, 256, (h, w//2))
    return frame


CONFIG = EmbedConfig(band_b_alpha=12, adaptive=True)
PAYLOAD = generate_payload(256, seed=42)


class TestBandBEmbedExtract:
    """Band B core embed/extract without attacks."""

    def test_perfect_recovery_noise(self):
        frame = make_noise_frame()
        wm = embed_band_b(frame, PAYLOAD, CONFIG)
        ext = extract_band_b(wm, 256)
        assert compute_brr(PAYLOAD, ext) == 100.0

    def test_perfect_recovery_gradient(self):
        frame = make_gradient_frame()
        wm = embed_band_b(frame, PAYLOAD, CONFIG)
        ext = extract_band_b(wm, 256)
        assert compute_brr(PAYLOAD, ext) == 100.0

    def test_psnr_above_28(self):
        """Noise frames have lowest PSNR due to competing AC energy.
        Real content averages 47.4 dB. Noise worst case: ~30 dB."""
        frame = make_noise_frame()
        result = embed_frame(frame, CONFIG)
        assert result.psnr > 28.0

    def test_psnr_gradient_above_40(self):
        """Low-texture content has high PSNR (less to compete with)."""
        frame = make_gradient_frame()
        result = embed_frame(frame, CONFIG)
        assert result.psnr > 40.0


class TestBandBAttacks:
    """Band B survival under real-world attacks."""

    def test_jpeg_q20(self):
        frame = make_noise_frame()
        wm = embed_band_b(frame, PAYLOAD, CONFIG)
        u8 = np.clip(wm, 0, 255).astype(np.uint8)
        _, buf = cv2.imencode('.jpg', u8, [cv2.IMWRITE_JPEG_QUALITY, 20])
        attacked = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float64)
        ext = extract_band_b(attacked, 256)
        assert compute_brr(PAYLOAD, ext) > 80.0

    def test_jpeg_q50(self):
        frame = make_noise_frame()
        wm = embed_band_b(frame, PAYLOAD, CONFIG)
        u8 = np.clip(wm, 0, 255).astype(np.uint8)
        _, buf = cv2.imencode('.jpg', u8, [cv2.IMWRITE_JPEG_QUALITY, 50])
        attacked = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float64)
        ext = extract_band_b(attacked, 256)
        assert compute_brr(PAYLOAD, ext) > 90.0

    def test_resize_90(self):
        frame = make_noise_frame()
        wm = embed_band_b(frame, PAYLOAD, CONFIG)
        u8 = np.clip(wm, 0, 255).astype(np.uint8)
        h, w = u8.shape
        small = cv2.resize(u8, (int(w * 0.9), int(h * 0.9)))
        back = cv2.resize(small, (w, h)).astype(np.float64)
        ext = extract_band_b(back, 256)
        assert compute_brr(PAYLOAD, ext) > 85.0

    def test_blur_3px(self):
        frame = make_noise_frame()
        wm = embed_band_b(frame, PAYLOAD, CONFIG)
        u8 = np.clip(wm, 0, 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(u8, (7, 7), 3.0).astype(np.float64)
        ext = extract_band_b(blurred, 256)
        assert compute_brr(PAYLOAD, ext) > 60.0  # Known weaker against blur

    def test_sharpen(self):
        frame = make_noise_frame()
        wm = embed_band_b(frame, PAYLOAD, CONFIG)
        u8 = np.clip(wm, 0, 255).astype(np.uint8)
        kernel = np.array([[-0.2, -0.2, -0.2], [-0.2, 2.6, -0.2], [-0.2, -0.2, -0.2]])
        sharpened = cv2.filter2D(u8, -1, kernel).astype(np.float64)
        ext = extract_band_b(sharpened, 256)
        assert compute_brr(PAYLOAD, ext) > 75.0


class TestFalsePositiveRate:
    """Verify 0% FPR on clean content."""

    def test_fpr_noise(self):
        frame = make_noise_frame(seed=999)  # Different seed from payload
        ext = extract_band_b(frame, 256)
        b = compute_brr(PAYLOAD, ext)
        assert b < 70.0  # Should NOT detect (random ~50%)

    def test_fpr_gradient(self):
        frame = make_gradient_frame()
        ext = extract_band_b(frame, 256)
        b = compute_brr(PAYLOAD, ext)
        assert b < 70.0

    def test_fpr_mixed(self):
        frame = make_mixed_frame(seed=123)
        ext = extract_band_b(frame, 256)
        b = compute_brr(PAYLOAD, ext)
        assert b < 70.0


class TestTextureGating:
    """Texture assessment module."""

    def test_noise_high_quality(self):
        frame = make_noise_frame()
        report = assess_frame_quality(frame)
        assert report["quality"] in ("excellent", "good")

    def test_gradient_low_quality(self):
        frame = make_gradient_frame()
        report = assess_frame_quality(frame)
        assert report["quality"] in ("poor", "fair")

    def test_embed_ratio(self):
        frame = make_noise_frame()
        report = assess_frame_quality(frame)
        assert report["embed_ratio"] > 80.0


class TestVerifyInterface:
    """High-level verify interface."""

    def test_detect_watermarked(self):
        frame = make_noise_frame()
        result = embed_frame(frame, CONFIG)
        vresult = verify_frame(result.frame, CONFIG)
        assert vresult.detected is True
        assert vresult.confidence == "high"

    def test_reject_clean(self):
        frame = make_noise_frame(seed=999)
        vresult = verify_frame(frame, CONFIG)
        assert vresult.detected is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
