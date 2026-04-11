"""ChromaAI verify: Detect and verify watermarks in video."""

import numpy as np
import cv2
from .core import (
    EmbedConfig, VerifyResult, extract_band_b, extract_band_a,
    generate_payload, compute_brr
)


DETECTION_THRESHOLD = 70.0  # >70% BRR = watermark detected
HIGH_CONFIDENCE = 90.0
MEDIUM_CONFIDENCE = 80.0


def verify_frame(frame_gray: np.ndarray,
                 config: EmbedConfig = EmbedConfig()) -> VerifyResult:
    """
    Verify watermark in a single grayscale frame.

    Returns VerifyResult with detection status and confidence.
    """
    payload = generate_payload(config.payload_size, config.seed)

    ext_b = extract_band_b(frame_gray, config.payload_size)
    b_brr = compute_brr(payload, ext_b)

    ext_a = extract_band_a(frame_gray, config.payload_size)
    a_brr = compute_brr(payload, ext_a)

    detected = b_brr > DETECTION_THRESHOLD

    if b_brr >= HIGH_CONFIDENCE:
        confidence = "high"
    elif b_brr >= MEDIUM_CONFIDENCE:
        confidence = "medium"
    elif b_brr >= DETECTION_THRESHOLD:
        confidence = "low"
    else:
        confidence = "none"

    return VerifyResult(
        detected=detected,
        band_b_brr=b_brr,
        band_a_brr=a_brr,
        confidence=confidence,
        payload_match=b_brr > MEDIUM_CONFIDENCE,
    )


def verify_video(input_path: str,
                 config: EmbedConfig = EmbedConfig(),
                 max_frames: int = 30) -> dict:
    """
    Verify watermark across video frames.

    Returns aggregated detection result.
    """
    cap = cv2.VideoCapture(input_path)
    results = []

    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
        result = verify_frame(gray, config)
        results.append(result)
        count += 1
    cap.release()

    if not results:
        return {
            "detected": False,
            "confidence": "none",
            "frames_analyzed": 0,
            "error": "Could not read video"
        }

    b_brrs = [r.band_b_brr for r in results]
    a_brrs = [r.band_a_brr for r in results]
    detected_count = sum(1 for r in results if r.detected)

    mean_b = float(np.mean(b_brrs))
    detection_rate = detected_count / len(results)

    if mean_b >= HIGH_CONFIDENCE:
        overall_confidence = "high"
    elif mean_b >= MEDIUM_CONFIDENCE:
        overall_confidence = "medium"
    elif mean_b >= DETECTION_THRESHOLD:
        overall_confidence = "low"
    else:
        overall_confidence = "none"

    return {
        "detected": detection_rate > 0.5,
        "confidence": overall_confidence,
        "band_b_brr_mean": round(mean_b, 2),
        "band_b_brr_min": round(float(np.min(b_brrs)), 2),
        "band_a_brr_mean": round(float(np.mean(a_brrs)), 2),
        "detection_rate": round(detection_rate * 100, 1),
        "frames_analyzed": len(results),
    }
