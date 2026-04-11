"""ChromaCascade embed: Watermark video frames with dual-band DCT."""

import numpy as np
import cv2
from .core import (
    EmbedConfig, EmbedResult, embed_band_b, embed_band_a,
    generate_payload, compute_psnr, frame_texture_stats
)


def embed_frame(frame_gray: np.ndarray,
                config: EmbedConfig = EmbedConfig()) -> EmbedResult:
    """
    Embed watermark into a single grayscale frame.

    Args:
        frame_gray: 2D float64 array (Y channel)
        config: EmbedConfig with alpha, gating, etc.

    Returns:
        EmbedResult with embedded frame and metrics.
    """
    payload = generate_payload(config.payload_size, config.seed)
    mean_tex, _ = frame_texture_stats(frame_gray)

    # Band B (ownership signal — always embedded)
    embedded = embed_band_b(frame_gray, payload, config)

    # Band A (payload capacity — texture-gated)
    embedded, used, skipped = embed_band_a(embedded, payload, config)

    p = compute_psnr(frame_gray, embedded)

    return EmbedResult(
        frame=embedded,
        psnr=p,
        blocks_embedded=used,
        blocks_skipped=skipped,
        quality_accepted=True,  # We always embed Band B; gating is Band A only
        mean_texture=mean_tex,
    )


def embed_video(input_path: str, output_path: str,
                config: EmbedConfig = EmbedConfig(),
                max_frames: int = 300) -> dict:
    """
    Embed watermark into a video file.

    Returns dict with per-frame metrics.
    """
    import subprocess
    import os

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    psnrs = []
    count = 0

    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
        result = embed_frame(gray, config)
        frames.append(result.frame)
        psnrs.append(result.psnr)
        count += 1
    cap.release()

    if not frames:
        raise ValueError(f"Could not read frames from {input_path}")

    # Write via raw YUV → ffmpeg H.265
    yuv_path = output_path + ".tmp.yuv"
    with open(yuv_path, 'wb') as f:
        for frame in frames:
            y = np.clip(frame, 0, 255).astype(np.uint8)
            f.write(y.tobytes())
            uv = np.full((height // 2, width // 2), 128, dtype=np.uint8)
            f.write(uv.tobytes())
            f.write(uv.tobytes())

    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
        "-s", f"{width}x{height}", "-r", str(int(fps)),
        "-i", yuv_path,
        "-c:v", "libx265", "-preset", "medium", "-crf", "20",
        "-x265-params", "ctu=32:min-cu-size=8",
        "-pix_fmt", "yuv420p", output_path
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)
    os.remove(yuv_path)

    return {
        "frames": count,
        "mean_psnr": float(np.mean(psnrs)),
        "min_psnr": float(np.min(psnrs)),
        "resolution": f"{width}x{height}",
    }
