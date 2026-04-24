"""chroma-ai texture gating: Pre-embed quality assessment."""

import numpy as np
from .core import frame_texture_stats, compute_texture_energy


def assess_frame_quality(frame_gray: np.ndarray,
                         min_texture: float = 50.0) -> dict:
    """
    Assess whether a frame has sufficient texture for reliable watermarking.

    Returns quality report with recommendation.
    """
    mean_tex, median_tex = frame_texture_stats(frame_gray)
    h, w = frame_gray.shape
    total_blocks = (h // 8) * (w // 8)

    # Count blocks above texture threshold
    embeddable = 0
    for by in range(0, h - 7, 8):
        for bx in range(0, w - 7, 8):
            if compute_texture_energy(frame_gray[by:by+8, bx:bx+8]) >= min_texture:
                embeddable += 1

    embed_ratio = embeddable / total_blocks if total_blocks > 0 else 0

    if mean_tex >= 500:
        quality = "excellent"
        recommendation = "Full embedding. High BRR expected (>95%)."
    elif mean_tex >= 100:
        quality = "good"
        recommendation = "Standard embedding. Good BRR expected (>85%)."
    elif mean_tex >= 50:
        quality = "fair"
        recommendation = "Band B reliable. Band A may have gaps."
    else:
        quality = "poor"
        recommendation = (
            "Content has very low texture. Band B uses adaptive boosting. "
            "Band A will skip most blocks. Watermark may not survive heavy compression."
        )

    return {
        "quality": quality,
        "mean_texture_energy": round(mean_tex, 1),
        "median_texture_energy": round(median_tex, 1),
        "embeddable_blocks": embeddable,
        "total_blocks": total_blocks,
        "embed_ratio": round(embed_ratio * 100, 1),
        "recommendation": recommendation,
    }
