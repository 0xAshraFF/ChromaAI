"""
chroma-ai Core Engine
DCT-domain video watermarking with Task 2 calibrated alphas.
"""

import numpy as np
import cv2
import subprocess
import os
import tempfile
from scipy.fft import dctn, idctn


def compute_texture(block):
    """AC energy of 8x8 block."""
    d = dctn(block, type=2, norm='ortho')
    return float(np.sum(d**2) - d[0, 0]**2)


def text_to_bits(text, max_bits=256):
    """Convert text string to binary payload."""
    raw = text.encode('utf-8')
    # First 2 bytes = length, then data
    length = min(len(raw), (max_bits // 8) - 2)
    data = bytes([length >> 8, length & 0xFF]) + raw[:length]
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    # Pad to max_bits
    while len(bits) < max_bits:
        bits.append(0)
    return np.array(bits[:max_bits], dtype=np.uint8)


def bits_to_text(bits):
    """Convert binary payload back to text string."""
    if len(bits) < 16:
        return ""
    # First 2 bytes = length
    length = 0
    for i in range(16):
        length = (length << 1) | int(bits[i])
    if length <= 0 or length > (len(bits) // 8) - 2:
        return "[unreadable]"
    text_bits = bits[16:]
    chars = []
    for i in range(0, length * 8, 8):
        if i + 8 > len(text_bits):
            break
        byte = 0
        for j in range(8):
            byte = (byte << 1) | int(text_bits[i + j])
        chars.append(byte)
    try:
        return bytes(chars).decode('utf-8', errors='replace')
    except:
        return "[decode error]"


def embed_frame(frame_gray, payload):
    """
    Embed payload into a single grayscale frame using Task 2 alphas.
    Returns: (watermarked_frame, block_info_list)
    block_info_list: list of dicts with {row, col, texture, alpha, bit}
    """
    h, w = frame_gray.shape
    emb = frame_gray.copy()
    block_info = []
    bi = 0

    for by in range(0, h - 7, 8):
        for bx in range(0, w - 7, 8):
            if bi >= len(payload):
                break
            blk = emb[by:by+8, bx:bx+8].copy()
            tex = compute_texture(blk)

            # Task 2 calibrated alphas
            if tex < 100:
                local_a = 37  # flat + low
            else:
                local_a = 32  # medium + high

            d = dctn(blk, type=2, norm='ortho')
            bit = payload[bi]
            if bit == 1:
                d[0, 2] = abs(d[0, 2]) + local_a
            else:
                d[0, 2] = -(abs(d[0, 2]) + local_a)

            emb[by:by+8, bx:bx+8] = idctn(d, type=2, norm='ortho')

            block_info.append({
                'row': by, 'col': bx,
                'texture': tex, 'alpha': local_a, 'bit': int(bit)
            })
            bi += 1

    return emb, block_info


def extract_frame(frame_gray, n_bits):
    """Extract n_bits from a grayscale frame."""
    h, w = frame_gray.shape
    bits = []
    for by in range(0, h - 7, 8):
        for bx in range(0, w - 7, 8):
            if len(bits) >= n_bits:
                return np.array(bits, dtype=np.uint8)
            d = dctn(frame_gray[by:by+8, bx:bx+8], type=2, norm='ortho')
            bits.append(1 if d[0, 2] > 0 else 0)
    return np.array(bits, dtype=np.uint8)


def compute_brr(original, extracted):
    """Bit Recovery Rate (%)."""
    n = min(len(original), len(extracted))
    if n == 0:
        return 0.0
    return float(np.mean(original[:n] == extracted[:n]) * 100)


def compute_psnr(a, b):
    """Peak Signal-to-Noise Ratio."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255**2 / mse)


def make_heatmap(frame_gray, block_info, mode='alpha'):
    """
    Create a heatmap overlay showing embedding strength or texture.
    mode: 'alpha' or 'texture'
    Returns: RGB image with heatmap overlay
    """
    h, w = frame_gray.shape
    overlay = np.zeros((h, w), dtype=np.float64)

    for info in block_info:
        r, c = info['row'], info['col']
        if mode == 'alpha':
            val = info['alpha']
        else:
            val = min(info['texture'], 2000)
        overlay[r:r+8, c:c+8] = val

    # Normalize to 0-255
    if overlay.max() > overlay.min():
        overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min()) * 255
    overlay = overlay.astype(np.uint8)

    # Apply colormap
    heatmap = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)

    # Blend with original
    base = cv2.cvtColor(frame_gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(base, 0.5, heatmap, 0.5, 0)
    return blended


def attack_h265(frame_gray, crf=28):
    """Encode through H.265 and decode back."""
    h, w = frame_gray.shape
    with tempfile.TemporaryDirectory() as tmpdir:
        yuv_path = os.path.join(tmpdir, "input.yuv")
        mp4_path = os.path.join(tmpdir, "output.mp4")

        # Write YUV420p
        with open(yuv_path, 'wb') as f:
            y = np.clip(frame_gray, 0, 255).astype(np.uint8)
            f.write(y.tobytes())
            uv = np.full((h // 2, w // 2), 128, dtype=np.uint8)
            f.write(uv.tobytes())
            f.write(uv.tobytes())

        cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
            "-s", f"{w}x{h}", "-r", "30", "-i", yuv_path,
            "-c:v", "libx265", "-preset", "medium", "-crf", str(crf),
            "-x265-params", "ctu=32:min-cu-size=8",
            "-pix_fmt", "yuv420p", mp4_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return None

        cap = cv2.VideoCapture(mp4_path)
        ret, fr = cap.read()
        cap.release()
        if not ret:
            return None
        return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float64)


def attack_jpeg(frame_gray, quality=20):
    """JPEG compress and decompress."""
    u8 = np.clip(frame_gray, 0, 255).astype(np.uint8)
    _, buf = cv2.imencode('.jpg', u8, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float64)


def attack_blur(frame_gray, sigma=3):
    """Gaussian blur."""
    ksize = int(sigma * 4) | 1  # ensure odd
    u8 = np.clip(frame_gray, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(u8, (ksize, ksize), sigma).astype(np.float64)


def attack_resize(frame_gray, scale=0.9):
    """Resize down and back up."""
    h, w = frame_gray.shape
    u8 = np.clip(frame_gray, 0, 255).astype(np.uint8)
    small = cv2.resize(u8, (int(w * scale), int(h * scale)))
    back = cv2.resize(small, (w, h))
    return back.astype(np.float64)


def attack_youtube_proxy(frame_gray):
    """Simulate YouTube: resize to 360p, H.264 CRF23, resize back, JPEG Q75."""
    h, w = frame_gray.shape
    u8 = np.clip(frame_gray, 0, 255).astype(np.uint8)
    small = cv2.resize(u8, (640, 360))

    with tempfile.TemporaryDirectory() as tmpdir:
        yuv_path = os.path.join(tmpdir, "yt.yuv")
        mp4_path = os.path.join(tmpdir, "yt.mp4")
        sh, sw = small.shape

        with open(yuv_path, 'wb') as f:
            f.write(small.tobytes())
            uv = np.full((sh // 2, sw // 2), 128, dtype=np.uint8)
            f.write(uv.tobytes())
            f.write(uv.tobytes())

        cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
            "-s", f"{sw}x{sh}", "-r", "30", "-i", yuv_path,
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-pix_fmt", "yuv420p", mp4_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return None

        cap = cv2.VideoCapture(mp4_path)
        ret, fr = cap.read()
        cap.release()
        if not ret:
            return None

        decoded = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float64)

    back = cv2.resize(np.clip(decoded, 0, 255).astype(np.uint8), (w, h))
    _, buf = cv2.imencode('.jpg', back, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float64)
