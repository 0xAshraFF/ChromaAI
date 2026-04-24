"""
chroma-ai — Before/After Benchmark: Original vs Task 2 Alpha Lookup
========================================================================
Runs the same 20 clips × 12 attacks under both alpha configurations so
reviewers can see a consistent before/after on an identical attack matrix.

  OLD (base_alpha=12, multiplier table):
      tex < 10    →  12 × 3.0 = 36
      tex < 100   →  12 × 2.0 = 24
      tex < 1000  →  12 × 1.5 = 18
      tex ≥ 1000  →  12 × 1.0 = 12

  NEW (Task 2 absolute lookup, +2 safety margin):
      tex < 100   →  37   (flat + low)
      tex ≥ 100   →  32   (medium + high)

Primary attack matrix (12 attacks, matches original benchmark):
  none, h265_crf28, h265_crf32, h264_crf28,
  jpeg_q90, jpeg_q70, blur_sigma5, blur_sigma10,
  awgn_10db, crop_center, resize_07, resize_14

Supplementary attacks appended after the comparison table.

Usage:
    python3 run_benchmark.py

Output:
    benchmark_results_task2.json
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cv2
import subprocess
import json
import time
import tempfile
from pathlib import Path
from scipy.fft import dctn, idctn

import hashlib

from src.core import (
    EmbedConfig, embed_band_b, extract_band_b,
    generate_payload, compute_psnr, compute_brr, compute_texture_energy,
    BAND_B_POSITION, BAND_A_POSITIONS, ZIGZAG_8x8,
)

# ── Config ────────────────────────────────────────────────────────────────────
W, H       = 854, 480
N_FRAMES   = 5          # frames per clip; codec attacks encode all N_FRAMES
PAYLOAD_N  = 256        # bits embedded per frame
PASS_BRR   = 95.0       # ≥ 95 % BRR = attack passed
PSNR_GATE  = 30.0       # minimum acceptable embedding PSNR
WORK_DIR   = tempfile.mkdtemp(prefix='cc_bench_')

CONFIG_NEW = EmbedConfig(adaptive=True)   # Task 2 lookup (in src/core.py)
PAYLOAD    = generate_payload(PAYLOAD_N, seed=42)

# Demo master key for benchmark use only — never ship a hardcoded key.
DEMO_MASTER_KEY = hashlib.sha256(b"chroma-ai-bench-demo").digest()
WRONG_KEY       = hashlib.sha256(b"chroma-ai-bench-wrong").digest()


# ── Old alpha embed (original multiplier table) ───────────────────────────────
_BASE_ALPHA = 12.0

def embed_old_alpha(frame: np.ndarray, payload: np.ndarray) -> np.ndarray:
    """
    Original Band B embedding with base_alpha=12 multiplier table.
    Kept here so core.py is not reverted.
    """
    h, w  = frame.shape
    emb   = frame.astype(np.float64)
    r, c  = BAND_B_POSITION
    bi    = 0

    for by in range(0, h - 7, 8):
        for bx in range(0, w - 7, 8):
            if bi >= len(payload):
                return np.clip(emb, 0, 255).astype(np.uint8)
            blk = emb[by:by+8, bx:bx+8].copy()
            tex = compute_texture_energy(blk)
            if tex < 10:
                la = _BASE_ALPHA * 3.0    # 36
            elif tex < 100:
                la = _BASE_ALPHA * 2.0    # 24
            elif tex < 1000:
                la = _BASE_ALPHA * 1.5    # 18
            else:
                la = _BASE_ALPHA          # 12
            d = dctn(blk, type=2, norm='ortho')
            d[r, c] = (abs(d[r, c]) + la) if payload[bi] == 1 \
                      else -(abs(d[r, c]) + la)
            emb[by:by+8, bx:bx+8] = idctn(d, type=2, norm='ortho')
            bi += 1

    return np.clip(emb, 0, 255).astype(np.uint8)


# ── Clip generators (single-channel uint8 luma) ───────────────────────────────
def _rng(seed): return np.random.RandomState(seed)

CLIPS = {
    'clip01_flat_black':    lambda i: np.zeros((H, W), dtype=np.uint8),
    'clip02_flat_gray':     lambda i: np.full((H, W), 128, dtype=np.uint8),
    'clip03_flat_white':    lambda i: np.full((H, W), 255, dtype=np.uint8),
    'clip04_gradient_h':    lambda i: np.tile(np.linspace(0, 255, W).astype(np.uint8), (H, 1)),
    'clip05_gradient_v':    lambda i: np.tile(np.linspace(0, 255, H).astype(np.uint8), (W, 1)).T,
    'clip06_noise_low':     lambda i: _rng(i).randint(100, 156, (H, W)).astype(np.uint8),
    'clip07_noise_med':     lambda i: _rng(i + 1000).randint(50, 206, (H, W)).astype(np.uint8),
    'clip08_noise_high':    lambda i: _rng(i + 2000).randint(0, 256, (H, W)).astype(np.uint8),
    'clip09_checker_8':     lambda i: ((np.indices((H, W)).sum(0) // 8) % 2 * 255).astype(np.uint8),
    'clip10_checker_16':    lambda i: ((np.indices((H, W)).sum(0) // 16) % 2 * 255).astype(np.uint8),
    'clip11_bars_v':        lambda i: np.tile((np.arange(W) // (W // 8) % 2 * 255).astype(np.uint8), (H, 1)),
    'clip12_sine_lf':       lambda i: np.clip(128 + 60 * np.sin(np.arange(W) * 2 * np.pi * 2 / W), 0, 255).astype(np.uint8)[None].repeat(H, 0),
    'clip13_perlin_approx': lambda i: np.clip(128 + _rng(i + 100).normal(0, 20, (H, W)), 0, 255).astype(np.uint8),
    'clip14_edges':         lambda i: np.where((np.arange(W) % 32 < 2) | (np.arange(H)[:, None] % 32 < 2), 255, 0).astype(np.uint8),
    'clip15_film_grain':    lambda i: np.clip(128 + _rng(i + 200).normal(0, 15, (H, W)), 0, 255).astype(np.uint8),
    'clip16_color_morph':   lambda i: np.full((H, W), int(128 + 60 * np.sin(i / 5)) % 256, dtype=np.uint8),
    'clip17_smpte_bars':    lambda i: np.tile((np.arange(W) // (W // 7) * (255 // 6)).astype(np.uint8), (H, 1)),
    'clip18_rng_bw':        lambda i: (_rng(i + 300).randint(0, 2, (H, W)) * 255).astype(np.uint8),
    'clip19_low_contrast':  lambda i: _rng(i + 400).randint(120, 136, (H, W)).astype(np.uint8),
    'clip20_mixed':         lambda i: np.where(np.arange(W) < W // 2,
                                _rng(i + 500).randint(0, 256, (H, W)).astype(np.uint8),
                                np.full((H, W), 128, dtype=np.uint8)),
}


# ── Attack helpers ────────────────────────────────────────────────────────────
def _codec(frames, codec, crf):
    raw      = b''.join(f.tobytes() for f in frames)
    enc_path = os.path.join(WORK_DIR, f'tmp_{codec}_{crf}.mkv')
    subprocess.run(
        ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
         '-s', f'{W}x{H}', '-r', '30', '-i', 'pipe:0',
         '-c:v', f'lib{codec}', '-crf', str(crf), '-preset', 'fast', enc_path],
        input=raw, capture_output=True, check=True,
    )
    dec = subprocess.run(
        ['ffmpeg', '-y', '-i', enc_path, '-pix_fmt', 'gray', '-f', 'rawvideo', 'pipe:1'],
        capture_output=True, check=True,
    )
    fsize = H * W
    return [np.frombuffer(dec.stdout[k*fsize:(k+1)*fsize], dtype=np.uint8).reshape(H, W)
            for k in range(len(frames)) if (k+1)*fsize <= len(dec.stdout)]

def _jpeg(frames, q):
    out = []
    for f in frames:
        _, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, q])
        out.append(cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE))
    return out

def _resize(frames, scale):
    return [cv2.resize(cv2.resize(f, (int(W*scale), int(H*scale)),
            interpolation=cv2.INTER_LINEAR), (W, H), interpolation=cv2.INTER_LINEAR)
            for f in frames]

def _crop(frame, cw=512, ch=384):
    y0, x0 = (H - ch) // 2, (W - cw) // 2
    out    = np.zeros_like(frame)
    out[y0:y0+ch, x0:x0+cw] = frame[y0:y0+ch, x0:x0+cw]
    return out

def _awgn(frames, snr_db):
    rng_a = np.random.RandomState(7)
    out   = []
    for f in frames:
        fp  = f.astype(np.float64)
        sig = np.mean(fp ** 2)
        std = np.sqrt(sig / (10 ** (snr_db / 10)))
        out.append(np.clip(fp + rng_a.normal(0, std, fp.shape), 0, 255).astype(np.uint8))
    return out


# ── Primary attack matrix (12 — matches original benchmark) ──────────────────
PRIMARY_ATTACKS = {
    'none':         (lambda fs: list(fs),                          'No attack (baseline)'),
    'h265_crf28':   (lambda fs: _codec(fs, 'x265', 28),           'H.265 CRF28'),
    'h265_crf32':   (lambda fs: _codec(fs, 'x265', 32),           'H.265 CRF32'),
    'h264_crf28':   (lambda fs: _codec(fs, 'x264', 28),           'H.264 CRF28'),
    'jpeg_q90':     (lambda fs: _jpeg(fs, 90),                     'JPEG QF=90'),
    'jpeg_q70':     (lambda fs: _jpeg(fs, 70),                     'JPEG QF=70'),
    'blur_sigma5':  (lambda fs: [cv2.GaussianBlur(f, (5,5),  5) for f in fs], 'Blur σ=5'),
    'blur_sigma10': (lambda fs: [cv2.GaussianBlur(f, (9,9), 10) for f in fs], 'Blur σ=10'),
    'awgn_10db':    (lambda fs: _awgn(fs, 10),                    'AWGN 10 dB'),
    'crop_center':  (lambda fs: [_crop(f) for f in fs],           'Crop 512×384 center'),
    'resize_07':    (lambda fs: _resize(fs, 0.7),                 'Resize 0.7× → back'),
    'resize_14':    (lambda fs: _resize(fs, 1.4),                 'Resize 1.4× → back'),
}

# ── Supplementary attacks (additional coverage, not in original matrix) ───────
SUPPLEMENTARY_ATTACKS = {
    'jpeg_q50':     (lambda fs: _jpeg(fs, 50),                    'JPEG QF=50'),
    'blur_sigma2':  (lambda fs: [cv2.GaussianBlur(f, (5,5),  2) for f in fs], 'Blur σ=2'),
    'awgn_20db':    (lambda fs: _awgn(fs, 20),                    'AWGN 20 dB'),
}


# ── Informed-adversary attacks (run only against keyed + unkeyed Band B) ─────
_ZERO_POSITIONS = [BAND_B_POSITION] + [ZIGZAG_8x8[i] for i in range(14, 22)]


def attack_zero_known(frames):
    """Zero the published Band B + Band A coefficients on every 8×8 block.

    Simulates the paper-reader attack: the attacker reads the spec and
    surgically removes signal at every known embedding location.
    Should destroy the unkeyed signal while leaving the keyed signal mostly
    intact (keyed embeds at rotated/shuffled positions).
    """
    out = []
    for f in frames:
        fp = f.astype(np.float64).copy()
        hh, ww = fp.shape
        for by in range(0, hh - 7, 8):
            for bx in range(0, ww - 7, 8):
                d = dctn(fp[by:by+8, bx:bx+8], type=2, norm='ortho')
                for r, c in _ZERO_POSITIONS:
                    d[r, c] = 0.0
                fp[by:by+8, bx:bx+8] = idctn(d, type=2, norm='ortho')
        out.append(np.clip(fp, 0, 255).astype(np.uint8))
    return out


def attack_averaging(test_frames, pool_watermarked, pool_originals, N):
    """Worst-case oracle averaging attack (attacker has paired originals).

    Estimates template = mean(watermarked - original) across N paired samples,
    then subtracts from each test frame. Against an unkeyed additive watermark
    the template converges quickly; against a keyed watermark each block's
    residual sign/position varies, so the mean collapses toward zero.
    """
    n_use = min(N, len(pool_watermarked), len(pool_originals))
    residuals = [pool_watermarked[i].astype(np.float64) - pool_originals[i].astype(np.float64)
                 for i in range(n_use)]
    template = np.mean(np.stack(residuals, axis=0), axis=0)
    out = []
    for f in test_frames:
        attacked = f.astype(np.float64) - template
        out.append(np.clip(attacked, 0, 255).astype(np.uint8))
    return out


def _extract_brr_unkeyed(frame_u8):
    ext = extract_band_b(frame_u8.astype(np.float64), PAYLOAD_N)
    return compute_brr(PAYLOAD, ext)


def _extract_brr_keyed(frame_u8, key, frame_idx):
    from src.core import extract_band_b as ex_b
    ext = ex_b(frame_u8.astype(np.float64), PAYLOAD_N, key=key, frame_idx=frame_idx)
    # Keyed path may skip blocks (P=0.75 gating) so it may return < PAYLOAD_N bits.
    return compute_brr(PAYLOAD[:len(ext)], ext)


def run_informed_adversary(emb_unkeyed, emb_keyed, originals, accepted):
    """Run informed-adversary attack suite on both unkeyed and keyed variants.

    Returns a dict keyed by attack name with per-variant BRR stats.
    ``emb_unkeyed``, ``emb_keyed``, ``originals`` are dicts:
        clip_name -> list of N_FRAMES arrays.
    """
    results = {}

    # Flatten pools for averaging: (watermarked_frame, original_frame) pairs.
    pool_wm_u, pool_wm_k, pool_o = [], [], []
    for name in accepted:
        pool_wm_u.extend(emb_unkeyed[name])
        pool_wm_k.extend(emb_keyed[name])
        pool_o.extend(originals[name])

    # ── attack_zero_known ────────────────────────────────────────────────────
    u_brrs, k_brrs = [], []
    for name in accepted:
        atk_u = attack_zero_known(emb_unkeyed[name])
        atk_k = attack_zero_known(emb_keyed[name])
        for f in atk_u:
            u_brrs.append(_extract_brr_unkeyed(f))
        for idx, f in enumerate(atk_k):
            k_brrs.append(_extract_brr_keyed(f, DEMO_MASTER_KEY, idx))
    results['zero_known'] = {
        'desc': 'Surgical DCT zeroing at published positions',
        'unkeyed_mean_brr': float(np.mean(u_brrs)),
        'keyed_mean_brr':   float(np.mean(k_brrs)),
        'n_samples':        len(u_brrs),
    }

    # ── attack_averaging_N ───────────────────────────────────────────────────
    for N in (10, 50, 200):
        u_brrs, k_brrs = [], []
        for name in accepted:
            atk_u = attack_averaging(emb_unkeyed[name], pool_wm_u, pool_o, N)
            atk_k = attack_averaging(emb_keyed[name],   pool_wm_k, pool_o, N)
            for f in atk_u:
                u_brrs.append(_extract_brr_unkeyed(f))
            for idx, f in enumerate(atk_k):
                k_brrs.append(_extract_brr_keyed(f, DEMO_MASTER_KEY, idx))
        results[f'averaging_{N}'] = {
            'desc': f'Oracle template averaging, N={N} paired samples',
            'unkeyed_mean_brr': float(np.mean(u_brrs)),
            'keyed_mean_brr':   float(np.mean(k_brrs)),
            'n_samples':        len(u_brrs),
        }

    # ── attack_wrong_key ─────────────────────────────────────────────────────
    # No frame modification — just extract with the wrong key on the keyed
    # stream (and with no key on the unkeyed stream for parity).
    u_brrs, k_brrs = [], []
    for name in accepted:
        for idx, f in enumerate(emb_unkeyed[name]):
            u_brrs.append(_extract_brr_unkeyed(f))   # baseline (same-key path)
        for idx, f in enumerate(emb_keyed[name]):
            k_brrs.append(_extract_brr_keyed(f, WRONG_KEY, idx))
    results['wrong_key'] = {
        'desc': 'Extract keyed stream with a different master key',
        'unkeyed_mean_brr': float(np.mean(u_brrs)),
        'keyed_mean_brr':   float(np.mean(k_brrs)),
        'n_samples':        len(u_brrs),
    }

    return results


# ── Core sweep: run one embed config through one attack set ───────────────────
def sweep(embedded: dict, accepted: list, attacks: dict) -> dict:
    """
    For each attack, apply to every accepted clip's embedded frames,
    extract, and compute BRR.  Returns {attack_name: result_dict}.
    """
    results = {}
    for atk_name, (atk_fn, atk_desc) in attacks.items():
        brrs = []
        t0   = time.time()
        err  = None
        for name in accepted:
            try:
                attacked = atk_fn(list(embedded[name]))
            except Exception as e:
                err = str(e); break
            for f in attacked:
                brrs.append(compute_brr(PAYLOAD, extract_band_b(f.astype(np.float64), PAYLOAD_N)))
        elapsed = time.time() - t0

        if err or not brrs:
            results[atk_name] = {'desc': atk_desc, 'error': err or 'no data'}
            continue

        mean_brr = float(np.mean(brrs))
        results[atk_name] = {
            'desc':       atk_desc,
            'mean_brr':   mean_brr,
            'min_brr':    float(np.min(brrs)),
            'pass_count': sum(1 for b in brrs if b >= PASS_BRR),
            'n_samples':  len(brrs),
            'pass_rate':  round(sum(1 for b in brrs if b >= PASS_BRR) / len(brrs) * 100, 1),
            'passed':     mean_brr >= PASS_BRR,
            'time_s':     round(elapsed, 2),
        }
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    t0_total = time.time()

    print('=' * 76)
    print('chroma-ai Benchmark — Before/After Alpha Lookup Comparison')
    print(f'  {len(CLIPS)} clips  ×  {N_FRAMES} frames  ×  {len(PRIMARY_ATTACKS)} primary attacks')
    print(f'  Pass threshold : BRR ≥ {PASS_BRR}%   PSNR gate : ≥ {PSNR_GATE} dB')
    print('=' * 76)

    # ── Embed both configs ───────────────────────────────────────────────────
    print('\nEmbedding — OLD alphas (base=12, multiplier table)…')
    emb_old, psnr_old = {}, {}
    for name, gen in CLIPS.items():
        origs = [gen(i).astype(np.float64) for i in range(N_FRAMES)]
        wms   = [embed_old_alpha(o, PAYLOAD) for o in origs]
        emb_old[name]  = [np.clip(w, 0, 255).astype(np.uint8) for w in wms]
        psnr_old[name] = float(np.mean([compute_psnr(o, w) for o, w in zip(origs, wms)]))
        print(f'  {name:<28}  PSNR={psnr_old[name]:5.1f} dB')

    print('\nEmbedding — NEW alphas (Task 2: tex<100→37, tex≥100→32)…')
    emb_new, psnr_new = {}, {}
    for name, gen in CLIPS.items():
        origs = [gen(i).astype(np.float64) for i in range(N_FRAMES)]
        wms   = [embed_band_b(o, PAYLOAD, CONFIG_NEW) for o in origs]
        emb_new[name]  = [np.clip(w, 0, 255).astype(np.uint8) for w in wms]
        psnr_new[name] = float(np.mean([compute_psnr(o, w) for o, w in zip(origs, wms)]))
        print(f'  {name:<28}  PSNR={psnr_new[name]:5.1f} dB')

    # quality gate (applied equally to both)
    accepted = [n for n in CLIPS if psnr_new[n] >= PSNR_GATE and psnr_old[n] >= PSNR_GATE]
    print(f'\nAccepted {len(accepted)}/{len(CLIPS)} clips'
          f'  |  OLD mean PSNR {np.mean([psnr_old[n] for n in accepted]):.1f} dB'
          f'  |  NEW mean PSNR {np.mean([psnr_new[n] for n in accepted]):.1f} dB')

    # ── FPR (same clean frames, independent of alpha config) ────────────────
    print('\nFPR check (clean frames)…')
    fp = 0
    for name, gen in CLIPS.items():
        ext = extract_band_b(gen(0).astype(np.float64), PAYLOAD_N)
        if compute_brr(PAYLOAD, ext) >= PASS_BRR:
            fp += 1
            print(f'  !! FP: {name}')
    print(f'  FPR: {fp}/{len(CLIPS)}')

    # ── Primary sweep ────────────────────────────────────────────────────────
    print('\nRunning primary attacks (OLD)…')
    res_old = sweep(emb_old, accepted, PRIMARY_ATTACKS)

    print('Running primary attacks (NEW)…')
    res_new = sweep(emb_new, accepted, PRIMARY_ATTACKS)

    # ── Comparison table ─────────────────────────────────────────────────────
    print('\n' + '=' * 76)
    print('PRIMARY ATTACK COMPARISON  (20 clips × 5 frames, pass = mean BRR ≥ 95%)')
    print('=' * 76)
    hdr = f'  {"Attack":<18}  {"Description":<22}  {"OLD BRR":>8}  {"NEW BRR":>8}  {"Δ":>6}  {"OLD":>5}  {"NEW":>5}'
    print(hdr)
    print('-' * 76)

    for atk_name, (_, atk_desc) in PRIMARY_ATTACKS.items():
        ro = res_old.get(atk_name, {})
        rn = res_new.get(atk_name, {})
        if 'error' in ro or 'error' in rn:
            print(f'  {atk_name:<18}  {atk_desc:<22}  ERROR')
            continue
        old_brr = ro['mean_brr']
        new_brr = rn['mean_brr']
        delta   = new_brr - old_brr
        old_ok  = '✓' if ro['passed'] else '✗'
        new_ok  = '✓' if rn['passed'] else '✗'
        sign    = '+' if delta >= 0 else ''
        print(f'  {atk_name:<18}  {atk_desc:<22}  {old_brr:>7.1f}%  {new_brr:>7.1f}%  '
              f'{sign}{delta:>5.1f}%  {old_ok:>5}  {new_ok:>5}')

    old_passed = sum(1 for v in res_old.values() if v.get('passed'))
    new_passed = sum(1 for v in res_new.values() if v.get('passed'))
    print('-' * 76)
    print(f'  {"Attacks passed":<42}  {old_passed:>7}/12   {new_passed:>7}/12')
    old_mp = np.mean([psnr_old[n] for n in accepted])
    new_mp = np.mean([psnr_new[n] for n in accepted])
    print(f'  {"Mean PSNR (accepted clips)":<42}  {old_mp:>6.1f} dB  {new_mp:>6.1f} dB')
    print(f'  {"FPR":<42}  {fp}/20 (unchanged)')
    print()

    # ── Supplementary attacks (new config only) ──────────────────────────────
    print('Running supplementary attacks (NEW alphas only)…')
    res_supp = sweep(emb_new, accepted, SUPPLEMENTARY_ATTACKS)

    # ── Informed-adversary: build keyed embeddings + attack suite ───────────
    print('\nEmbedding — keyed variant (ChaCha20/HKDF)…')
    from src.core import embed_band_b as _eb
    originals = {}
    emb_keyed = {}
    for name, gen in CLIPS.items():
        origs = [gen(i).astype(np.float64) for i in range(N_FRAMES)]
        originals[name] = [np.clip(o, 0, 255).astype(np.uint8) for o in origs]
        keyed = [_eb(o, PAYLOAD, CONFIG_NEW, key=DEMO_MASTER_KEY, frame_idx=i)
                 for i, o in enumerate(origs)]
        emb_keyed[name] = [np.clip(k, 0, 255).astype(np.uint8) for k in keyed]

    print('Running informed-adversary attacks…')
    res_informed = run_informed_adversary(emb_new, emb_keyed, originals, accepted)

    print('\nINFORMED-ADVERSARY ATTACKS (unkeyed vs keyed)')
    print('-' * 76)
    print(f'  {"Attack":<22}  {"Unkeyed BRR":>12}  {"Keyed BRR":>12}  {"Δ (keyed-unkeyed)":>18}')
    print('-' * 76)
    for atk, r in res_informed.items():
        delta = r['keyed_mean_brr'] - r['unkeyed_mean_brr']
        print(f'  {atk:<22}  {r["unkeyed_mean_brr"]:>11.1f}%  {r["keyed_mean_brr"]:>11.1f}%  {delta:>+17.1f}%')

    # ── Channel-noise regression guard ──────────────────────────────────────
    baseline_crf28 = 95.9
    actual_crf28 = res_new.get('h265_crf28', {}).get('mean_brr')
    if actual_crf28 is not None:
        delta = actual_crf28 - baseline_crf28
        tag = 'OK' if delta >= -2.0 else 'CHANNEL_NOISE_REGRESSION'
        print(f'\n[{tag}] h265_crf28 new={actual_crf28:.1f}% vs baseline {baseline_crf28:.1f}% '
              f'(Δ {delta:+.1f}%)')

    print('\nSUPPLEMENTARY ATTACKS (Task 2 alphas, not in original matrix)')
    print('-' * 60)
    print(f'  {"Attack":<18}  {"Description":<22}  {"NEW BRR":>8}  {"Pass":>5}')
    print('-' * 60)
    for atk_name, (_, atk_desc) in SUPPLEMENTARY_ATTACKS.items():
        r = res_supp.get(atk_name, {})
        if 'error' in r:
            print(f'  {atk_name:<18}  {atk_desc:<22}  ERROR')
            continue
        mark = '✓' if r['passed'] else '✗'
        print(f'  {atk_name:<18}  {atk_desc:<22}  {r["mean_brr"]:>7.1f}%  {mark:>5}')

    # ── Final summary ────────────────────────────────────────────────────────
    print('\n' + '=' * 76)
    print('SUMMARY')
    print(f'  Clips accepted:        {len(accepted)}/20')
    print(f'  PSNR (old → new):      {old_mp:.1f} dB → {new_mp:.1f} dB'
          f'  ({new_mp - old_mp:+.1f} dB)')
    print(f'  Attacks passed:        {old_passed}/12 → {new_passed}/12')
    print(f'  FPR:                   {fp}/20')
    for key in ('h265_crf28', 'h265_crf32'):
        ro = res_old.get(key, {}); rn = res_new.get(key, {})
        if 'mean_brr' in ro and 'mean_brr' in rn:
            desc = PRIMARY_ATTACKS[key][1]
            print(f'  {desc:<22}  {ro["mean_brr"]:.1f}% → {rn["mean_brr"]:.1f}%'
                  f'  ({"PASS" if rn["passed"] else "FAIL"})')
    print(f'  Total time:            {time.time() - t0_total:.0f}s')
    print('=' * 76)

    # ── Save JSON ────────────────────────────────────────────────────────────
    out = {
        'config': {
            'old_alpha': {'base': 12, 'multipliers': {'flat': 3.0, 'low': 2.0, 'medium': 1.5, 'high': 1.0}},
            'new_alpha': {'tex_lt_100': 37, 'tex_gte_100': 32},
            'payload_bits': PAYLOAD_N,
            'frames_per_clip': N_FRAMES,
            'psnr_gate_db': PSNR_GATE,
            'pass_brr_pct': PASS_BRR,
            'clips': len(CLIPS),
            'accepted_clips': len(accepted),
        },
        'clip_psnr': {n: {'old': round(psnr_old[n], 2), 'new': round(psnr_new[n], 2)}
                      for n in CLIPS},
        'fpr_count': fp,
        'primary_attacks': {
            atk: {
                'desc': PRIMARY_ATTACKS[atk][1],
                'old':  res_old.get(atk, {}),
                'new':  res_new.get(atk, {}),
            }
            for atk in PRIMARY_ATTACKS
        },
        'supplementary_attacks': res_supp,
        'informed_adversary': res_informed,
        'summary': {
            'old_attacks_passed': old_passed,
            'new_attacks_passed': new_passed,
            'old_mean_psnr': round(old_mp, 2),
            'new_mean_psnr': round(new_mp, 2),
            'channel_noise_crf28_mean_brr': actual_crf28,
            'channel_noise_crf28_baseline': baseline_crf28,
        },
    }
    out_path = Path(__file__).parent / 'benchmark_results_task2.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    run()
