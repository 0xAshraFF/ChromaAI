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

from src.core import (
    EmbedConfig, embed_band_b, extract_band_b,
    generate_payload, compute_psnr, compute_brr, compute_texture_energy,
    BAND_B_POSITION,
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
        'summary': {
            'old_attacks_passed': old_passed,
            'new_attacks_passed': new_passed,
            'old_mean_psnr': round(old_mp, 2),
            'new_mean_psnr': round(new_mp, 2),
        },
    }
    out_path = Path(__file__).parent / 'benchmark_results_task2.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    run()
