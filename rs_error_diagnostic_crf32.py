"""
rs_error_diagnostic_crf32.py — Reed-Solomon Error Pattern Analysis (CRF32)
===========================================================================
Identical to rs_error_diagnostic.py but:
  - Round-trips through H.265 CRF32 (harder codec setting)
  - Adds 3 pathological frames: flat_black, low_contrast_gradient, checkerboard
  - 8 frames total

Output: printed tables + rs_error_diagnostic_crf32_results.json
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cv2
import subprocess
import json
import tempfile
from pathlib import Path
from scipy.fft import dctn, idctn

# ── Constants (matching chroma_cascade.py) ────────────────────────────────────
W, H           = 854, 480
BPR            = W // 8          # blocks per row  = 106
BPC            = H // 8          # blocks per col  =  60
TOTAL_BLOCKS   = BPR * BPC       # 6 360
BAND_B         = (0, 2)          # DCT position carrying the payload
N_BITS         = 256             # payload length
BASE_ALPHA     = 12.0            # original base strength
GROUP_SIZE     = 8               # bits per RS symbol group
N_GROUPS       = N_BITS // GROUP_SIZE   # 32
CRF            = 32              # ← CRF32
WORK_DIR       = tempfile.mkdtemp(prefix='rs_diag_crf32_')

RNG     = np.random.RandomState(42)
PAYLOAD = RNG.randint(0, 2, N_BITS).astype(np.uint8)


# ── Texture / alpha helpers ───────────────────────────────────────────────────
def texture_energy(block: np.ndarray) -> float:
    """AC energy of an 8×8 block in ortho-normalised DCT domain."""
    d = dctn(block.astype(np.float64), type=2, norm='ortho')
    return float(np.sum(d ** 2) - d[0, 0] ** 2)


def adaptive_alpha(tex: float) -> float:
    """
    Original multiplier table (base_alpha=12):
        tex < 10    →  12 × 3.0 = 36
        tex < 100   →  12 × 2.0 = 24
        tex < 1000  →  12 × 1.5 = 18
        tex ≥ 1000  →  12 × 1.0 = 12
    """
    if tex < 10:
        return BASE_ALPHA * 3.0
    elif tex < 100:
        return BASE_ALPHA * 2.0
    elif tex < 1000:
        return BASE_ALPHA * 1.5
    return BASE_ALPHA


# ── Embed / extract (Band B only) ────────────────────────────────────────────
def embed(frame: np.ndarray, payload: np.ndarray) -> tuple:
    """
    Embed `payload` into Band B of a single-channel float64 frame.
    Returns (embedded_uint8, block_alpha_list).
    """
    h, w    = frame.shape
    emb     = frame.copy()
    r, c    = BAND_B
    bi      = 0
    alphas  = []

    for by in range(0, h - 7, 8):
        for bx in range(0, w - 7, 8):
            if bi >= len(payload):
                return np.clip(emb, 0, 255).astype(np.uint8), alphas
            blk = emb[by:by+8, bx:bx+8].copy()
            tex = texture_energy(blk)
            la  = adaptive_alpha(tex)
            alphas.append(la)

            d = dctn(blk, type=2, norm='ortho')
            d[r, c] = (abs(d[r, c]) + la) if payload[bi] == 1 \
                      else -(abs(d[r, c]) + la)
            emb[by:by+8, bx:bx+8] = idctn(d, type=2, norm='ortho')
            bi += 1

    return np.clip(emb, 0, 255).astype(np.uint8), alphas


def extract(frame: np.ndarray, n: int) -> np.ndarray:
    """Extract n bits from Band B of a single-channel frame."""
    h, w  = frame.shape
    bits  = []
    r, c  = BAND_B

    for by in range(0, h - 7, 8):
        for bx in range(0, w - 7, 8):
            if len(bits) >= n:
                return np.array(bits, dtype=np.uint8)
            d = dctn(frame[by:by+8, bx:bx+8].astype(np.float64), type=2, norm='ortho')
            bits.append(1 if d[r, c] >= 0 else 0)

    return np.array(bits[:n], dtype=np.uint8)


# ── H.265 round-trip ──────────────────────────────────────────────────────────
def h265_roundtrip(frame_u8: np.ndarray, crf: int = CRF) -> np.ndarray:
    """
    Single-frame H.265 encode/decode at given CRF.
    Uses ctu=32:min-cu-size=8 per chroma_cascade.py specification.
    """
    enc_path = os.path.join(WORK_DIR, f'diag_crf{crf}.mkv')

    subprocess.run(
        ['ffmpeg', '-y',
         '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{W}x{H}', '-r', '1',
         '-i', 'pipe:0',
         '-c:v', 'libx265', '-crf', str(crf), '-preset', 'medium',
         '-x265-params', 'ctu=32:min-cu-size=8',
         '-pix_fmt', 'yuv420p', enc_path],
        input=frame_u8.tobytes(), capture_output=True, check=True,
    )

    dec = subprocess.run(
        ['ffmpeg', '-y', '-i', enc_path,
         '-pix_fmt', 'gray', '-f', 'rawvideo', 'pipe:1'],
        capture_output=True, check=True,
    )
    fsize = H * W
    raw   = dec.stdout[:fsize]
    if len(raw) < fsize:
        raise RuntimeError(f'decode underrun: got {len(raw)} bytes, expected {fsize}')
    return np.frombuffer(raw, dtype=np.uint8).reshape(H, W)


# ── Frame generators ──────────────────────────────────────────────────────────
def frame_gradient() -> np.ndarray:
    """Horizontal luminance ramp — very flat, dominated by DC."""
    return np.tile(np.linspace(30, 225, W).astype(np.float64), (H, 1))


def frame_noise() -> np.ndarray:
    """Full-range random noise — maximum AC energy."""
    return np.random.RandomState(1).randint(0, 256, (H, W)).astype(np.float64)


def frame_perlin() -> np.ndarray:
    """Perlin-like organic texture from summed sinusoid octaves."""
    rng  = np.random.RandomState(2)
    out  = np.zeros((H, W), dtype=np.float64)
    amp, freq = 60.0, 1.0
    for _ in range(5):
        px = rng.uniform(0, 2 * np.pi)
        py = rng.uniform(0, 2 * np.pi)
        xs = np.linspace(0, freq * 2 * np.pi, W) + px
        ys = np.linspace(0, freq * 2 * np.pi, H) + py
        out += amp * np.outer(np.sin(ys), np.cos(xs))
        amp  *= 0.5
        freq *= 2.1
    return np.clip(128.0 + out, 0, 255)


def frame_edges() -> np.ndarray:
    """Grid of sharp edges every 32 pixels."""
    base = np.full((H, W), 128.0)
    base[np.arange(H) % 32 < 2, :] = 255.0
    base[:, np.arange(W) % 32 < 2] = 0.0
    return base


def frame_mixed() -> np.ndarray:
    """Left half: smooth gradient; right half: noise."""
    out           = frame_gradient().copy()
    rng           = np.random.RandomState(3)
    out[:, W//2:] = rng.randint(0, 256, (H, W - W//2)).astype(np.float64)
    return out


def frame_flat_black() -> np.ndarray:
    """
    Flat black (Y=16, broadcast legal minimum).
    Nearly-zero AC energy → maximum multiplier (×3.0 → α=36), but the
    codec can represent this very efficiently and still destroy low-alpha
    embedded content at CRF32.
    """
    return np.full((H, W), 16.0, dtype=np.float64)


def frame_low_contrast_gradient() -> np.ndarray:
    """
    Gentle horizontal gradient ranging 100–130 (30 luma units total).
    Minimal AC energy per block → adaptive alpha will be at or near ×3.0.
    Tests whether the codec compresses nearly-flat regions aggressively at CRF32.
    """
    return np.tile(np.linspace(100, 130, W).astype(np.float64), (H, 1))


def frame_checkerboard() -> np.ndarray:
    """
    8×8-pixel hard checkerboard (0 and 255).
    Maximum high-frequency energy within each 8×8 DCT block — the energy
    lands entirely in AC coefficients, driving adaptive alpha to its lowest
    multiplier (×1.0 → α=12).  This is the worst-case for Band B survival
    because the codec's quantisation targets exactly these coefficients.
    """
    xs = np.arange(W) // 8
    ys = np.arange(H) // 8
    return (((xs[np.newaxis, :] + ys[:, np.newaxis]) % 2) * 255).astype(np.float64)


FRAMES = [
    ('gradient',             frame_gradient()),
    ('noise',                frame_noise()),
    ('perlin',               frame_perlin()),
    ('edges',                frame_edges()),
    ('mixed',                frame_mixed()),
    ('flat_black',           frame_flat_black()),
    ('low_contrast_gradient',frame_low_contrast_gradient()),
    ('checkerboard',         frame_checkerboard()),
]


# ── Analysis helpers ──────────────────────────────────────────────────────────
def max_consecutive_run(errors: np.ndarray) -> int:
    best = cur = 0
    for b in errors:
        if b:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def group_error_counts(errors: np.ndarray, group_size: int = GROUP_SIZE) -> list:
    n = len(errors) // group_size
    return [int(errors[i*group_size:(i+1)*group_size].sum()) for i in range(n)]


def bit_to_block(bit_idx: int) -> tuple:
    return divmod(bit_idx, BPR)


def spatial_cluster_score(error_positions: list) -> float:
    if len(error_positions) < 2:
        return 0.0
    blocks = [bit_to_block(p) for p in error_positions]
    adj = total = 0
    for i in range(len(blocks)):
        for j in range(i + 1, min(i + 5, len(blocks))):
            dr = abs(blocks[i][0] - blocks[j][0])
            dc = abs(blocks[i][1] - blocks[j][1])
            if dr + dc <= 2:
                adj += 1
            total += 1
    return adj / total if total else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    all_results = {}

    print('=' * 72)
    print(f'Reed-Solomon Error Pattern Diagnostic — H.265 CRF{CRF} (ctu=32:min-cu-size=8)')
    print(f'  Payload: {N_BITS} bits  |  Groups: {N_GROUPS} × {GROUP_SIZE} bits  |  base_alpha={int(BASE_ALPHA)}')
    print('=' * 72)

    frame_summaries = []

    for name, frame in FRAMES:
        print(f'\n── {name.upper()} ──────────────────────────────────────────────')

        wm_u8, alphas = embed(frame, PAYLOAD)

        alpha_counts = {}
        for a in alphas[:N_BITS]:
            alpha_counts[a] = alpha_counts.get(a, 0) + 1
        print(f'  Alpha distribution (first {N_BITS} blocks):')
        for a_val in sorted(alpha_counts):
            pct = alpha_counts[a_val] / N_BITS * 100
            print(f'    α={int(a_val):2d} ({a_val/BASE_ALPHA:.1f}×)  {alpha_counts[a_val]:4d} blocks  ({pct:5.1f}%)')

        try:
            decoded = h265_roundtrip(wm_u8)
        except subprocess.CalledProcessError as e:
            print(f'  ffmpeg ERROR: {e.stderr.decode()[-200:]}')
            continue

        extracted = extract(decoded.astype(np.float64), N_BITS)
        errors    = (PAYLOAD[:N_BITS] != extracted[:N_BITS]).astype(np.uint8)

        n_errors   = int(errors.sum())
        brr        = (1 - n_errors / N_BITS) * 100
        err_pos    = list(np.where(errors)[0])
        max_run    = max_consecutive_run(errors)
        group_errs = group_error_counts(errors)
        groups_any = sum(1 for g in group_errs if g > 0)
        groups_maj = sum(1 for g in group_errs if g > GROUP_SIZE // 2)
        spatial_sc = spatial_cluster_score(err_pos)

        print(f'\n  BRR:              {brr:.1f}%  ({n_errors}/{N_BITS} errors)')
        print(f'  Max consec run:   {max_run} bits')
        print(f'  Groups any err:   {groups_any}/{N_GROUPS}')
        print(f'  Groups >50% err:  {groups_maj}/{N_GROUPS}  (majority-corrupt)')
        print(f'  Spatial cluster:  {spatial_sc:.3f}  (0=scattered, 1=adjacent)')

        print(f'\n  Group error counts (32 groups of 8 consecutive bits):')
        row = '  '
        for i, g in enumerate(group_errs):
            marker = '!' if g > GROUP_SIZE // 2 else ('.' if g == 0 else str(g))
            row += f'{marker:>2}'
            if (i + 1) % 16 == 0:
                print(row); row = '  '
        if row.strip():
            print(row)
        print('  (legend: . = no error, 1-4 = count, ! = majority-corrupt ≥5/8)')

        print(f'\n  Spatial error map (each cell = 4×4 block region, X=any error):')
        map_rows = (BPC + 3) // 4   # ceil div so last partial band is included
        map_cols = (BPR + 3) // 4
        err_map = np.zeros((map_rows, map_cols), dtype=int)
        for pos in err_pos:
            br, bc = bit_to_block(pos)
            err_map[min(br // 4, map_rows - 1), min(bc // 4, map_cols - 1)] += 1
        for mr in range(map_rows):
            row_str = '  '
            for mc in range(map_cols):
                row_str += 'X' if err_map[mr, mc] > 0 else '.'
            print(row_str)

        alpha_err_dist = {}
        if err_pos:
            err_alphas = [alphas[p] for p in err_pos if p < len(alphas)]
            for a in err_alphas:
                alpha_err_dist[a] = alpha_err_dist.get(a, 0) + 1
            print(f'\n  Alpha values at error positions:')
            for a_val in sorted(alpha_err_dist):
                n_a_total = alpha_counts.get(a_val, 0)
                n_a_err   = alpha_err_dist[a_val]
                err_rate  = n_a_err / n_a_total * 100 if n_a_total else 0
                print(f'    α={int(a_val):2d}  {n_a_err:3d}/{n_a_total} errors  ({err_rate:.1f}% block-err-rate)')

        frame_summaries.append({
            'frame':                   name,
            'brr':                     round(brr, 2),
            'n_errors':                n_errors,
            'max_consec_run':          max_run,
            'groups_any_err':          groups_any,
            'groups_majority_corrupt': groups_maj,
            'spatial_cluster_score':   round(spatial_sc, 4),
            'group_error_counts':      group_errs,
            'error_positions':         err_pos,
            'alpha_distribution':      {str(int(k)): v for k, v in alpha_counts.items()},
            'alpha_at_errors':         {str(int(k)): v for k, v in alpha_err_dist.items()},
        })
        all_results[name] = frame_summaries[-1]

    # ── Summary table ─────────────────────────────────────────────────────────
    print('\n\n' + '=' * 72)
    print('SUMMARY TABLE')
    print('=' * 72)
    hdr = f'  {"Frame":<24}  {"BRR":>6}  {"Errors":>7}  {"MaxRun":>7}  {"AnyErrGrp":>10}  {"MajCorrGrp":>11}  {"SpatialSc":>10}'
    print(hdr)
    print('-' * 80)
    for s in frame_summaries:
        print(f'  {s["frame"]:<24}  {s["brr"]:>5.1f}%  {s["n_errors"]:>6}/{N_BITS}'
              f'  {s["max_consec_run"]:>7}  {s["groups_any_err"]:>6}/{N_GROUPS}'
              f'  {s["groups_majority_corrupt"]:>7}/{N_GROUPS}'
              f'  {s["spatial_cluster_score"]:>10.3f}')

    # ── RS sizing recommendation ──────────────────────────────────────────────
    print('\n' + '=' * 72)
    print('REED-SOLOMON SIZING RECOMMENDATION')
    print('=' * 72)

    if not frame_summaries:
        print('  No data.')
    else:
        max_errors        = max(s['n_errors']                   for s in frame_summaries)
        max_run_all       = max(s['max_consec_run']             for s in frame_summaries)
        max_maj_grp       = max(s['groups_majority_corrupt']    for s in frame_summaries)
        mean_spatial      = np.mean([s['spatial_cluster_score'] for s in frame_summaries])
        worst_brr         = min(s['brr']                        for s in frame_summaries)
        sym_errors_needed = max(s['groups_any_err']             for s in frame_summaries)
        worst_frame       = min(frame_summaries, key=lambda s: s['brr'])['frame']

        print(f'\n  Worst-case frame: {worst_frame}')
        print(f'    BRR            : {worst_brr:.1f}%')
        print(f'    Bit errors     : {max_errors}/{N_BITS}')
        print(f'    Symbol errors  : {sym_errors_needed}/{N_GROUPS}  (groups with ≥1 bit error)')
        print(f'    Max consec run : {max_run_all} bits')
        print(f'    Majority-corrupt groups : {max_maj_grp}/{N_GROUPS}')
        print(f'    Mean spatial cluster score : {mean_spatial:.3f}')

        print(f'\n  RS code sizing (shortened over 32 symbols = 256 bits):')
        for t in (2, 4, 6, 8, 12):
            k       = N_GROUPS - 2 * t
            capable = sym_errors_needed <= t
            mark    = '✓ sufficient' if capable else '✗ insufficient'
            if k > 0:
                eff = k / N_GROUPS * 100
                print(f'    RS(32,{k:2d})  t={t:2d}  data={k*8:3d} bits  overhead={2*t*8} bits  '
                      f'efficiency={eff:.0f}%  →  {mark}')

        print(f'\n  Interleaving strategy:')
        is_bursty  = max_run_all >= 4 or max_maj_grp >= 2
        is_spatial = mean_spatial >= 0.25

        if not is_bursty and not is_spatial:
            verdict = (
                'NONE NEEDED — errors are sparse and randomly distributed.\n'
                '  Simple consecutive grouping (bits 0-7 → symbol 0, etc.) is sufficient.'
            )
        elif is_bursty and is_spatial:
            verdict = (
                'STRIDE INTERLEAVING RECOMMENDED.\n'
                '  Errors cluster spatially (codec processes CTUs in raster order).\n'
                '  Map symbol i → bit positions {i, i+32, i+64, …} so each RS codeword\n'
                '  symbol draws from a different spatial region of the frame.\n'
                '  This spreads a localised burst across all symbols rather than\n'
                '  concentrating ≥t+1 errors in one codeword.'
            )
        elif is_bursty and not is_spatial:
            verdict = (
                'PSEUDO-RANDOM PERMUTATION RECOMMENDED.\n'
                '  Errors are bursty but not spatially predictable.\n'
                '  A seeded Fisher-Yates shuffle of bit→symbol assignment will\n'
                '  statistically break up bursts better than fixed stride.'
            )
        else:
            verdict = (
                'LIGHT STRIDE INTERLEAVING SUFFICIENT.\n'
                '  Errors show mild spatial clustering but are not heavily bursty.\n'
                '  Stride-2 or stride-4 interleave provides adequate protection.'
            )

        print(f'  → {verdict}')

        all_results['recommendation'] = {
            'crf':                CRF,
            'worst_frame':        worst_frame,
            'is_bursty':          is_bursty,
            'is_spatial':         is_spatial,
            'max_errors':         max_errors,
            'max_run_all':        max_run_all,
            'max_maj_groups':     max_maj_grp,
            'sym_errors_needed':  sym_errors_needed,
            'mean_spatial_score': round(float(mean_spatial), 4),
            'verdict':            verdict.split('\n')[0],
        }

    out_path = Path(__file__).parent / 'rs_error_diagnostic_crf32_results.json'
    out_path.write_text(json.dumps(all_results, indent=2,
                                   default=lambda x: int(x) if hasattr(x, 'item') else x))
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    run()
