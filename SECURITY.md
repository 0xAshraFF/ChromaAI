# chroma-ai Security Model

This document scopes the threat model for the **keyed** variant of the
chroma-ai watermark (`src/keyed_crypto.py` + the `key` parameter on the
four `embed_band_*` / `extract_band_*` functions in `src/core.py`). The
unkeyed variant remains the default; it is calibrated for channel-noise
robustness, not adversarial robustness.

## Threat model (Cayre–Bas, 2005)

Three scenarios from the Cayre–Bas taxonomy, ordered by attacker knowledge:

| Code | Attacker has access to | Keyed variant |
|---|---|---|
| **WOA** (Watermarked-Only Attack) | Watermarked content only. No originals, no key. | **Defended.** |
| **KOA** (Known-Original Attack) | Watermarked + the corresponding clean original. | **Partially defended.** Per-block residuals reveal the keystream's position/sign choices at each block, but not the key itself. Template averaging collapses because each block's residual is keyed. |
| **KMA** (Known-Message / key-leak Attack) | The master key. | **NOT defended.** Full recovery is trivial — treat a leaked key as full compromise. |

## What the keyed variant defends against

- **WOA template estimation.** Averaging residuals across many watermarked
  frames (reverse-SynthID style) no longer converges to a static template;
  each block's embedding position is rotated through
  `{(0,2),(0,3),(1,2),(2,1)}` and its sign XOR-scrambled by a ChaCha20
  keystream. With N=200 paired samples the estimate collapses toward zero.
- **Surgical coefficient zeroing.** A paper-reader attack that zeros
  `DCT(0,2)` and `zigzag[14:22]` on every block destroys the unkeyed
  signal but only partially dents the keyed signal: ~75% of keyed blocks
  embed at rotated positions, and Band A shuffles its 4-of-8 coefficient
  set per block.
- **Block-level template copy/forge.** The keystream binds `(frame_idx, by,
  bx)` into the nonce, so a residual lifted from one block does not survive
  being pasted into another.

## What the keyed variant does NOT defend against

- **KMA / key leak.** Any holder of the master key can embed, extract, or
  forge watermarks. Treat key material like a private signing key: per-creator
  keys, hardware-backed storage, rotation, audit logging.
- **Single-sample forgery.** Open research problem
  (Muller et al. 2024; Jain et al. 2025). A sufficiently powerful adversary
  with one watermarked frame can, in principle, craft frames that verify
  under the same key. The keyed variant raises the bar but does not close
  this gap.
- **V2V model regeneration.** If a generative model substantially rewrites
  pixels, the watermark does not survive. We treat this as correct
  behaviour: the output is a derived work. Provenance for such outputs
  belongs to C2PA metadata, not the signal-domain watermark.
- **Heavy channel noise breaking texture-gate agreement.** Band A's texture
  gate is computed on the received frame. Very heavy blur or noise can push
  a block across the `texture_gate=50` threshold on one side only, desyncing
  embed and extract. Under those attacks the unkeyed path already fails for
  the same reason.
- **Timing / side-channel leakage.** The HKDF + ChaCha20 primitives we use
  are from `cryptography`'s hazmat layer; they're constant-time for the
  key-material path but we do not currently defend against cache-timing on
  the block-gating branch. Not a concern for the benchmark; revisit before
  production.

## Key management guidance

- **Per-creator master keys.** A compromised creator key should never taint
  other creators' watermarks.
- **HKDF hierarchy.** The implementation derives one subkey per
  GOP (`GOP_SIZE=30` frames) via HKDF-SHA256 with salt `b"chroma-ai-v1"`
  and info `f"GOP:{gop_idx}"`. Subkeys are cached (`lru_cache(maxsize=2)`),
  so we re-derive once per GOP boundary.
- **Nonce discipline.** The 12-byte ChaCha20 nonce is
  `struct.pack(">III", frame_idx_in_gop, by, bx)`. Uniqueness is guaranteed
  within a GOP as long as `(frame_idx_in_gop, by, bx)` triples are unique,
  which they are by construction. Different GOPs use different subkeys, so
  nonce reuse across GOPs is benign.
- **Buffer hygiene.** Keystream bytes are materialised as `bytearray`, then
  zeroed via `keyed_crypto.zero_buffer` immediately after the per-block
  bytes are consumed. Subkeys live in the `lru_cache` — rotate the master
  key often enough that this cache turnover is acceptable for your threat
  model, or clear the cache (`_cached_subkey.cache_clear()`) on key rotation.
- **Do not log keys, nonces, or raw keystream.** No module in this repo
  logs key material; verify this invariant if you fork.

## Known caveats of the current implementation

- **Entropy trade-off in Band A.** The keyed path uses 4 of the 8 Band-A
  coefficients per block (Fisher-Yates shuffle, take first 4). This halves
  redundancy vs. the unkeyed 8, which is acceptable because the keyed layer
  targets adversarial robustness, not channel noise — but it means keyed
  Band A is slightly more brittle to heavy compression than unkeyed
  Band A. Use RS(32,28) on the payload as the ECC buffer (see
  `rs_codec.py`'s ordering invariant).
- **Texture-gate divergence under heavy blur.** See above.
- **Benchmark demo key.** `run_benchmark.py` hardcodes a SHA-256 demo key
  for reproducibility. **Do not ship code that reuses this constant.**
