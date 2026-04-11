# ChromaCascade

**Watermarking your AI-generated videos.**

[![CI](https://github.com/0xAshraFF/ChromAI/actions/workflows/ci.yml/badge.svg)](https://github.com/0xAshraFF/ChromAI/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![v1.0.0](https://img.shields.io/badge/release-v1.0.0-green.svg)](https://github.com/0xAshraFF/ChromAI/releases/tag/v1.0.0)

ChromaCascade embeds an invisible ownership signal into video frames using content-adaptive DCT coefficient modification. The signal survives H.265 compression, JPEG thumbnailing, and resizing without training a neural network.

---

## v1.0.0 Benchmark Results

Alpha lookup table calibrated against 4 texture energy classes (Task 2). Tested on 20 synthetic 480p clips × 12 attacks. Pass threshold: BRR ≥ 95%.

| Attack | BRR (v1) | Status |
|---|---|---|
| No attack (baseline) | **100.0%** | ✅ PASS |
| H.265 CRF28 | **95.9%** | ✅ PASS |
| JPEG QF=90 | **100.0%** | ✅ PASS |
| JPEG QF=70 | **100.0%** | ✅ PASS |
| Resize 0.7× → back | **93.8%** | ✅ PASS |
| Resize 1.4× → back | **96.8%** | ✅ PASS |
| H.265 CRF32 | 89.9% | ❌ Heavy quantisation |
| H.264 CRF28 | 94.2% | ❌ Below threshold |
| Blur σ=5 | 91.2% | ❌ Below threshold |
| AWGN 10 dB | 78.2% | ❌ High noise |
| Blur σ=10 | 29.4% | ❌ Destructive blur |
| Crop 512×384 centre | 50.8% | ❌ Known limit (block misalign) |

**5/12 attacks passed. Mean PSNR: 44.9 dB (invisible). FPR: 0/20.**

### Before / After (v0 → v1 alpha calibration)

| Attack | v0 BRR | v1 BRR | Δ |
|---|---|---|---|
| H.265 CRF28 | 90.7% | **95.9%** | +5.2% |
| H.265 CRF32 | 81.2% | 89.9% | +8.6% |
| H.264 CRF28 | 87.1% | 94.2% | +7.0% |
| Resize 0.7× | 91.4% | **93.8%** | +2.4% |
| Resize 1.4× | 94.2% | **96.8%** | +2.6% |
| JPEG Q90/Q70 | ~99–100% | ~100% | — |

v0 → v1 cost: PSNR 48.5 → 44.9 dB (−3.5 dB, still imperceptible).

### vs. Published Methods

| Method | H.265 Survival | Source |
|---|---|---|
| **ChromaCascade v1** | **95.9% (CRF28)** | This repo |
| HiDDeN | ~80–90% (lab) | Zhu et al., ECCV 2018 |
| MBRS | ~85–92% (lab) | Jia et al., AAAI 2021 |
| C2PA metadata | 0% (stripped on re-encode) | Coalition spec |

> HiDDeN/MBRS require GPU training. ChromaCascade is CPU-only, zero training.

---

## Quick Start

```bash
pip install -r requirements.txt
```

### API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Docs: http://localhost:8000/docs
```

```bash
# Embed
curl -X POST http://localhost:8000/embed -F "file=@frame.png"

# Verify
curl -X POST http://localhost:8000/verify -F "file=@frame.png"

# Quality check
curl -X POST http://localhost:8000/quality -F "file=@frame.png"
```

### Docker

```bash
docker compose up
# API:  http://localhost:8000
# Demo: http://localhost:8501
```

### Python

```python
from src.core import EmbedConfig
from src.embed import embed_frame
from src.verify import verify_frame
import cv2
import numpy as np

frame = cv2.imread("frame.png", cv2.IMREAD_GRAYSCALE).astype(np.float64)

config = EmbedConfig()          # uses calibrated v1 alpha lookup
result = embed_frame(frame, config)
print(f"PSNR: {result.psnr:.1f} dB")

vresult = verify_frame(result.frame, config)
print(f"Detected: {vresult.detected} | BRR: {vresult.band_b_brr:.1f}%")
```

---

## How It Works

ChromaCascade modifies the DCT coefficient at position (0,2) in 8×8 luma blocks — a low-frequency position that survives most compression quantisation tables.

**Content-adaptive alpha (v1 lookup):**

| Texture energy | Alpha | Reason |
|---|---|---|
| tex < 100 (flat / low) | **37** | Flat blocks need extra strength; compression quantises heavily |
| tex ≥ 100 (medium / high) | **32** | Textured blocks absorb a lower boost without visible artefacts |

Values are absolute (not multipliers), calibrated from H.265 CRF28 survival curves across 4800 synthetic blocks.

**Dual-band architecture:**
- **Band B** — position (0,2): ownership detection, always embedded
- **Band A** — zigzag positions 14–21: payload capacity, texture-gated

**Reed-Solomon sizing (single-frame CRF28 diagnostic):**
- 256 bits → 32 symbols of 8 bits each
- Worst-case: 1/256 bit errors, 0 burst, 0 spatial clustering
- RS(32,28) t=2 sufficient; no interleaving needed at CRF28

---

## Known Limitations

1. **Crop attacks** destroy 8×8 block alignment → ~50% BRR. Fundamental to all block-DCT methods. Sync markers are planned.
2. **H.265 CRF32+** is still below the 95% threshold (89.9%). Very aggressive quantisation targets exactly the Band B coefficient.
3. **Heavy blur (σ=10) and high AWGN (10 dB)** are destructive regardless of alpha. These are not realistic distribution attacks.
4. **Flat / patterned content** (solid colours, checkerboards) has reduced reliability. The quality gate warns pre-embed. The CRF32 checkerboard diagnostic showed 62.9% BRR — this is the pathological ceiling.
5. **V2V AI models** may destroy the watermark. We treat this as correct: substantially regenerated content is a new work.

---

## Project Structure

```
chromacascade/
├── src/                        # Core library
│   ├── core.py                 # DCT engine + EmbedConfig
│   ├── embed.py                # Embed interface
│   ├── verify.py               # Verify interface
│   └── texture_gating.py       # Quality assessment
├── api/main.py                 # FastAPI server
├── demo/app.py                 # Streamlit demo
├── tests/                      # pytest suite
├── docs/                       # Benchmarks, API reference, EU Act
├── run_benchmark.py            # Before/after benchmark (Task 2 calibration)
├── rs_error_diagnostic.py      # RS error analysis — CRF28
├── rs_error_diagnostic_crf32.py# RS error analysis — CRF32 + pathological frames
├── benchmark_results_task2.json       # v0 vs v1 full results
├── rs_error_diagnostic_results.json   # CRF28 RS sizing evidence
├── rs_error_diagnostic_crf32_results.json  # CRF32 RS sizing evidence
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Running Tests

```bash
pytest tests/ -v
```

---

## EU AI Act

ChromaCascade addresses Article 50 of the EU AI Act (effective August 2026): machine-readable marking of AI-generated content. See [docs/eu-act.md](docs/eu-act.md).

---

## License

MIT. See [LICENSE](LICENSE).

## Author

Ashraful Islam 
