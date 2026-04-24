# chroma-ai Development Notes

## Project Summary
chroma-ai is a video watermarking application that embeds invisible ownership signals into video frames. The signal survives H.265 compression, JPEG thumbnailing, and resizing without training a neural network.

**Repository:** https://github.com/0xAshraFF/ChromaAI  
**Status:** MVP deployed on Streamlit Cloud  
**Last Updated:** 2026-04-14

---

## What We've Built

### Core Features
- ✅ Upload video (MP4, MOV, AVI, WebM, MKV)
- ✅ Embed invisible watermark with owner identity (name|email)
- ✅ Test watermark against compression attacks (H.265, JPEG, blur, resize, YouTube pipeline)
- ✅ Download watermarked video as MP4
- ✅ Visual comparison and heatmaps
- ✅ Full benchmark testing across 12 attack types

### Tech Stack
- **Frontend:** Streamlit (MVP, not production-ready)
- **Video Processing:** OpenCV, FFmpeg
- **Watermarking Algorithm:** DCT-domain embedding (Band B at position 0,2)
- **Deployment:** Streamlit Cloud
- **Repository:** GitHub (private or public with branch protection)

---

## Important Changes Made

### Rebranding (Completed)
- Renamed project: `ChromaCascade` → `ChromaAI` → `chroma-ai`
- Updated all docstrings, comments, and UI text
- Renamed folders: `chromacascade demo` → `chromaai_demo`
- Updated `pyproject.toml` name to `chroma-ai`

### Deployment Setup
- Created `streamlit_app.py` wrapper at repo root for easier Streamlit deployment
- Added `packages.txt` with `ffmpeg` to ensure Streamlit Cloud installs it
- Set up branch protection on `main` (1 approval required)

### Stability Fixes
- ✅ Hard-capped frame processing at 30 frames max (prevents blank screen crashes)
- ✅ Extended video encoding timeout from 120s to 300s
- ✅ Added graceful ffmpeg error handling (no more unhandled crashes)
- ✅ Added ffmpeg availability check at app startup
- ✅ Show warning if ffmpeg is missing

### UI Improvements
- Added caption explaining the 30-frame hard cap
- Added warning banner if ffmpeg is unavailable
- Improved download button visibility and placement

---

## Known Issues & Solutions

### Issue: App crashes when processing many frames
**Root Cause:** Video encoding with FFmpeg takes too long, Streamlit times out  
**Solution:** Hard-capped at 30 frames; extended timeout to 300s; graceful error handling  
**Status:** ✅ Fixed

### Issue: Streamlit Cloud shows blank screen
**Root Cause:** FFmpeg not installed in deployment environment  
**Solution:** Added `packages.txt` to tell Streamlit Cloud to install ffmpeg  
**Status:** ✅ Fixed (pending redeploy verification)

### Issue: Missing context on session resume
**Solution:** This NOTES.md file persists in the repo  
**Status:** ✅ Implemented

---

## Deployment Status

### Current Deployment
- **URL:** https://chroma-ai.streamlit.app/ (or custom URL from Streamlit Cloud)
- **Branch:** main
- **Main File:** streamlit_app.py
- **Auto-Deploy:** Enabled (pulls from GitHub automatically)

### To Redeploy Manually
1. Visit Streamlit Cloud dashboard
2. Open your app settings
3. Click "Redeploy" or "Run"

---

## File Structure

```
chromaai/
├── chromaai_demo/          # Main Streamlit demo app
│   ├── app.py              # Main entry point (updated to chroma-ai branding)
│   ├── engine.py           # Core watermarking engine
│   └── requirements.txt     # Demo-specific deps
├── src/                     # Core library (DCT watermarking)
├── api/                     # FastAPI server (optional)
├── docs/                    # Documentation
├── streamlit_app.py         # Root wrapper for Streamlit Cloud
├── requirements.txt         # All Python dependencies
├── packages.txt             # System packages (ffmpeg)
├── pyproject.toml           # Project config
└── NOTES.md                 # This file
```

---

## Key Commands

### Local Development
```bash
# Configure Python environment
python -m venv /Users/ash/Downloads/.venv
source /Users/ash/Downloads/.venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run locally
cd /Users/ash/Downloads/chromaai/chromaai_demo
streamlit run app.py
```

### Git Operations
```bash
# View commit history
git log --oneline

# Pull latest changes
git pull origin main

# Create and push a feature branch
git checkout -b feature-name
git add .
git commit -m "Description"
git push origin feature-name
```

### Deployment
- Push to `main` → Streamlit Cloud auto-redeploys
- Check Streamlit Cloud logs if deployment fails

---

## What to Work On Next

### High Priority
1. **Test ffmpeg fix** — Verify app works on Streamlit Cloud with the new `packages.txt`
2. **Monitor performance** — Check if 30-frame hard cap is sufficient or if more is needed
3. **User testing** — Get feedback on the demo from actual users

### Medium Priority
1. **UI Polish** — Better branding, cleaner layout
2. **File size limits** — Add validation for uploaded video size
3. **Error messaging** — More user-friendly error explanations

### Low Priority (Future Product)
1. **Authentication** — User accounts and login
2. **Storage** — Save watermarked videos for users
3. **Batch processing** — Handle multiple videos
4. **Payment integration** — Monetization (subscription/pay-per-use)
5. **Migration to production stack** — FastAPI backend + React frontend (not Streamlit)

---

## Important Notes for Resume Sessions

**Before restarting:**
1. `git pull origin main` to get latest code
2. Check this NOTES.md for current status
3. Review the git log to see what was changed

**When deploying:**
- Always push to `main` branch
- Wait ~1-2 minutes for Streamlit Cloud to auto-deploy
- If issues persist, manually trigger a redeploy

**Known Streamlit Limitations:**
- Frame cap at 30 prevents crashes but limits throughput
- FFmpeg subprocess calls are CPU-intensive
- Long video processing may timeout even with 300s limit

---

## Contact & Resources

**GitHub:** https://github.com/0xAshraFF/ChromaAI  
**Streamlit Cloud:** https://share.streamlit.io/  
**FFmpeg Docs:** https://ffmpeg.org/  
**Streamlit Docs:** https://docs.streamlit.io/  

---

*Last Updated: 2026-04-14  
Session: Initial development and stabilization*

---

## V2 Item — Keyed Security (COMPLETE)
- src/keyed_crypto.py: ChaCha20 + HKDF, 6 unit tests
- src/core.py: additive key param, backward-compat when key=None
- SECURITY.md: Cayre-Bas threat model documented
- Benchmarks:
  - Channel-noise Δ 0.0% vs baseline (CRF28 95.9%)
  - Informed-adversary BRR retention:
    - zero_known: 58.4% -> 89.0% (+30.7%)
    - averaging_10: 67.1% -> 99.9% (+32.8%)
    - averaging_50: 44.7% -> 92.9% (+48.1%)
    - averaging_200: 32.8% -> 95.1% (+62.2%)
    - wrong_key: 50.0% noise floor (as designed)
- Results: benchmark_results_task2.json -> informed_adversary section
