# EU AI Act Compliance Mapping

## Relevant Provisions

**Article 50 — Transparency obligations for certain AI systems**

The EU AI Act (effective August 2026) requires that AI-generated content be identifiably marked. Specifically:

> Providers of AI systems that generate synthetic audio, image, video or text content shall ensure that the outputs are marked in a machine-readable format and detectable as artificially generated or manipulated.

## How ChromaAI Addresses Art. 50

| Requirement | ChromaAI Implementation | Status |
|---|---|---|
| Machine-readable marking | Band B DCT watermark (invisible, signal-level) | ✅ Implemented |
| Detectable as AI-generated | POST /verify endpoint returns detection + confidence | ✅ Implemented |
| Survives distribution | 95.9% BRR through H.265 CRF28 (v1 calibrated alphas) | ✅ Verified |
| Does not degrade content | Mean PSNR 44.9 dB (imperceptible; threshold 40 dB) | ✅ Verified |
| False positive control | 0% FPR on 20-clip clean test suite | ✅ Verified |

## Limitations Relevant to Compliance

ChromaAI is a watermarking tool, not a complete compliance solution. Organizations should note:

1. **Watermark does not survive cropping.** If content is cropped >10%, the watermark may be undetectable. This is a fundamental limitation of block-based DCT methods.

2. **Video-to-video AI models (V2V)** may destroy the watermark entirely. We frame this as a feature: if a V2V model creates substantially new content from your watermarked input, the output is a derived work and the original ownership signal should not persist.

3. **Compliance requires more than watermarking.** Organizations should also implement metadata (C2PA), logging, and disclosure practices. ChromaAI complements these approaches — it does not replace them.

## Recommended Deployment

For EU AI Act compliance, we recommend:

1. Embed ChromaAI watermark at point of generation
2. Also attach C2PA metadata (complementary, not redundant)
3. Log generation events with timestamps
4. Provide /verify endpoint to downstream consumers
5. Document limitations in your compliance filing
