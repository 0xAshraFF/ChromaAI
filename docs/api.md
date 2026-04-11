# ChromaCascade API Reference

## Base URL

```
http://localhost:8000
```

## Endpoints

### `POST /embed`

Embed Band B ownership watermark into an image frame.

**Request:** `multipart/form-data`
- `file`: Image file (PNG/JPG)
- `alpha` (optional): Embedding strength, default 12.0
- `seed` (optional): Payload seed, default 42

**Response:**
```json
{
  "status": "embedded",
  "psnr_db": 47.4,
  "mean_texture": 5690.1,
  "blocks_embedded_band_a": 164,
  "blocks_skipped_band_a": 348,
  "download": "/download/wm_abc12345.png"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/embed \
  -F "file=@frame.png" \
  -F "alpha=12"
```

### `POST /verify`

Verify watermark presence in an image frame.

**Request:** `multipart/form-data`
- `file`: Image file (PNG/JPG)
- `seed` (optional): Must match embed seed, default 42

**Response:**
```json
{
  "detected": true,
  "confidence": "high",
  "band_b_brr": 95.31,
  "band_a_brr": 78.52,
  "interpretation": "Watermark DETECTED. Band B BRR: 95.3% (threshold: 70%). Confidence: high."
}
```

### `POST /quality`

Pre-embed texture assessment.

**Response:**
```json
{
  "quality": "good",
  "mean_texture_energy": 5690.1,
  "embeddable_blocks": 4200,
  "total_blocks": 5700,
  "embed_ratio": 73.7,
  "recommendation": "Standard embedding. Good BRR expected (>85%)."
}
```

### `GET /benchmark`

Returns independently verified benchmark results.

### `GET /docs`

Interactive Swagger/OpenAPI documentation.

## Docker Quick Start

```bash
docker compose up
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```
