# XLiteOCR

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Self-hosted, CPU-only OCR.** XLiteOCR turns images and PDFs into structured,
machine-readable data: text with bounding boxes, the detected color of each line,
and an optional structured layout with markdown, HTML tables, and figures
vectorized to SVG. It runs entirely on your own infrastructure, needs no GPU, and
is assembled only from permissively licensed components.

Open source under the **Apache License 2.0**. See [COMMERCIAL-USE.md](COMMERCIAL-USE.md)
for the full dependency license clearance and [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)
for the complete manifest.

## What it does

- **Core (fast):** text + bounding boxes + **per-region text color** (the
  dominant glyph-stroke color of each line, as hex / rgb / name).
- **Structured (optional, `?structured=true`):** markdown + typed layout blocks
  (title / text / table / figure) + table HTML + **figures vectorized to SVG**.

## Stack

| Layer            | Component                                  | License         |
| ---------------- | ------------------------------------------ | --------------- |
| Text det+rec     | PaddleOCR (PP-OCRv3 det / PP-OCRv4 en rec) | Apache-2.0      |
| Structure/tables | PP-Structure                               | Apache-2.0      |
| PDF raster       | pypdfium2 (PDFium, V8-free)                | BSD-3           |
| Raster→SVG       | VTracer                                    | MIT             |
| Color/imaging    | Pillow + numpy                             | MIT-CMU / BSD-3 |
| API              | FastAPI + uvicorn                          | MIT / BSD-3     |

> PaddleOCR 2.10.0 `lang='en'` serves PP-OCRv3 detection + PP-OCRv4 English
> recognition (both Apache-2.0). PP-OCRv5 multilingual weights are a config swap;
> the pipeline shape is identical.

## Quick start

Requires Python 3.11+ (CPU only).

```bash
# Download the source: https://edgexene.io/downloads/xliteocr-source.zip
unzip xliteocr-source.zip
cd xliteocr

python -m venv venv
venv/bin/pip install -r requirements.txt

# Run the service (binds to 127.0.0.1:3011)
venv/bin/uvicorn app.server:app --host 127.0.0.1 --port 3011 --workers 1
```

Model weights download from the upstream PaddleOCR distribution on first run.

Optional: a [PM2](https://pm2.keymetrics.io/) process file is included.

```bash
pm2 start ecosystem.config.js
```

## API

```
GET  /health
POST /ocr                  multipart file=<image|pdf>   -> text + boxes + color
POST /ocr?structured=true  multipart file=<image|pdf>   -> + markdown + blocks + figures
```

Example:

```bash
curl -s -F file=@samples/synthetic.png http://127.0.0.1:3011/ocr | jq .
curl -s -F file=@samples/doc.pdf 'http://127.0.0.1:3011/ocr?structured=true' | jq .
```

Response:

```json
{"pages":[{"page":0,
  "lines":[{"text":"RED WORLD","box":[[x,y]],"confidence":0.97,
            "color":{"hex":"#c80101","rgb":[200,1,1],"name":"red"}}],
  "full_text":"...",
  "markdown":"...","blocks":[],"figures":[{"box":[],"type":"figure","svg":"<svg..."}]
}]}
```

## Tests

```bash
venv/bin/python -m pytest tests/ -v
```

`tests/test_compliance.py` is a hard license gate: it fails the build if any
GPL/AGPL component is present or if the bundled PDFium is not V8-free.

## Notes and honest caveats

- **No authentication is built in.** It binds to localhost and is meant to sit
  behind your own reverse proxy / auth. See [SECURITY.md](SECURITY.md).
- **Uploaded documents are processed in memory and never written to disk** by the
  service.
- Formula (LaTeX) recognition is **off by default** (`XLITE_FORMULA=1` to enable);
  its model is large and slow, against the lightweight goal.
- `tools/curate.py` is a data-curation scaffold for future fine-tuning; it does
  not train anything on its own.
- Figure→SVG via VTracer is geometric tracing: excellent on logos, line art, and
  diagrams; approximate on photographs.

## License

Apache License 2.0. Copyright 2026 EdgeXene LLC. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).
