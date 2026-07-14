# XLiteOCR — Commercial-Use License Clearance

XLiteOCR is built so it can be **owned and commercialized freely**. Every runtime
dependency is permissive (Apache-2.0 / MIT / BSD / MPL-2.0 / PSF). **No GPL or
AGPL anywhere.** This file records the actual verification performed, not
assumptions.

## Verification method

Three independent checks, all run against the installed `venv` (not just stated
intent — `tests/test_compliance.py` re-runs them as a gate):

1. **Full classifier scan** over every installed distribution for
   `GNU General Public` / `Affero` / `GPLv2/3` / `AGPL` → **0 hits**.
2. **UNKNOWN-license resolution** — packages whose license `pip-licenses` could
   not parse (a metadata-format gap, newer SPDX `License-Expression` fields) were
   resolved from their own metadata classifiers. Every one is MIT / BSD / Apache /
   PSF. List below.
3. **Binary-wheel / bundled-license inspection** — the native libraries that
   wheels ship were inspected directly, since a source-license audit cannot see
   what a precompiled binary bundles.

## Key components

| Component                     | Role                     | License            | Notes                                                         |
| ----------------------------- | ------------------------ | ------------------ | ------------------------------------------------------------- |
| paddleocr (code)              | OCR + structure pipeline | Apache-2.0         | repo LICENSE                                                  |
| PP-OCR / PP-Structure weights | det/rec/layout/table     | Apache-2.0         | Baidu release                                                 |
| paddlepaddle (CPU)            | runtime                  | Apache-2.0         |                                                               |
| opencv wheels (×3)            | paddleocr internals      | Apache-2.0         | transitive via paddleocr 2.x; never imported by XLiteOCR code |
| pypdfium2 + PDFium            | PDF raster               | Apache-2.0 / BSD-3 | **V8-disabled** (verified)                                    |
| vtracer                       | raster→SVG               | MIT                | replaces GPL `potrace`                                        |
| Pillow                        | imaging (our code)       | MIT-CMU (HPND)     | XLiteOCR's own imaging path, with numpy                       |
| numpy / scikit-learn / scipy  | math                     | BSD-3              |                                                               |
| fastapi / pydantic            | API                      | MIT                |                                                               |
| uvicorn / click               | server                   | BSD-3              |                                                               |
| certifi / tqdm                | misc                     | MPL-2.0            | file-level copyleft, NOT viral — commercial-safe              |

## UNKNOWN-license packages, resolved

All resolved to permissive licenses via metadata classifiers:

ImageIO=BSD-2 · RapidFuzz=MIT · anyio=MIT · cffi=MIT · click=BSD-3 ·
cryptography=Apache-2.0/BSD-3 · idna=BSD-3 · joblib=BSD-3 · lazy-loader=BSD-3 ·
narwhals=MIT · networkx=BSD-3 · packaging=Apache/BSD · pycparser=BSD-3 ·
pydantic(+core)=MIT · pyparsing=MIT · scikit-learn=BSD-3 · termcolor=MIT ·
typing-inspection=MIT · typing_extensions=PSF-2.0 · urllib3=MIT.

## Pitfalls explicitly avoided (each would poison commercial use)

- **poppler (GPL)** — NOT used. PDF rasterization is PDFium via pypdfium2 (BSD-3).
- **potrace (GPL)** — NOT used. Raster→SVG is VTracer (MIT).
- **OpenCV wheels — transitive via paddleocr 2.x (correction, 2026-07-13).**
  Earlier revisions of this file claimed OpenCV was not installed; that was
  wrong. paddleocr 2.x installs `opencv-python`, `opencv-contrib-python` and
  `opencv-python-headless` (all Apache-2.0 — the license gate resolves and
  passes them). XLiteOCR's own code never imports `cv2`; our imaging path is
  Pillow + numpy. The historical codec-bundling concern is a provenance caveat
  inside PaddleOCR's stack, not a license-gate failure.
- **dots.mocr weights** — EXCLUDED. Custom non-Apache weights license (unclear
  commercial terms) and GPU-bound. Its feature set is reached via PP-Structure +
  VTracer instead.

## PDFium V8 status

PDFium's optional V8 JavaScript engine would add licensing/footprint complexity.
The installed build is **V8-free**, verified three ways:

- `libpdfium.so` is **5.5 MB** (a V8-enabled build is 80–200+ MB).
- **Zero** `v8::` / `snapshot_blob` / `natives_blob` strings in the binary.
- No bundled `libv8` anywhere in the environment.
- The `FPDFDoc_*JavaScriptAction*` and `IPDF_JSPLATFORM` symbols present are
  public PDFium-header artifacts (read JS-action _metadata_ / form-fill platform),
  **not** the V8 engine.

PDFium's bundled third-party license texts (`LicenseRef-PdfiumThirdParty.txt`,
etc.) contain **0** GPL mentions.

## Re-running the gate

```
venv/bin/python -m pytest tests/test_compliance.py -v
```

The gate fails the build on any GPL/AGPL package or a V8-enabled PDFium build.
