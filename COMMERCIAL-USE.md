# XLiteOCR — Commercial-Use License Clearance

XLiteOCR is built so it can be **owned and commercialized freely**. **No GPL,
AGPL or LGPL anywhere.** The resolved dependency set is permissive apart from
MPL-2.0 (certifi, tqdm), which is weak copyleft and file-scoped, plus
OLDAP-2.8 (lmdb) and PSF-2.0 (typing_extensions). This file records the actual
verification performed, not assumptions.

## Verification method

Three independent checks, all run against the installed environment rather
than stated intent, and re-run as a gate by `tests/test_compliance.py`:

1. **Exact SPDX identifier policy** (`tools/license_policy.py`). Every
   installed distribution is resolved to a set of SPDX identifiers, and every
   identifier must appear on an explicit allow-list. It is default-deny, so a
   prohibited or simply unrecognized license needs no rule of its own, and
   `OR` is treated as strictly as `AND`: a prohibited identifier is refused
   wherever it appears, rather than being satisfied by the other branch.

   This replaced a case-insensitive substring test, which is not a licensing
   question and let five expressions through, among them
   `MIT AND GPL-3.0-only` (it contains "MIT") and `CC-BY-NC-4.0` (it contains
   "CC-BY"). `tests/test_license_regressions.py` keeps all five failing.

2. **Prose scan for copyleft markers** over every installed distribution, kept
   as an independent second look at the raw metadata strings. On its own it is
   not sufficient, and that is the point: an SPDX identifier spells it
   `GPL-3.0-only`, which matches none of the prose markers, so this check and
   check 1 cover different failure modes.

3. **Binary-wheel / bundled-license inspection** — the native libraries that
   wheels ship were inspected directly, since a source-license audit cannot
   see what a precompiled binary bundles.

`THIRD_PARTY_LICENSES.md` is generated from the installed environment by
`tools/gen_third_party_licenses.py`, and `--check` fails the suite when it
drifts. It, not this file, is the authoritative per-package list.

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

## Per-package resolution

Per-package resolution is no longer maintained by hand here. It is generated
into `THIRD_PARTY_LICENSES.md`, with the source of each decision recorded per
row (an SPDX expression, an exact classifier match, or a version-scoped
exception).

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

- `libpdfium.so` is **7.3 MB** (a V8-enabled build is 80-200+ MB). Measured
  against the pinned pypdfium2 5.13.0, native PDFium 153.0.7999.0.
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
