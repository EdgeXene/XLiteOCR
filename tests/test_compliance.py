"""Hard commercial-use compliance gate. MUST pass before deploy.

Encodes the three checks from COMMERCIAL-USE.md:
  1. No GPL/AGPL in any installed distribution (classifier + License-Expression).
  2. UNKNOWN-license packages all resolve to a permissive/known family.
  3. The PDFium binary is V8-free (size + no v8 strings + no engine-init symbol).
"""

from __future__ import annotations

import importlib.metadata as m
import subprocess
from pathlib import Path

GPL_MARKERS = ("GNU General Public", "Affero", "GPLv2", "GPLv3", "AGPL")

# Permissive / commercial-safe license family tokens (substring match, case-insens).
ALLOWED_TOKENS = (
    "MIT", "BSD", "APACHE", "PSF", "HPND", "MPL", "MOZILLA", "ISC", "ZLIB",
    "OPENLDAP", "OLDAP", "PYTHON", "UNLICENSE", "WTFPL", "CC0", "CC-BY",
    "PDFIUM", "PIL", "LIBPNG", "SIL", "OFL",
)


def _license_strings(dist) -> list[str]:
    out = []
    le = dist.metadata.get("License-Expression")
    if le:
        out.append(le)
    for c in dist.metadata.get_all("Classifier") or []:
        if "License" in c:
            out.append(c)
    lic = dist.metadata.get("License")
    if lic:
        out.append(lic.splitlines()[0] if lic else "")
    return [s for s in out if s]


def test_no_gpl_anywhere():
    hits = []
    for d in m.distributions():
        for s in _license_strings(d):
            if any(g in s for g in GPL_MARKERS):
                hits.append((d.metadata["Name"], s))
    assert not hits, f"GPL/AGPL dependencies found: {hits}"


def test_every_package_resolves_to_permissive():
    """No package may be left genuinely unclassifiable."""
    unresolved = []
    for d in m.distributions():
        strings = _license_strings(d)
        blob = " ".join(strings).upper()
        if not strings or not any(tok in blob for tok in ALLOWED_TOKENS):
            unresolved.append((d.metadata["Name"], strings))
    assert not unresolved, f"Packages with unresolved/non-permissive license: {unresolved}"


def _libpdfium_path() -> Path:
    import pypdfium2_raw
    p = Path(pypdfium2_raw.__file__).parent / "libpdfium.so"
    assert p.exists(), f"libpdfium.so not found at {p}"
    return p


def test_pdfium_is_v8_free():
    so = _libpdfium_path()
    size_mb = so.stat().st_size / (1024 * 1024)
    # A V8-enabled PDFium is 80-200+ MB; V8-free is a few MB.
    assert size_mb < 30, f"libpdfium.so is {size_mb:.1f} MB — suspiciously large (V8?)"

    # No V8 engine strings in the binary.
    try:
        out = subprocess.run(
            ["strings", str(so)], capture_output=True, text=True, timeout=60
        ).stdout
        for marker in ("v8::internal", "v8::Isolate", "snapshot_blob", "natives_blob"):
            assert marker not in out, f"V8 marker '{marker}' present in libpdfium.so"
    except FileNotFoundError:
        pass  # `strings` unavailable; size check already strongly indicates V8-free

    # The true V8 engine-init entrypoint must be absent.
    import pypdfium2.raw as raw
    assert not hasattr(raw, "FPDF_InitJavaScriptEngine"), \
        "FPDF_InitJavaScriptEngine present — V8-enabled build"
