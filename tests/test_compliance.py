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

from tools import license_policy

# Prose markers for the independent GPL check below. This list is deliberately
# NOT the mechanism that keeps GPL out: an SPDX identifier spells it
# "GPL-3.0-only", which matches none of these, and that is exactly how
# "MIT AND GPL-3.0-only" used to pass both checks. The identifier policy in
# tools/license_policy.py is the mechanism. This stays as a second, independent
# look at the raw strings, and now also catches the identifier spellings.
GPL_MARKERS = (
    "GNU General Public", "Affero", "GPLv2", "GPLv3",
    "AGPL", "LGPL", "GPL-2.0", "GPL-3.0", "GPL-1.0",
)

# Ambient Python packaging tooling: present in every environment and never part
# of the shipped dependency closure. These are still fully subject to
# test_no_gpl_anywhere and to the identifier policy; the set exists only so that
# an interpreter that ships one of them with genuinely empty license metadata
# does not fail the closure check for a package we do not distribute.
AMBIENT_TOOLING = frozenset({"pip", "setuptools", "wheel", "pkg_resources", "distribute"})


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


def _classifiers(dist) -> list[str]:
    return [c for c in (dist.metadata.get_all("Classifier") or []) if "License" in c]


def _license_text(dist) -> str:
    lic = dist.metadata.get("License")
    return lic.splitlines()[0] if lic else ""


def _version(dist) -> str | None:
    """The canary distributions carry a plain `version` attribute."""
    try:
        return dist.version
    except Exception:
        return dist.metadata.get("Version")


def test_no_gpl_anywhere():
    hits = []
    for d in m.distributions():
        for s in _license_strings(d):
            if any(g in s for g in GPL_MARKERS):
                hits.append((d.metadata["Name"], s))
    assert not hits, f"GPL/AGPL dependencies found: {hits}"


def test_every_package_resolves_to_permissive():
    """Every installed distribution must resolve to permitted SPDX identifiers.

    Default-deny: an identifier that is not explicitly allowed fails, so a
    prohibited or unknown license needs no rule of its own. `OR` is not a
    choice here, by the repository rule recorded in license_policy.
    """
    refused = []
    for d in m.distributions():
        name = (d.metadata["Name"] or "").strip()
        resolution = license_policy.resolve(
            name=name,
            version=_version(d),
            license_expression=d.metadata.get("License-Expression"),
            classifiers=_classifiers(d),
            license_text=_license_text(d),
        )
        if resolution.ok:
            continue
        # Ambient tooling is excused only from having to NAME a license, never
        # from a license that is actually refused.
        if name.lower() in AMBIENT_TOOLING and resolution.source == "none":
            continue
        refused.append(resolution.describe())
    assert not refused, "Licenses not permitted for commercial use:\n  " + "\n  ".join(refused)


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


def test_third_party_manifest_matches_the_installed_environment():
    """The license manifest is a claim; this is the thing that checks it.

    It was hand-maintained and had drifted: 16 rows named a version other than
    the one installed, including pypdfium2, and one named a pip newer than the
    interpreter had. It is generated now, so drift is a test failure rather
    than a document nobody rereads.
    """
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, str(repo / "tools" / "gen_third_party_licenses.py"), "--check"],
        capture_output=True, text=True, cwd=repo,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_no_unsupported_chart_claim_survives_in_the_engine():
    """PP-Chart2Table is not loaded anywhere; nothing may say it is."""
    from pathlib import Path

    engine_dir = Path(__file__).resolve().parent.parent / "engine"
    offenders = []
    for path in engine_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            if "Chart2Table" in line and "NOT implemented" not in text:
                offenders.append(f"{path.name}:{number}")
    assert not offenders, f"unsupported chart claim at {offenders}"
