"""Commercial-use license policy for the installed dependency closure.

This replaces a case-insensitive substring test. That test asked whether a
permissive family name appeared *anywhere* in a package's license metadata,
which is not a licensing question. Five things passed it that must not:

    MIT AND GPL-3.0-only                 contains "MIT"
    MIT OR GPL-3.0-only                  contains "MIT"
    CC-BY-NC-4.0                         contains "CC-BY"
    CC-BY-ND-4.0                         contains "CC-BY"
    LicenseRef-Totally-MIT-Not-Allowed   contains "MIT"

None of them was caught by the separate GPL check either, because that check
looks for prose markers ("GNU General Public", "AGPL") and an SPDX identifier
spells it "GPL-3.0-only".

The policy here is default-deny over exact SPDX identifiers:

  1. Prefer `License-Expression`. Canonicalize it with `packaging.licenses`,
     which validates the syntax and normalizes case, then pull out every
     identifier it names.
  2. EVERY identifier must appear in ALLOWED_LICENSES, and every `WITH`
     exception in ALLOWED_EXCEPTIONS. Anything not named is refused, so GPL,
     AGPL, LGPL, NonCommercial, NoDerivatives, private `LicenseRef-` ids and
     plain typos are all refused without needing their own rule.
  3. `OR` is treated exactly like `AND`. General SPDX semantics let a consumer
     pick either side of an `OR`, so `MIT OR GPL-3.0-only` is usable under MIT.
     This repository's rule is stricter on purpose: a prohibited identifier is
     refused wherever it appears, so nothing downstream has to remember which
     branch was chosen. COMMERCIAL-USE.md is the source of that rule.
  4. Only when there is no valid expression do the legacy fields apply, and
     only by EXACT match against the tables below. No substring test survives
     anywhere in this module.

`LEGACY_*` entries are deliberately tedious. Each maps one exact string that a
real installed distribution publishes today to the identifiers it means. A
string that is not in the table does not resolve, and a package that resolves
to nothing fails the gate rather than passing quietly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from packaging.licenses import (
    InvalidLicenseExpression,
    canonicalize_license_expression,
)

# ---------------------------------------------------------------- the policy

# Exact SPDX identifiers permitted in the shipped closure. Adding one is a
# licensing decision, so it is made here, once, by name.
ALLOWED_LICENSES = frozenset({
    "MIT",
    "MIT-CMU",
    "MIT-0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "0BSD",
    "Apache-2.0",
    "MPL-2.0",
    "ISC",
    "Zlib",
    "libpng-2.0",
    "PSF-2.0",
    "Python-2.0",
    "Python-2.0.1",
    "HPND",
    "OLDAP-2.8",
    "Unlicense",
    "CC0-1.0",
    "PostgreSQL",
    "OFL-1.1",
    "BSL-1.0",
})

# `WITH` exception identifiers permitted beside an allowed license.
ALLOWED_EXCEPTIONS = frozenset({
    "LLVM-exception",
    "Classpath-exception-2.0",
})

# Identifiers called out by name purely so the failure message can say WHY,
# rather than the generic "not on the allow-list". Membership here changes no
# decision: default-deny already refuses every one of them.
PROHIBITED_REASONS = {
    "GPL": "strong copyleft, refused by COMMERCIAL-USE.md",
    "AGPL": "network copyleft, refused by COMMERCIAL-USE.md",
    "LGPL": "weak copyleft, not cleared for static linking in this build",
    "-NC-": "NonCommercial, incompatible with commercial use",
    "-ND-": "NoDerivatives, incompatible with shipping a modified build",
    "SSPL": "server-side public license, not open source",
    "BUSL": "business source license, time-delayed and not permissive",
}

# Exact `Classifier:` strings, mapped to what they actually assert. The OSI
# "BSD License" and "Apache Software License" classifiers name a family rather
# than a version, so they map to the set of members this policy accepts. That
# is honest about the ambiguity instead of guessing a version.
LEGACY_CLASSIFIERS = {
    "License :: OSI Approved :: MIT License": {"MIT"},
    "License :: OSI Approved :: BSD License": {"BSD-2-Clause", "BSD-3-Clause"},
    "License :: OSI Approved :: Apache Software License": {"Apache-2.0"},
    "License :: OSI Approved :: ISC License (ISCL)": {"ISC"},
    "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)": {"MPL-2.0"},
    "License :: OSI Approved :: Python Software Foundation License": {"PSF-2.0"},
    "License :: OSI Approved :: zlib/libpng License": {"Zlib"},
    "License :: OSI Approved :: The Unlicense (Unlicense)": {"Unlicense"},
}

# Exact `License:` free-text values published by distributions installed today
# that are not valid SPDX expressions. Checked only after expression parsing
# fails on both `License-Expression` and this same string.
LEGACY_LICENSE_TEXT = {
    "MIT License": {"MIT"},
    "Apache 2.0": {"Apache-2.0"},
    "Apache License 2.0": {"Apache-2.0"},
    "Apache Software License": {"Apache-2.0"},
    "BSD 3-Clause": {"BSD-3-Clause"},
    "3-Clause BSD License": {"BSD-3-Clause"},
    "BSD License": {"BSD-2-Clause", "BSD-3-Clause"},
}

# Distributions whose published metadata cannot express their licensing, cleared
# individually against the file evidence in the wheel. Scoped to a name AND the
# exact versions inspected: a later version re-enters the normal path and fails
# until someone looks at it again. This is not a family-wide waiver.
@dataclass(frozen=True)
class ScopedException:
    versions: frozenset
    licenses: frozenset
    evidence: str


METADATA_EXCEPTIONS = {
    # `License:` reads "BSD-3-Clause, Apache-2.0, dependency licenses", which is
    # a comma list and not a parseable SPDX expression. The wheel ships
    # LICENSES/ with the pypdfium2 BSD-3-Clause text and the bundled PDFium
    # Apache-2.0 text; PDFium's own third-party components are covered there.
    # The V8-free property of the binary is asserted separately by
    # test_pdfium_is_v8_free, which is about capability, not licensing.
    "pypdfium2": ScopedException(
        versions=frozenset({"5.13.0"}),
        licenses=frozenset({"BSD-3-Clause", "Apache-2.0"}),
        evidence="wheel LICENSES/ dir: pypdfium2 BSD-3-Clause + bundled PDFium Apache-2.0",
    ),
}


# ------------------------------------------------------------------ resolving

@dataclass
class Resolution:
    """What a single distribution resolved to, and how."""

    name: str
    version: str | None
    licenses: frozenset = field(default_factory=frozenset)
    exceptions: frozenset = field(default_factory=frozenset)
    source: str = "none"
    ok: bool = False
    reason: str = ""

    def describe(self) -> str:
        ids = ", ".join(sorted(self.licenses)) or "nothing"
        return f"{self.name} {self.version or '?'} [{self.source}] -> {ids}: {self.reason}"


def parse_expression(text):
    """Canonicalize an SPDX expression and split out its identifiers.

    Returns (licenses, exceptions) or None when `text` is not a valid
    expression. Canonicalization normalizes case first, so a lowercase
    `gpl-3.0-only` becomes `GPL-3.0-only` and cannot slip past a comparison.
    """
    if not text or not text.strip():
        return None
    try:
        canonical = canonicalize_license_expression(text.strip())
    except (InvalidLicenseExpression, ValueError, TypeError):
        return None

    licenses, exceptions = set(), set()
    tokens = str(canonical).replace("(", " ").replace(")", " ").split()
    after_with = False
    for token in tokens:
        upper = token.upper()
        if upper in ("AND", "OR"):
            after_with = False
            continue
        if upper == "WITH":
            after_with = True
            continue
        (exceptions if after_with else licenses).add(token)
        after_with = False
    return frozenset(licenses), frozenset(exceptions)


def _why_refused(identifier: str) -> str:
    upper = identifier.upper()
    # Longest marker first, so LGPL-2.1-only is reported as weak copyleft
    # rather than matching the "GPL" substring inside it and being described
    # as strong copyleft. The verdict is the same either way; the reason a
    # human reads should still be the right one.
    for marker in sorted(PROHIBITED_REASONS, key=len, reverse=True):
        if marker in upper:
            return PROHIBITED_REASONS[marker]
    if upper.startswith("LICENSEREF-"):
        return "private LicenseRef identifier, meaning unknown to this policy"
    return "not on the allow-list"


def _judge(resolution: Resolution) -> Resolution:
    """Apply default-deny to an already-resolved set of identifiers."""
    refused = [
        f"{i} ({_why_refused(i)})"
        for i in sorted(resolution.licenses)
        if i not in ALLOWED_LICENSES
    ]
    refused += [
        f"WITH {e} (exception not on the allow-list)"
        for e in sorted(resolution.exceptions)
        if e not in ALLOWED_EXCEPTIONS
    ]
    if refused:
        resolution.ok = False
        resolution.reason = "refused: " + "; ".join(refused)
    else:
        resolution.ok = True
        resolution.reason = "all identifiers permitted"
    return resolution


def resolve(name, version, license_expression, classifiers, license_text):
    """Decide one distribution. Every argument is plain metadata, never a dist.

    Order matters: an explicit, valid expression wins, and the legacy tables are
    consulted only when no expression parses. That prevents a stale classifier
    from overriding a package's own current answer.
    """
    name = (name or "").strip()
    resolution = Resolution(name=name, version=version)

    # 1. The package's own expression field.
    parsed = parse_expression(license_expression)
    if parsed is not None:
        resolution.licenses, resolution.exceptions = parsed
        resolution.source = "License-Expression"
        return _judge(resolution)

    # 2. A `License:` value that happens to be a valid expression, e.g. tqdm's
    #    "MPL-2.0 AND MIT". Parsed, not pattern-matched.
    parsed = parse_expression(license_text)
    if parsed is not None:
        resolution.licenses, resolution.exceptions = parsed
        resolution.source = "License (parsed as SPDX)"
        return _judge(resolution)

    # 3. Exact classifier match. The most specific classifier wins: a bare
    #    "License :: OSI Approved" carries no information and is ignored.
    for classifier in classifiers or []:
        mapped = LEGACY_CLASSIFIERS.get(classifier.strip())
        if mapped:
            resolution.licenses = frozenset(mapped)
            resolution.source = f"Classifier {classifier.strip()!r}"
            return _judge(resolution)

    # 4. Exact legacy free-text match.
    mapped = LEGACY_LICENSE_TEXT.get((license_text or "").strip())
    if mapped:
        resolution.licenses = frozenset(mapped)
        resolution.source = "License (exact legacy string)"
        return _judge(resolution)

    # 5. A named, version-scoped exception backed by inspected file evidence.
    exception = METADATA_EXCEPTIONS.get(name.lower())
    if exception and version in exception.versions:
        resolution.licenses = exception.licenses
        resolution.source = f"scoped exception ({exception.evidence})"
        return _judge(resolution)

    resolution.ok = False
    resolution.source = "none"
    resolution.reason = (
        "no valid SPDX expression, no exact classifier or legacy mapping, and no "
        "version-scoped exception; add a mapping only after reading the license"
    )
    return resolution
