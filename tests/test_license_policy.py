"""The license policy itself, tested apart from the installed environment.

tests/test_license_regressions.py drives the policy through the real gate with
planted distributions. This file tests the decision function directly, so a
change in policy shows up as a named behavior change rather than as a package
somewhere in the closure quietly resolving differently.
"""

from __future__ import annotations

import pytest

from tools import license_policy as policy


def judge(expression=None, classifiers=None, license_text=None,
          name="canary", version="0.0.1"):
    return policy.resolve(
        name=name,
        version=version,
        license_expression=expression,
        classifiers=classifiers or [],
        license_text=license_text,
    )


# ------------------------------------------------------------ the five canaries

@pytest.mark.parametrize("expression", [
    "MIT AND GPL-3.0-only",
    "MIT OR GPL-3.0-only",
    "CC-BY-NC-4.0",
    "CC-BY-ND-4.0",
    "LicenseRef-Totally-MIT-Not-Allowed",
])
def test_the_substring_era_acceptances_are_refused(expression):
    """Each of these passed the old substring gate. None may pass now.

    The old gate asked whether "MIT" or "CC-BY" appeared anywhere in the
    string. All five contain one of those and none is acceptable.
    """
    assert not judge(expression).ok


def test_mit_is_still_accepted():
    """MIT is a real license and the positive control. It must not regress."""
    r = judge("MIT")
    assert r.ok
    assert r.licenses == frozenset({"MIT"})
    assert r.source == "License-Expression"


# --------------------------------------------------------------- or is not a choice

def test_or_is_treated_as_strictly_as_and():
    """Deliberately stricter than general SPDX, and the reason is recorded.

    Under ordinary SPDX semantics `MIT OR GPL-3.0-only` is usable under MIT.
    This repository refuses a prohibited identifier wherever it appears so that
    nothing downstream has to track which branch was taken.
    """
    assert not judge("MIT OR GPL-3.0-only").ok
    assert not judge("MIT AND GPL-3.0-only").ok
    # A dual license whose branches are BOTH permitted is fine.
    assert judge("Apache-2.0 OR BSD-2-Clause").ok


def test_a_prohibited_identifier_nested_in_parentheses_is_found():
    assert not judge("MIT AND (Apache-2.0 OR LGPL-2.1-only)").ok


# ------------------------------------------------------------- case and spelling

def test_case_is_normalized_before_comparison():
    """A lowercase spelling must not slip past an identifier comparison."""
    assert not judge("gpl-3.0-only").ok
    assert judge("mit").ok


def test_copyleft_families_are_reported_with_the_right_reason():
    """The verdict and the explanation must both be right.

    "GPL" is a substring of "LGPL", so a naive scan calls LGPL strong copyleft.
    """
    assert "weak copyleft" in judge("LGPL-2.1-only").reason
    assert "strong copyleft" in judge("GPL-3.0-only").reason
    assert "network copyleft" in judge("AGPL-3.0-only").reason


def test_a_real_license_that_is_simply_not_allowed_is_refused_as_such():
    """Default-deny: no rule is needed for a license nobody thought about.

    EPL-2.0 is a valid SPDX identifier and a real open-source license. It is
    not on the allow-list, so it is refused, and the reason says so plainly
    rather than implying the identifier was unrecognizable.
    """
    r = judge("EPL-2.0")
    assert not r.ok
    assert r.source == "License-Expression"
    assert "not on the allow-list" in r.reason


def test_an_identifier_that_is_not_valid_spdx_fails_at_the_parse_step():
    """A different refusal path from the one above, and worth distinguishing.

    A typo or an invented identifier never becomes a set of identifiers at all,
    so it falls through every source and ends with nothing resolved.
    """
    r = judge("SomeLicense-9.9")
    assert not r.ok
    assert r.source == "none"


# ------------------------------------------------------------------- exceptions

def test_a_with_exception_is_checked_separately_from_the_license():
    assert judge("Apache-2.0 WITH LLVM-exception").ok
    r = judge("Apache-2.0 WITH Nonexistent-exception-1.0")
    assert not r.ok
    assert "exception" in r.reason


# --------------------------------------------------------------- source ordering

def test_a_valid_expression_beats_a_stale_classifier():
    """A package's own current answer wins over legacy metadata."""
    r = judge("GPL-3.0-only", classifiers=["License :: OSI Approved :: MIT License"])
    assert not r.ok
    assert r.source == "License-Expression"


def test_a_license_field_that_is_valid_spdx_is_parsed_not_pattern_matched():
    """tqdm publishes `License: MPL-2.0 AND MIT`, which is a real expression."""
    r = judge(license_text="MPL-2.0 AND MIT")
    assert r.ok
    assert r.licenses == frozenset({"MPL-2.0", "MIT"})
    assert "parsed as SPDX" in r.source


def test_a_license_field_that_is_valid_spdx_is_still_subject_to_the_policy():
    assert not judge(license_text="MIT AND GPL-3.0-only").ok


def test_legacy_strings_match_exactly_and_never_by_substring():
    """The whole class of defect being removed."""
    assert judge(license_text="Apache 2.0").ok
    # A string that merely CONTAINS a known one must not resolve.
    r = judge(license_text="Apache 2.0 and also GPL-3.0-only")
    assert not r.ok
    assert r.source == "none"


def test_the_bare_osi_classifier_carries_no_information():
    """pyclipper publishes a bare `License :: OSI Approved` beside a real one."""
    r = judge(classifiers=["License :: OSI Approved"])
    assert not r.ok
    r = judge(classifiers=["License :: OSI Approved",
                           "License :: OSI Approved :: MIT License"])
    assert r.ok


def test_an_ambiguous_family_classifier_maps_to_the_members_it_could_mean():
    """numpy and scipy publish only `BSD License`, with no version."""
    r = judge(classifiers=["License :: OSI Approved :: BSD License"])
    assert r.ok
    assert r.licenses == frozenset({"BSD-2-Clause", "BSD-3-Clause"})


# -------------------------------------------------------------- scoped exception

def test_the_pdfium_exception_is_scoped_to_the_inspected_version():
    """A waiver for a package is not a waiver for its next release."""
    text = "BSD-3-Clause, Apache-2.0, dependency licenses"
    ok = judge(license_text=text, name="pypdfium2", version="5.13.0")
    assert ok.ok
    assert "scoped exception" in ok.source

    later = judge(license_text=text, name="pypdfium2", version="5.14.0")
    assert not later.ok, "a new version must be re-inspected, not inherited"


def test_the_exception_does_not_generalize_to_other_packages():
    r = judge(license_text="BSD-3-Clause, Apache-2.0, dependency licenses",
              name="something-else", version="5.13.0")
    assert not r.ok


# ------------------------------------------------------------------ no resolution

def test_a_package_with_no_usable_metadata_fails_rather_than_passing_quietly():
    r = judge()
    assert not r.ok
    assert r.source == "none"


def test_unparseable_free_text_does_not_resolve():
    """numpy's `License:` is a copyright notice; scikit-image's is `Files: *`."""
    assert not judge(license_text="Copyright (c) 2005-2023, NumPy Developers.").ok
    assert not judge(license_text="Files: *").ok
