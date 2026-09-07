"""Planted license metadata that the deployment gate must reject."""
from email.message import Message

import pytest

from tests import test_compliance as gate


class Distribution:
    version = "0.0.1"

    def __init__(self, expression):
        self.metadata = Message()
        self.metadata["Name"] = "xliteocr-license-canary"
        self.metadata["License-Expression"] = expression


@pytest.mark.parametrize("expression", [
    "MIT AND GPL-3.0-only",
    "MIT OR GPL-3.0-only",
    "CC-BY-NC-4.0",
    "CC-BY-ND-4.0",
    "LicenseRef-Totally-MIT-Not-Allowed",
])
def test_prohibited_metadata_fails_gate(monkeypatch, expression):
    monkeypatch.setattr(gate.m, "distributions", lambda: [Distribution(expression)])
    with pytest.raises(AssertionError):
        gate.test_every_package_resolves_to_permissive()


def test_mit_positive_control(monkeypatch):
    monkeypatch.setattr(gate.m, "distributions", lambda: [Distribution("MIT")])
    gate.test_no_gpl_anywhere()
    gate.test_every_package_resolves_to_permissive()
