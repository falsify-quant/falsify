"""One version number, in one place.

This is not housekeeping. `attest.py` writes `falsify_version` into the *hashed* body of
every attestation, and `verify()` compares it against the running library. A second copy
of the number that drifts turns a tamper-evidence document into one that misstates the
provenance of its own arithmetic -- and it drifted the first time it was given the
chance: 0.1.1 shipped to PyPI reporting itself as 0.1.0.
"""

from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path

import falsify_quant

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_the_library_reports_the_version_it_was_installed_as():
    assert falsify_quant.__version__ == metadata.version("falsify-quant")


def test_there_is_no_second_copy_of_the_number_in_the_source():
    """A hardcoded release number is how the drift happened. Guard the shape.

    The uninstalled-checkout sentinel is allowed, since it is deliberately not a version
    anyone could mistake for one.
    """
    src = Path(falsify_quant.__file__).read_text(encoding="utf-8")
    hardcoded = [v for v in re.findall(r'__version__\s*=\s*"([^"]+)"', src)
                 if v != "0.0.0+unknown"]
    assert not hardcoded, f"__version__ must be derived, not written down: {hardcoded}"
    assert "_metadata.version(" in src, "the version should come from distribution metadata"


def test_the_installed_version_matches_pyproject():
    """Only meaningful from a checkout; skipped implicitly when there is no pyproject."""
    if not PYPROJECT.exists():  # running against an installed sdist without the file
        return
    declared = re.search(r'^version = "([^"]+)"', PYPROJECT.read_text(encoding="utf-8"),
                         re.M).group(1)
    assert falsify_quant.__version__ == declared, (
        f"pyproject says {declared}, the installed library says "
        f"{falsify_quant.__version__} -- reinstall, or the release will ship a lie"
    )
