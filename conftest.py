"""Make the repository root importable during tests.

`strategies/` and `corpus/` are repository content, not installed packages -- the wheel
deliberately ships `falsify/` and nothing else, so a user who installs `falsify-quant`
does not also receive a study harness and a folder of other people's trading rules.

That leaves their tests unable to import them anywhere the repository root is not already
on `sys.path`. Locally it was, because an editable install puts it there, so the whole
suite passed on this machine and failed on every CI runner -- the worst-shaped failure
available, since the local signal says green.

pytest's `prepend` import mode inserts a root `conftest.py`'s directory into `sys.path`
on its own, but relying on that means relying on an import-mode default that a future
`pytest.ini` could change without anyone connecting the two. The insert is explicit.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
