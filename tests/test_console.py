"""The console must not be able to kill a run over a typographic dash.

Every finding this tool prints is prose, and the prose has em-dashes and arrows in
it. A stream that cannot encode them raises UnicodeEncodeError at the moment of
printing, which for this program means after all the computation is done.

`cli.main` has guarded against that for a while. Its two sibling entry points did
not, and the consequences there are worse than a lost report: the line `gui.main`
would have died on is the one telling the user not to expose the port, printed
after the socket is already listening, and `watch.main` is a daemon whose whole
job is to still be running when it has something to say.
"""

from __future__ import annotations

import io
import importlib

import pytest

from falsify_quant.cli import _utf8_console

# The characters that actually appear in printed strings. The arrow and the infinity
# sign are the interesting ones: unlike the dashes they are not in cp1252 either, so
# a Windows console fails on them too.
AWKWARD = "—–…·×→∞"  # em/en dash, ellipsis, middot, times, arrow, infinity


@pytest.mark.parametrize("module", ["cli", "gui", "watch"])
def test_every_entry_point_makes_the_console_utf8(module, monkeypatch):
    """Three commands ship; the guard belongs to all three or it is decoration."""
    import falsify_quant.cli as cli

    called: list[str] = []
    monkeypatch.setattr(cli, "_utf8_console", lambda: called.append(module))

    mod = importlib.import_module(f"falsify_quant.{module}")
    with pytest.raises(SystemExit):
        mod.main(["--help"])

    assert called == [module], "entry point printed before fixing the console"


def test_it_is_called_before_anything_is_printed(capsys, monkeypatch):
    """Ordering is the whole point -- a guard applied after the first print is none."""
    import falsify_quant.cli as cli

    seen: list[str] = []
    monkeypatch.setattr(cli, "_utf8_console",
                        lambda: seen.append(capsys.readouterr().out))
    with pytest.raises(SystemExit):
        cli.main(["--help"])

    assert seen == [""], f"printed {seen[0]!r} before reconfiguring the stream"


def test_reconfiguring_lets_the_awkward_characters_through():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="ascii", errors="strict")

    with pytest.raises(UnicodeEncodeError):
        stream.write(AWKWARD)
        stream.flush()

    stream.reconfigure(encoding="utf-8", errors="replace")
    stream.write(AWKWARD)
    stream.flush()
    assert AWKWARD.encode("utf-8") in raw.getvalue()


def test_a_stream_that_refuses_to_be_reconfigured_is_survivable():
    """Reconfigure is best-effort. Failing to improve the console is not fatal;
    raising while trying to would be exactly the crash this exists to prevent."""

    class Stubborn:
        def reconfigure(self, **kw):
            raise OSError("not a real terminal")

    import sys

    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = Stubborn()          # type: ignore[assignment]
    try:
        _utf8_console()                            # must not raise
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def test_the_printed_strings_are_the_ones_this_covers():
    """If someone reaches for a new symbol, this test is where they find out.

    Not a style rule -- the point is that the set of characters the guard has to
    carry stays known, rather than growing quietly until one of them meets a
    stream that will not take it.
    """
    import pathlib

    used = set()
    for p in pathlib.Path(__file__).resolve().parent.parent.joinpath(
            "falsify_quant").glob("*.py"):
        used |= {c for c in p.read_text(encoding="utf-8") if ord(c) > 127}

    # Spelled by codepoint: a non-breaking space is invisible in source, and the
    # other two are only ever inside HTML the report writes as UTF-8.
    allowed = set(AWKWARD) | {" ", "÷", "ó"}
    unexpected = used - allowed
    assert not unexpected, (
        "new non-ASCII characters in the package: "
        + ", ".join(f"U+{ord(c):04X}" for c in sorted(unexpected))
    )
