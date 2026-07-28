"""The demo is the first thing most people will run, and the only thing many will run.

It is also the one command whose *output* is the claim. If the three cases ever stopped
separating -- if noise started passing, or the leaky rule stopped being caught -- the
demo would still print three numbers and look entirely healthy while advertising the
opposite of what the library does. So the assertions here are on the verdicts, not on
the exit code.

It must also stay offline. A demo that fetches prices fails on the machine of the person
least willing to give it a second try.
"""

from __future__ import annotations

import falsify_quant.data
from falsify_quant.cli import main


def _run(capsys, monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("the demo tried to reach the network")

    monkeypatch.setattr(falsify_quant.data, "_get", refuse)
    code = main(["--demo"])
    return code, capsys.readouterr().out


def test_the_demo_separates_an_edge_from_noise_from_a_bug(capsys, monkeypatch):
    code, out = _run(capsys, monkeypatch)

    assert code == 0
    assert "SURVIVED" in out, "a genuine edge must survive, or the tool only ever says no"
    assert "NO EDGE FOUND" in out, "noise must die"
    assert "BROKEN" in out, "one-bar lookahead must be caught outright"


def test_the_demo_needs_no_network_no_files_and_no_arguments(capsys, monkeypatch):
    """Pinned by the monkeypatch above: reaching for prices raises rather than hangs."""
    code, out = _run(capsys, monkeypatch)
    assert code == 0
    assert "No network" in out


def test_the_demo_points_somewhere_after_it_finishes(capsys, monkeypatch):
    """Three numbers and no next step is a screensaver."""
    _, out = _run(capsys, monkeypatch)
    assert "falsify-quant mystrategy.py" in out


def test_help_does_not_advertise_the_old_command_name(capsys):
    """The epilog said `python -m falsify` for a while after the module was renamed."""
    try:
        main(["--help"])
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "python -m falsify " not in out
    assert "falsify-quant --demo" in out
