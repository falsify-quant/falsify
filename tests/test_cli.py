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


# --------------------------------------------------------------------------------------
# Reading price files from the places people actually get them.
#
# Each of these was rejected outright until 2026-08-01. Losing somebody at "your
# header is spelled wrong" wastes the one moment they were willing to try the tool.
# --------------------------------------------------------------------------------------

import csv as _csv

import pytest

from falsify_quant.cli import CLOSE_NAMES, _load_csv, _number


def _write(tmp_path, name, header, rows):
    p = tmp_path / name
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return p


@pytest.mark.parametrize("column", ["Close", "Adj Close", "Close/Last", "close",
                                    "PRICE", "Settle", "px_last"])
def test_close_column_is_found_whatever_the_source_calls_it(tmp_path, column):
    rows = [[f"2020-01-{i:02d}", 100 + i] for i in range(1, 21)]
    p = _write(tmp_path, "p.csv", ["Date", column], rows)
    bars = _load_csv(p, None)
    assert len(bars.close) == 20
    assert bars.close[0] == 101


def test_yahoo_prefers_plain_close_when_both_are_present(tmp_path):
    rows = [[f"2020-01-{i:02d}", 10 + i, 999] for i in range(1, 11)]
    p = _write(tmp_path, "y.csv", ["Date", "Close", "Adj Close"], rows)
    assert _load_csv(p, None).close[0] == 11        # "close" comes first in CLOSE_NAMES


def test_dollar_signs_and_thousands_separators_are_read(tmp_path):
    """Nasdaq's own export writes prices as $1,234.56."""
    rows = [[f"2020-01-{i:02d}", f"${1000 + i},00" .replace(",00", ".00")]
            for i in range(1, 11)]
    p = _write(tmp_path, "n.csv", ["Date", "Close/Last"], rows)
    bars = _load_csv(p, None)
    assert bars.close[0] == 1001.0


def test_number_parsing_handles_the_common_decorations():
    assert _number("$1,234.56") == pytest.approx(1234.56)
    assert _number(" 42 ") == 42.0
    assert _number("(3.5)") == -3.5          # accounting negative


def test_missing_close_column_lists_what_would_have_worked(tmp_path):
    p = _write(tmp_path, "bad.csv", ["Date", "Volume"], [["2020-01-01", 5]])
    with pytest.raises(SystemExit) as exc:
        _load_csv(p, None)
    msg = str(exc.value)
    assert "recognised" in msg
    assert "adj close" in msg                 # tells them the fix, not just the fault
    assert "Volume" in msg                    # and what it actually saw


def test_unparseable_price_column_says_where_to_look(tmp_path):
    p = _write(tmp_path, "junk.csv", ["Date", "Close"],
               [["2020-01-01", "n/a"], ["2020-01-02", "12"]])
    with pytest.raises(SystemExit) as exc:
        _load_csv(p, None)
    assert "could not read" in str(exc.value)


def test_close_names_are_all_lowercase():
    """Lookup is done on a lowercased header, so an uppercase entry could never match."""
    assert all(n == n.lower() for n in CLOSE_NAMES)
