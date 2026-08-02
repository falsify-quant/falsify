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


# --------------------------------------------------------------------------------------
# Dates decide the calendar, and the calendar scales every annualised number.
# A misparsed date column is worse than no date column: it sets bars_per_year to
# something confident and wrong, and nothing downstream can notice.
# --------------------------------------------------------------------------------------

from falsify_quant.cli import _parse_dates


def test_iso_dates():
    ts = _parse_dates(["2020-01-01", "2020-01-02", "2020-01-03"])
    assert ts is not None and len(ts) == 3


def test_us_slash_dates_are_read(tmp_path):
    """Nasdaq's own export writes MM/DD/YYYY, and was previously unreadable."""
    ts = _parse_dates(["01/03/2023", "01/04/2023", "01/05/2023"])
    assert ts is not None
    assert ts[1] - ts[0] == 86400


def test_a_format_that_shuffles_the_order_is_rejected():
    """03/04 then 03/03 parses under %m/%d but goes backwards, so it is not trusted.

    Accepting it would set the bar spacing from a scrambled series.
    """
    assert _parse_dates(["03/04/2024", "03/03/2024"]) is None


def test_duplicate_dates_are_rejected():
    """Repeated timestamps give a nonsense spacing; strictly increasing or nothing."""
    assert _parse_dates(["2020-01-01", "2020-01-01", "2020-01-02"]) is None


def test_unparseable_returns_none_rather_than_a_partial_series():
    assert _parse_dates(["not a date", "2020-01-02"]) is None
    assert _parse_dates(["", ""]) is None


def test_other_common_export_shapes():
    for values in (["2020/01/01", "2020/01/02"],
                   ["01-Jan-2020", "02-Jan-2020"],
                   ["20200101", "20200102"]):
        assert _parse_dates(values) is not None, values


def test_weekday_only_daily_data_infers_a_trading_calendar(tmp_path):
    """End to end: the inferred calendar must land near 252, not 365."""
    import csv as _c
    import datetime as _dt

    d, rows = _dt.date(2023, 1, 3), []
    for i in range(300):
        while d.weekday() >= 5:
            d += _dt.timedelta(days=1)
        rows.append([d.strftime("%m/%d/%Y"), f"${100 + i * 0.5:,.2f}"])
        d += _dt.timedelta(days=1)

    p = tmp_path / "nq.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = _c.writer(fh)
        w.writerow(["Date", "Close/Last"])
        w.writerows(rows)

    bars = _load_csv(p, None)
    assert bars.ts is not None
    assert 240 < bars.inferred_bars_per_year < 280
