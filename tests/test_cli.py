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


def test_nothing_anyone_can_copy_still_names_the_old_module():
    """The epilog was fixed and pinned; `strategies/trend.py` kept the dead command.

    `python -m falsify` exits "No module named falsify". Guarding one file and not
    the others left the canonical worked example telling people to run something
    that cannot work, which is worse than the help text doing it -- an example is
    what gets copied. So the check covers everything a reader can copy from.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    sources = [*root.glob("*.md"), *(root / "strategies").glob("*.py"), root / "selftest.py"]

    guilty = [p.name for p in sources
              if p.exists() and "python -m falsify " in p.read_text(encoding="utf-8")]
    assert not guilty, f"these still name the pre-rename module: {guilty}"


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


# --------------------------------------------------------------------------------------
# Importing the strategy file
#
# Mistyping the filename is the most common thing anyone does wrong, and it produced
# eight frames of importlib internals ending in FileNotFoundError -- which reads as the
# tool being broken rather than the command. Every one of these is a user error, so
# every one of them gets a sentence instead of a traceback.
# --------------------------------------------------------------------------------------


def _load(path):
    from falsify_quant.cli import _load_module

    return _load_module(path)


def test_a_missing_strategy_file_is_not_a_traceback(tmp_path):
    with pytest.raises(SystemExit) as exc:
        _load(tmp_path / "typo.py")

    msg = str(exc.value)
    assert "no strategy file at" in msg
    assert "Traceback" not in msg and "importlib" not in msg


def test_a_missing_file_lists_what_is_actually_there(tmp_path):
    """Nine times out of ten the file they meant is sitting next to the one they typed."""
    (tmp_path / "momentum.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "carry.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "__init__.py").write_text("", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        _load(tmp_path / "momentumm.py")

    msg = str(exc.value)
    assert "momentum.py" in msg and "carry.py" in msg
    assert "__init__" not in msg, "dunder files are noise, not candidates"
    assert "--new momentumm.py" in msg


def test_a_missing_file_in_a_missing_directory_still_explains_itself(tmp_path):
    with pytest.raises(SystemExit) as exc:
        _load(tmp_path / "nowhere" / "s.py")

    assert "no strategy file at" in str(exc.value)


def test_a_directory_is_named_as_one(tmp_path):
    """Asserts the whole phrase, not the word.

    `tmp_path` is built from the test's own name, so a bare `"directory" in msg`
    passes on the path alone -- including when the message is a permission error
    from trying to open the directory as a file.
    """
    with pytest.raises(SystemExit) as exc:
        _load(tmp_path)

    assert "is a directory, not a strategy file" in str(exc.value)


def test_a_syntax_error_keeps_pythons_message_and_drops_the_frames(tmp_path):
    """Python names the file, the line and the character. That part is worth keeping."""
    p = tmp_path / "bad.py"
    p.write_text("GRID = {'fast': [5]}\ndef strategy(bars, fast=5)\n    pass\n",
                 encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        _load(p)

    msg = str(exc.value)
    assert "could not be parsed" in msg
    assert "SyntaxError" in msg and "line 2" in msg
    assert "importlib" not in msg and "_bootstrap" not in msg


def test_a_missing_import_names_the_package_and_the_fix(tmp_path):
    """Ported strategies routinely import pandas, which this does not depend on."""
    p = tmp_path / "needs.py"
    p.write_text("import definitely_not_installed_xyz\nGRID = {'fast': [5]}\n"
                 "def strategy(bars, fast=5):\n    return None\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        _load(p)

    msg = str(exc.value)
    assert "definitely_not_installed_xyz" in msg
    assert "pip install definitely_not_installed_xyz" in msg


def test_a_working_file_still_imports(tmp_path):
    """The guards must not have made the ordinary case harder to reach."""
    p = tmp_path / "ok.py"
    p.write_text("GRID = {'fast': [5]}\ndef strategy(bars, fast=5):\n    return None\n",
                 encoding="utf-8")

    mod = _load(p)
    assert callable(mod.strategy) and mod.GRID == {"fast": [5]}


def test_a_missing_csv_is_not_a_traceback(tmp_path):
    """`--csv` and the strategy argument are the two places a filename gets typed."""
    from falsify_quant.cli import _load_csv

    (tmp_path / "prices.csv").write_text("Date,Close\n2020-01-01,1\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        _load_csv(tmp_path / "prices.cvs", None)      # the transposition everyone makes

    msg = str(exc.value)
    assert "no file at" in msg
    assert "prices.csv" in msg, "the file they meant is sitting right next to it"
    assert "Traceback" not in msg


def test_a_directory_passed_as_a_csv_is_named_as_one(tmp_path):
    """Whole phrase again: `tmp_path` carries the word "directory" in it already."""
    from falsify_quant.cli import _load_csv

    with pytest.raises(SystemExit) as exc:
        _load_csv(tmp_path, None)

    assert "is a directory, not a CSV file" in str(exc.value)


@pytest.mark.parametrize("header,rows", [
    ("Date,Open,High,Low,Close,Adj Close,Volume",
     ["2020-01-{:02d},10,11,9,10.5,10.5,1000".format(i) for i in range(1, 29)]),
    ("Date,Close/Last,Volume,Open,High,Low",
     ["01/{:02d}/2020,$1{,}0.50,\"1,000\",$10.00,$11.00,$9.00".replace("{,}", "")
      .format(i) for i in range(1, 29)]),
])
def test_the_two_exports_people_actually_have_both_load(tmp_path, header, rows):
    """Yahoo writes ISO dates, Nasdaq writes MM/DD/YYYY and $1,234.56.

    Verified end to end against generated files in both shapes: same bar count, same
    verdict. Losing somebody at "your column is spelled wrong" wastes the one moment
    they were willing to try it.
    """
    from falsify_quant.cli import _load_csv

    p = tmp_path / "x.csv"
    p.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")

    bars = _load_csv(p, None)
    assert len(bars.close) == 28
    assert bars.close[0] > 0
