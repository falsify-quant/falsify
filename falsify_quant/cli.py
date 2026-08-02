"""Command line entry point.

Nothing to write and nothing to download -- see what a verdict looks like first:

    falsify-quant --demo

Don't start from a blank file. This writes a working one you can run immediately:

    falsify-quant --new mystrategy.py

Then on your own strategy:

    falsify-quant mystrategy.py --symbol BTC-USD --market crypto-perp
    falsify-quant mystrategy.py --symbol AAPL   --market equity
    falsify-quant mystrategy.py --csv prices.csv --market crypto-spot

Add `--attest` to write a tamper-evident copy of the verdict alongside the report, and
`--verify FILE` to check one somebody sent you:

    falsify-quant mystrategy.py --symbol AAPL --attest
    falsify-quant --verify their-verdict.json

Your strategy file needs two names at module level:

    def strategy(bars, fast=10, slow=50):   # bars in, target weights out
        ...

    GRID = {"fast": [5, 10, 20], "slow": [50, 100, 200]}

and may optionally define `valid(params) -> bool` to skip nonsensical combinations.

`bars` is a Bars object, not a DataFrame: `bars.close` and friends are numpy arrays.
Return one weight per bar, NaN during the warmup. `--new` writes all of that for you.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import run, write_report
from .harness import StrategyError, grid_size
from .spec import PRESETS, Bars

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
RED, GREEN, YELLOW, CYAN = "\033[31m", "\033[32m", "\033[33m", "\033[36m"


def _utf8_console() -> None:
    """Stop a Windows console from killing the run over a typographic dash.

    The findings are written in prose and contain em-dashes and arrows. A default
    Windows terminal is cp1252, so printing them raises UnicodeEncodeError and takes
    down the whole run *after* all the computation is done -- the worst possible time.
    Reconfigure to UTF-8, and fall back to replacing unencodable characters if the
    stream will not take it.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def _load_module(path: Path):
    """Import the user's strategy file, and treat every way that fails as a user error.

    Mistyping the filename is the most common thing anyone does wrong here, and it used
    to produce eight frames of importlib internals ending in `FileNotFoundError`. That
    is the first thing a new user sees, it looks like the tool is broken rather than the
    command, and the GUI -- which confines paths to its served root -- had a clean
    message for exactly this case while the CLI did not.
    """
    if path.is_dir():
        raise SystemExit(
            f"{path} is a directory, not a strategy file.\n"
            "Point at the .py file itself."
        )
    if not path.exists():
        near = sorted(p.name for p in path.parent.glob("*.py")) if path.parent.is_dir() else []
        near = [n for n in near if not n.startswith("__")][:8]
        found = f"\nPython files in {path.parent}: {', '.join(near)}" if near else ""
        raise SystemExit(
            f"no strategy file at {path}{found}\n\n"
            f"To start from one that works:  falsify-quant --new {path.name}"
        )

    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod

    try:
        spec.loader.exec_module(mod)
    except SyntaxError as exc:
        # Python's own message names the file, the line and the character, which is
        # everything needed. The frames above it are all importlib and all noise.
        detail = "".join(traceback.format_exception_only(type(exc), exc)).rstrip()
        raise SystemExit(f"{path.name} could not be parsed.\n\n{detail}") from None
    except ModuleNotFoundError as exc:
        # Ported strategies routinely import pandas, which this does not depend on.
        raise SystemExit(
            f"{path.name} imports {exc.name!r}, which is not installed.\n"
            f"Install it into the same environment as falsify:  "
            f"pip install {exc.name}"
        ) from None

    if not hasattr(mod, "strategy"):
        raise SystemExit(
            f"{path.name} has no `strategy` function.\n"
            "Expected:  def strategy(bars, **params) -> np.ndarray"
        )
    if not hasattr(mod, "GRID"):
        raise SystemExit(
            f"{path.name} has no `GRID` dict.\n"
            'Expected:  GRID = {"fast": [5, 10, 20], "slow": [50, 100, 200]}\n'
            "The grid is not optional — without it there is no trial count to deflate by, "
            "and the headline statistic cannot be computed honestly."
        )
    return mod


TEMPLATE = '''"""A strategy for falsify to attack. Runs as-is -- edit it into yours.

THE CONTRACT, which is the whole of it:

    strategy(bars, **params) -> one target weight per bar

  * `bars` is a Bars object, NOT a DataFrame. `bars.close`, `bars.open`,
    `bars.high`, `bars.low`, `bars.volume` are plain numpy arrays.
  * Return ONE weight PER BAR, same length as `bars.close`. 1.0 = fully long,
    0.0 = flat, -1.0 = fully short. Fractions are fine.
  * Use NaN for the warmup. The engine reads NaN as flat. Do not return a
    shorter array to skip the warmup.
  * Use only data up to and including bar i when computing weight i. falsify
    checks this by re-running with future bars withheld, and a mismatch is the
    one finding that voids the whole report.

Execution lag and costs are NOT your job -- the engine applies both. If you
shift your own signal forward a bar you will be charged the lag twice.
"""

import numpy as np


def strategy(bars, fast=20, slow=100):
    """Long while the fast trailing mean is above the slow one, else flat."""
    c = np.asarray(bars.close, dtype=float)
    f = _sma(c, int(fast))
    s = _sma(c, int(slow))

    w = np.where(f > s, 1.0, 0.0)
    w[np.isnan(f) | np.isnan(s)] = np.nan      # warmup -> flat
    return w


def _sma(x, n):
    """Trailing mean. NaN until n values exist -- never backfilled."""
    out = np.full(len(x), np.nan)
    if n < 1 or n > len(x):
        return out
    csum = np.cumsum(np.insert(x, 0, 0.0))
    out[n - 1:] = (csum[n:] - csum[:-n]) / n
    return out


# Every combination here is a lottery ticket the deflation has to charge you for,
# so keep this small and justify each value. Four fast x three slow is twelve
# trials; a hundred is a different and much harder claim to defend.
GRID = {
    "fast": [10, 20, 40],
    "slow": [50, 100, 200],
}


def valid(p):
    """Optional. Combinations rejected here are not counted as trials."""
    return p["fast"] < p["slow"]
'''


def _write_template(path: Path) -> int:
    """`--new`: the blank page is where most people stop, so remove it."""
    if path.exists():
        raise SystemExit(f"{path} already exists -- pick another name, or delete it")
    if path.suffix != ".py":
        path = path.with_suffix(".py")
    try:
        path.write_text(TEMPLATE, encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"could not write {path}: {exc}")
    print(f"{BOLD}wrote {path}{RESET}\n")
    print("It is a working moving-average strategy, so you can score it right now:\n")
    print(f"    {BOLD}falsify-quant {path} --symbol AAPL --market equity{RESET}\n")
    print(f"{DIM}Then edit strategy() into your own idea. The contract is at the "
          f"top of the file.{RESET}")
    return 0


# What the places people actually get price data from call the close column.
# "Close/Last" is Nasdaq's own export, "Adj Close" is Yahoo's, "Settle" is futures
# continuations. Rejecting a file because of its header is a pointless place to lose
# somebody who has done everything else right.
CLOSE_NAMES = ("close", "close/last", "close_last", "closelast", "adj close",
               "adj_close", "adjclose", "adjusted close", "c", "price", "last",
               "last price", "px_last", "settle", "settlement")


def _number(text) -> float:
    """Parse a price cell.

    Nasdaq exports prices as `$123.45` and plenty of sources use thousands
    separators. Both are unambiguous, so refusing them only makes the user
    reformat a file to prove a point.
    """
    s = str(text).strip().replace(",", "").replace("$", "").replace(" ", "")
    if s.startswith("(") and s.endswith(")"):     # accounting negative
        s = "-" + s[1:-1]
    return float(s)


# Tried in order after ISO. %m/%d/%Y is the Nasdaq/US export shape; %d/%m/%Y is the
# rest of the world and is genuinely ambiguous against it, which is why the result is
# only accepted if it comes out strictly increasing. A misparsed date column is worse
# than none at all -- it silently sets the calendar every annualised number is scaled
# by -- so the monotonicity check is the point, not a nicety.
_DATE_FORMATS = ("%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%d-%b-%Y", "%b %d, %Y",
                 "%m/%d/%Y %H:%M", "%Y%m%d")


def _parse_dates(values: list[str]) -> np.ndarray | None:
    """Timestamps from a date column, or None if no format reads them coherently."""
    def finish(parsed: list[float]) -> np.ndarray | None:
        if len(parsed) != len(values):
            return None
        arr = np.array(parsed, dtype=np.float64)
        # Strictly increasing is the guard: an ambiguous format that happens to
        # parse (03/04 as March 4 vs April 3) will shuffle the order, and a
        # shuffled series would silently mis-scale the calendar.
        return arr if len(arr) < 2 or np.all(np.diff(arr) > 0) else None

    iso: list[float] = []
    for v in values:
        text = v.strip().replace("Z", "+00:00")
        try:
            d = datetime.fromisoformat(text)
        except ValueError:
            iso = []
            break
        iso.append(d.replace(tzinfo=d.tzinfo or timezone.utc).timestamp())
    if (out := finish(iso)) is not None:
        return out

    for fmt in _DATE_FORMATS:
        parsed: list[float] = []
        for v in values:
            try:
                d = datetime.strptime(v.strip(), fmt)
            except ValueError:
                parsed = []
                break
            parsed.append(d.replace(tzinfo=timezone.utc).timestamp())
        if (out := finish(parsed)) is not None:
            return out
    return None


def _load_csv(path: Path, symbol: str | None) -> Bars:
    """Read a CSV with a close column, and optionally date/open/high/low/volume."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path} is empty")

    cols = {k.lower().strip(): k for k in rows[0]}
    close_key = next((cols[k] for k in CLOSE_NAMES if k in cols), None)
    if close_key is None:
        raise SystemExit(
            f"{path} has no column this recognises as the close price.\n"
            f"  found      : {', '.join(rows[0])}\n"
            f"  recognised : {', '.join(CLOSE_NAMES)}\n"
            "Rename your price column to `close` and try again."
        )

    def col(*names):
        key = next((cols[n] for n in names if n in cols), None)
        if key is None:
            return None
        try:
            return np.array([_number(r[key]) for r in rows])
        except (ValueError, TypeError):
            return None

    ts = None
    for name in ("date", "time", "timestamp", "datetime"):
        if name in cols:
            key = cols[name]
            try:
                ts = np.array([float(r[key]) for r in rows])
            except ValueError:
                ts = _parse_dates([str(r[key]) for r in rows])
            break

    try:
        close = np.array([_number(r[close_key]) for r in rows])
    except (ValueError, TypeError) as exc:
        raise SystemExit(
            f"{path}: could not read column {close_key!r} as numbers ({exc}).\n"
            "Check for blank rows, a footer line, or a 'null' placeholder."
        )

    return Bars(
        close=close,
        open=col("open", "o"), high=col("high", "h"), low=col("low", "l"),
        volume=col("volume", "vol", "v"), ts=ts,
        symbol=symbol or path.stem,
    )


def _parse_anchor(text: str | None, parser) -> dict | None:
    if not text:
        return None
    kind, sep, ref = text.partition("=")
    if not sep or not ref.strip():
        parser.error(f"--anchor wants KIND=REF, got {text!r} "
                     f"(e.g. git=https://github.com/you/repo/commit/abc123)")
    return {"kind": kind.strip(), "ref": ref.strip()}


def _verify(path: Path) -> int:
    """Check an attestation somebody handed you. Non-zero exit means do not trust it."""
    from .attest import read_attestation, verify

    try:
        att = read_attestation(path)
    except (OSError, ValueError) as exc:
        print(f"{RED}cannot read {path}: {exc}{RESET}", file=sys.stderr)
        return 2

    result = verify(att)
    band = GREEN if result.ok and not result.warnings else (YELLOW if result.ok else RED)
    print(f"{band}{BOLD}{result.summary()}{RESET}")
    print(f"{DIM}{att.label} {att.score:.0f}/100 · "
          f"{att.body.get('subject', {}).get('symbol', '?')} · "
          f"created {att.created_utc}{RESET}")
    print(f"{DIM}sha256 {att.content_hash}{RESET}\n")

    for c in result.checks:
        mark, colour = ("ok", GREEN) if c.ok else (
            ("warn", YELLOW) if c.severity == "warning" else ("FAIL", RED))
        print(f"  {colour}{mark:>4}{RESET}  {BOLD}{c.name}{RESET}")
        print(f"        {c.detail}")

    if result.ok:
        print(f"\n{DIM}Intact means the arithmetic in front of you is the arithmetic that "
              f"was done. It is not a claim that the strategy works, and — without an "
              f"anchor — not a claim about when this was produced.{RESET}")
    return 0 if result.ok else 1


def _demo() -> int:
    """Three verdicts on synthetic data, offline, with nothing to write first.

    One case would prove nothing. A tool that answered "overfit" to everything would
    look identical to this one on a strategy that deserved it, and a tool that agreed
    with everything would look identical on the case that works. Showing all three
    together is the only demonstration that says anything: a real edge has to survive,
    noise has to die, and a bug that reads one bar ahead has to be caught outright.
    """
    from . import examples, run

    grid = {"fast": [5, 10, 20], "slow": [50, 100, 200]}
    cases = [
        ("A genuine edge", "should survive",
         examples.ma_crossover, examples.trending_market(4000, seed=7)),
        ("The same rule on noise", "should find nothing",
         examples.ma_crossover, examples.random_market(4000, seed=7)),
        ("A rule that peeks one bar ahead", "should be caught outright",
         examples.ma_crossover_leaky, examples.trending_market(4000, seed=7)),
    ]

    print(f"\n{BOLD}Three strategies you did not write, on prices that do not exist.{RESET}")
    print(f"{DIM}No network, no data feed, no files. A few seconds.{RESET}\n")

    for title, expectation, strategy, bars in cases:
        print(f"{BOLD}{title}{RESET} {DIM}— {expectation}{RESET}")
        verdict = run(strategy, bars, PRESETS["equity"], grid, n_permutations=50)
        colour = GREEN if verdict.score >= 50 else (YELLOW if verdict.score >= 20 else RED)
        print(f"  {colour}{BOLD}{verdict.score:5.1f}/100  {verdict.label}{RESET}\n")

    print(f"{DIM}The third one is the point. It is the same strategy as the first with a "
          f"single index shifted the wrong way — the kind of mistake that produces a "
          f"beautiful equity curve and no money.{RESET}")
    print(f"\nOn something of your own:\n"
          f"  {BOLD}falsify-quant mystrategy.py --symbol AAPL --market equity{RESET}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    _utf8_console()
    p = argparse.ArgumentParser(
        prog="falsify-quant",
        description="Try to prove a trading strategy is nothing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("strategy", type=Path, nargs="?",
                   help="python file defining strategy() and GRID")
    p.add_argument("--demo", action="store_true",
                   help="score three example strategies offline and exit; needs no "
                        "file, no network and no data")
    p.add_argument("--new", type=Path, metavar="FILE",
                   help="write a working strategy template you can run immediately, "
                        "then edit")
    p.add_argument("--verify", type=Path, metavar="FILE",
                   help="check an attestation and exit; exits non-zero if tampered")
    p.add_argument("--attest", nargs="?", const=True, default=None, metavar="FILE",
                   help="also write a tamper-evident verdict (default: alongside --out)")
    p.add_argument("--anchor", metavar="KIND=REF",
                   help="where the hash was published, e.g. git=https://.../commit/abc")
    p.add_argument("--market", default="equity", choices=sorted(PRESETS),
                   help="cost model preset (default: equity)")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--symbol", help="ticker or product id, e.g. AAPL or BTC-USD")
    src.add_argument("--csv", type=Path, help="local OHLCV csv instead of fetching")
    p.add_argument("--interval", default="1h", help="crypto bar size (default: 1h)")
    p.add_argument("--bars", type=int, default=5000, help="how many bars to load")
    p.add_argument("--permutations", type=int, default=100,
                   help="synthetic histories for the noise test (default: 100)")
    p.add_argument("--null", default="iid", choices=("iid", "block"),
                   help="noise null: iid shuffle, or block resample (stricter)")
    p.add_argument("-o", "--out", type=Path, default=Path("falsify-report.html"))
    p.add_argument("--no-open", action="store_true", help="do not open the report")
    args = p.parse_args(argv)

    if args.demo:
        return _demo()
    if args.verify:
        return _verify(args.verify)
    if args.new:
        return _write_template(args.new)
    if args.strategy is None:
        p.error("need a strategy file. Try `--demo` to see a verdict first, or "
                "`--new mystrategy.py` to get a working one to edit")

    mod = _load_module(args.strategy)
    spec = PRESETS[args.market]

    if args.csv:
        bars = _load_csv(args.csv, args.symbol)
    elif args.symbol:
        from .data import GRANULARITY, load
        print(f"{DIM}fetching {args.symbol}…{RESET}", file=sys.stderr)
        try:
            bars = load(args.symbol, asset_class=spec.asset_class,
                        interval=args.interval, bars=args.bars)
        except (RuntimeError, OSError) as exc:
            # A rejected ticker or a dropped connection is not a bug in this program.
            raise SystemExit(
                f"\n{RED}{BOLD}Could not fetch {args.symbol}.{RESET}\n\n{exc}\n\n"
                f"{DIM}--market {args.market} looks for a "
                f"{'crypto product id like BTC-USD' if spec.asset_class == 'crypto' else 'stock ticker like AAPL'}"
                f". Use --csv to score your own file instead.{RESET}\n")
        if spec.asset_class == "crypto" and args.interval in GRANULARITY:
            # The preset is written for hourly bars. Retime it to whatever was asked
            # for, so --interval does not silently invalidate every annualised number.
            spec = spec.at_bars_per_year(365.25 * 86400 / GRANULARITY[args.interval])
    else:
        p.error("need --symbol or --csv")

    # A local CSV carries no promise about bar size, so take the calendar from the data.
    if args.csv and (actual := bars.inferred_bars_per_year):
        spec = spec.at_bars_per_year(actual)
    elif args.csv:
        # No usable date column, so the calendar falls back to the preset's. Every
        # annualised figure -- Sharpe, the deflation benchmark, turnover per year --
        # is scaled by it, so silently guessing wrong inflates or deflates the
        # headline by a constant factor with nothing anywhere looking odd. The
        # calendar check downstream cannot catch this either: with no timestamps
        # there is no bar spacing for it to compare against.
        print(f"\n{YELLOW}{BOLD}warning{RESET}  no readable date column in "
              f"{args.csv.name}, so annualised figures assume "
              f"{spec.bars_per_year:,.0f} bars/year from --market {args.market}.\n"
              f"         If these are not {args.market} bars at that frequency, "
              f"Sharpe and everything derived from it are wrong by a constant "
              f"factor.\n"
              f"         Add a date column, or pick a --market whose calendar "
              f"matches your bars.", file=sys.stderr)

    n = grid_size(mod.GRID)
    print(f"{DIM}{len(bars):,} bars of {bars.symbol} · {spec.name} · "
          f"{n:,} parameter combinations{RESET}", file=sys.stderr)

    try:
        verdict = run(
            mod.strategy, bars, spec, mod.GRID,
            valid=getattr(mod, "valid", None),
            n_permutations=args.permutations,
            permutation_method=args.null,
            progress=lambda s: print(f"{DIM}  {s}{RESET}", file=sys.stderr),
        )
    except StrategyError as exc:
        # A mistake in the user's strategy file is not a crash in this program, and
        # printing a traceback tells them about our call stack instead of their bug.
        raise SystemExit(f"\n{RED}{BOLD}Your strategy could not be run.{RESET}\n\n{exc}\n")
    except ValueError as exc:
        # sweep() raises these for conditions the user chose and can change: too few
        # bars, a grid with no valid combinations, a grid over max_trials. Each
        # already carries an explanation, so show it rather than a stack.
        raise SystemExit(f"\n{RED}{BOLD}Cannot run this.{RESET}\n\n{exc}\n")

    if (warning := verdict.meta.get("sweep_warning")):
        print(f"\n{YELLOW}{BOLD}warning{RESET}  {warning}", file=sys.stderr)

    band = (GREEN if verdict.score >= 60 else YELLOW if verdict.score >= 40 else RED)
    print(f"\n{band}{BOLD}{verdict.score:.0f}/100  {verdict.label}{RESET}")
    print(f"{DIM}{verdict.summary}{RESET}\n")
    for f in verdict.ordered_findings:
        tag = "FATAL" if (f.fatal and f.score <= 0) else f"{f.score:.2f}"
        c = GREEN if f.score >= 0.8 else (YELLOW if f.score >= 0.5 else RED)
        print(f"  {c}{tag:>5}{RESET}  {BOLD}{f.title}{RESET}")
        print(f"         {f.headline}")
    if verdict.advice:
        print(f"\n{BOLD}What to fix, in order{RESET}")
        for i, a in enumerate(verdict.advice, 1):
            print(f"  {CYAN}{i}.{RESET} {a}")

    out = write_report(verdict, args.out)
    print(f"\n{DIM}report: {out.resolve()}{RESET}")

    if args.attest is not None:
        from .attest import ANCHOR_HELP, attest, write_attestation

        dest = Path(args.attest) if args.attest is not True else out.with_suffix(".attest.json")
        att = attest(verdict, strategy_source=args.strategy,
                     anchor=_parse_anchor(args.anchor, p))
        write_attestation(att, dest)
        print(f"{DIM}attestation: {dest.resolve()}{RESET}")
        print(f"{BOLD}sha256 {att.content_hash}{RESET}")
        if not att.anchor:
            print(f"\n{YELLOW}This document dates itself, which proves nothing — you "
                  f"chose the date.{RESET}\n{DIM}{ANCHOR_HELP}{RESET}")

    if not args.no_open:
        import webbrowser
        webbrowser.open(out.resolve().as_uri())

    return 0


if __name__ == "__main__":
    sys.exit(main())
