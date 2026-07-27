"""Command line entry point.

    python -m falsify mystrategy.py --symbol BTC-USD --market crypto-perp
    python -m falsify mystrategy.py --symbol AAPL   --market equity
    python -m falsify mystrategy.py --csv prices.csv --market crypto-spot

Your strategy file needs two names at module level:

    def strategy(bars, fast=10, slow=50):   # bars in, target weights out
        ...

    GRID = {"fast": [5, 10, 20], "slow": [50, 100, 200]}

and may optionally define `valid(params) -> bool` to skip nonsensical combinations.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import run, write_report
from .harness import grid_size
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
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)

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


def _load_csv(path: Path, symbol: str | None) -> Bars:
    """Read a CSV with a close column, and optionally date/open/high/low/volume."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path} is empty")

    cols = {k.lower().strip(): k for k in rows[0]}
    close_key = next((cols[k] for k in ("close", "c", "price", "last") if k in cols), None)
    if close_key is None:
        raise SystemExit(f"{path} has no close/price column (found: {list(rows[0])})")

    def col(*names):
        key = next((cols[n] for n in names if n in cols), None)
        if key is None:
            return None
        try:
            return np.array([float(r[key]) for r in rows])
        except (ValueError, TypeError):
            return None

    ts = None
    for name in ("date", "time", "timestamp", "datetime"):
        if name in cols:
            key = cols[name]
            try:
                ts = np.array([float(r[key]) for r in rows])
            except ValueError:
                parsed = []
                for r in rows:
                    v = str(r[key]).strip().replace("Z", "+00:00")
                    try:
                        d = datetime.fromisoformat(v)
                    except ValueError:
                        parsed = []
                        break
                    if d.tzinfo is None:
                        d = d.replace(tzinfo=timezone.utc)
                    parsed.append(d.timestamp())
                ts = np.array(parsed) if parsed else None
            break

    return Bars(
        close=np.array([float(r[close_key]) for r in rows]),
        open=col("open", "o"), high=col("high", "h"), low=col("low", "l"),
        volume=col("volume", "vol", "v"), ts=ts,
        symbol=symbol or path.stem,
    )


def main(argv: list[str] | None = None) -> int:
    _utf8_console()
    p = argparse.ArgumentParser(
        prog="falsify",
        description="Try to prove a trading strategy is nothing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("strategy", type=Path, help="python file defining strategy() and GRID")
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

    mod = _load_module(args.strategy)
    spec = PRESETS[args.market]

    if args.csv:
        bars = _load_csv(args.csv, args.symbol)
    elif args.symbol:
        from .data import GRANULARITY, load
        print(f"{DIM}fetching {args.symbol}…{RESET}", file=sys.stderr)
        bars = load(args.symbol, asset_class=spec.asset_class,
                    interval=args.interval, bars=args.bars)
        if spec.asset_class == "crypto" and args.interval in GRANULARITY:
            # The preset is written for hourly bars. Retime it to whatever was asked
            # for, so --interval does not silently invalidate every annualised number.
            spec = spec.at_bars_per_year(365.25 * 86400 / GRANULARITY[args.interval])
    else:
        p.error("need --symbol or --csv")

    # A local CSV carries no promise about bar size, so take the calendar from the data.
    if args.csv and (actual := bars.inferred_bars_per_year):
        spec = spec.at_bars_per_year(actual)

    n = grid_size(mod.GRID)
    print(f"{DIM}{len(bars):,} bars of {bars.symbol} · {spec.name} · "
          f"{n:,} parameter combinations{RESET}", file=sys.stderr)

    verdict = run(
        mod.strategy, bars, spec, mod.GRID,
        valid=getattr(mod, "valid", None),
        n_permutations=args.permutations,
        permutation_method=args.null,
        progress=lambda s: print(f"{DIM}  {s}{RESET}", file=sys.stderr),
    )

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

    if not args.no_open:
        import webbrowser
        webbrowser.open(out.resolve().as_uri())

    return 0


if __name__ == "__main__":
    sys.exit(main())
