"""Known-answer tests. If these do not pass, falsify is not measuring anything.

There are exactly two ways a tool like this can be worthless:

  1. It fails to catch a fake. A strategy fit to pure noise scores well.
  2. It fails to pass a real one. Everything is "overfit", which sounds rigorous and
     is exactly as informative as a stopped clock.

Case A and B check the first failure mode. Case C checks the second, and is the one
that actually constrains the design -- it is easy to build a tool that says no.

Cases D and E do the same pair of checks for the universe selection test: D ships the
two luckiest of twelve noise assets and must be caught, E ships two arbitrary assets
from a universe that genuinely trends and must NOT be called selection bias.
"""

from __future__ import annotations

import sys
import time

import falsify_quant
from falsify_quant.examples import (
    ma_crossover,
    ma_crossover_leaky,
    random_market,
    trending_market,
    zscore_threshold_leaky,
)
from falsify_quant.spec import EQUITY_LIQUID

GRID = {"fast": [3, 5, 8, 12, 20, 30], "slow": [40, 60, 90, 130, 180, 250]}
VALID = lambda p: p["fast"] < p["slow"]  # noqa: E731

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
RED, GREEN, YELLOW = "\033[31m", "\033[32m", "\033[33m"


def show(name: str, verdict, expect: str) -> bool:
    ok = {
        "low": verdict.score < 40 and not verdict.broken,
        "broken": verdict.broken,
        "high": verdict.score >= 60,
    }[expect]

    colour = GREEN if ok else RED
    print(f"\n{BOLD}{name}{RESET}")
    print(f"  {colour}{verdict.score:5.1f}/100  {verdict.label}{RESET}   "
          f"{DIM}(expected {expect}){RESET}")
    for f in verdict.ordered_findings:
        tag = "FATAL" if (f.fatal and f.score <= 0) else f"{f.score:.2f} "
        c = GREEN if f.score >= 0.8 else (YELLOW if f.score >= 0.5 else RED)
        print(f"    {c}{tag:>5}{RESET}  {DIM}{f.title:<20}{RESET} {f.headline[:96]}")
    return ok


def main() -> int:
    results = []
    t0 = time.time()

    # ---------------------------------------------------------------- A: pure noise
    # A moving-average crossover swept over 36 combinations of geometric Brownian motion.
    # The best cell will look tradeable. It cannot be, because there is nothing there.
    bars = random_market(3000, seed=7, symbol="GBM-NOISE")
    v = falsify_quant.run(ma_crossover, bars, EQUITY_LIQUID, GRID, valid=VALID,
                    n_permutations=100, seed=1)
    results.append(show("A. Best of 36 variants fit to pure noise", v, "low"))
    falsify_quant.write_report(v, "reports/noise.html")

    # ------------------------------------------------------------- B: lookahead bug
    # Same idea, smoothed with a centred window so half the kernel is in the future.
    # Run on pure noise: with a leak, even noise produces a spectacular equity curve,
    # which is precisely why this bug is so seductive.
    bars = random_market(3000, seed=11, symbol="CENTRED-MA")
    v = falsify_quant.run(ma_crossover_leaky, bars, EQUITY_LIQUID, GRID, valid=VALID,
                    n_permutations=5, seed=2)
    results.append(show("B. Centred moving average (reads n/2 bars ahead)", v, "broken"))
    falsify_quant.write_report(v, "reports/leak.html")

    # ------------------------------------------------ B2: the subtler normalisation leak
    # Full-sample z-score compared against a fixed threshold. Unlike the crossover case,
    # here the scale factor does not cancel, so it is a genuine leak.
    v = falsify_quant.run(zscore_threshold_leaky, bars, EQUITY_LIQUID,
                    {"lookback": [10, 20, 40], "entry": [0.5, 1.0, 1.5]},
                    n_permutations=5, seed=4)
    results.append(show("B2. Full-sample z-score vs a fixed threshold", v, "broken"))

    # --------------------------------------------------------------- C: real signal
    # A market with genuine slow trends (drift half-life ~100 bars) traded by a strategy
    # that operates at that horizon. Signal and instrument matched, deliberately strong.
    #
    # An earlier version of this case used AR(1) returns, and failed -- correctly. AR(1)
    # momentum lives at lag 1, so a 20/90 crossover cannot capture it no matter how large
    # phi gets. That is a real distinction worth keeping: "no signal" and "wrong
    # instrument for the signal" produce identical-looking equity curves.
    bars = trending_market(5000, half_life=100, seed=3, symbol="TRENDING")
    v = falsify_quant.run(ma_crossover, bars, EQUITY_LIQUID, GRID, valid=VALID,
                    n_permutations=100, seed=3)
    results.append(show("C. Genuine slow trends, traded at the right horizon", v, "high"))
    falsify_quant.write_report(v, "reports/real.html")

    # ------------------------------------------------------- D: universe selection bias
    # Twelve assets of pure noise. Ship the two that happened to score best. The
    # per-asset numbers are honest; the choice is the whole result.
    noise_universe = {f"N{i:02d}": random_market(2500, seed=100 + i, symbol=f"N{i:02d}")
                      for i in range(12)}
    ranked = sorted(
        noise_universe,
        key=lambda s: falsify_quant.sharpe(
            falsify_quant.sweep(ma_crossover, noise_universe[s], EQUITY_LIQUID, GRID,
                          valid=VALID).returns[:, 0].astype(float)),
        reverse=True)
    uv = falsify_quant.run_universe(ma_crossover, noise_universe, EQUITY_LIQUID, GRID,
                              chosen=ranked[:2], params={"fast": 12, "slow": 90},
                              valid=VALID, min_bars=300)
    ok_d = uv.label == "SELECTION BIAS" or uv.score < 40
    results.append(ok_d)
    print(f"\n{BOLD}D. Best 2 of 12 pure-noise assets, shipped{RESET}")
    print(f"  {(GREEN if ok_d else RED)}{uv.score:5.1f}/100  {uv.label}{RESET}   "
          f"{DIM}(expected selection bias){RESET}")
    print(f"    {DIM}selection p {uv.selection_p:.3f} · breadth {uv.breadth:.0%} · "
          f"median Sharpe {uv.median_sharpe:.2f} · basket DSR {uv.universe_dsr:.2f}{RESET}")

    # ------------------------------------------------------- E: a genuinely broad edge
    # Twelve assets that ALL trend. Ship two arbitrary ones -- not the winners. The tool
    # must not cry selection bias just because a universe was involved.
    trend_universe = {f"T{i:02d}": trending_market(2500, half_life=100, seed=200 + i,
                                                   symbol=f"T{i:02d}")
                      for i in range(12)}
    uv2 = falsify_quant.run_universe(ma_crossover, trend_universe, EQUITY_LIQUID, GRID,
                               chosen=["T05", "T06"], params={"fast": 12, "slow": 90},
                               valid=VALID, min_bars=300)
    ok_e = uv2.label == "BROAD"
    results.append(ok_e)
    print(f"\n{BOLD}E. Two arbitrary assets from a universe that all trends{RESET}")
    print(f"  {(GREEN if ok_e else RED)}{uv2.score:5.1f}/100  {uv2.label}{RESET}   "
          f"{DIM}(expected broad){RESET}")
    print(f"    {DIM}selection p {uv2.selection_p:.3f} · breadth {uv2.breadth:.0%} · "
          f"median Sharpe {uv2.median_sharpe:.2f} · basket DSR {uv2.universe_dsr:.2f}{RESET}")

    # --------------------------------------------------- F/G/H: the live monitor
    # The monitor has its own two failure modes: missing a bot that has drifted from
    # its strategy, and crying wolf at one that has not.
    import numpy as np

    from falsify_quant.monitor import Fill, LiveRecord, replay_divergence
    from falsify_quant.spec import Bars

    n = 1500
    base_ts = 1_750_000_000
    src = trending_market(n, half_life=80, seed=31)
    mbars = Bars(close=src.close, ts=np.arange(n) * 3600.0 + base_ts, symbol="MON")

    def hold_above_ma(bars, window=100):
        c = np.asarray(bars.close, float)
        w = int(window)
        s = np.full(len(c), np.nan)
        cs = np.cumsum(np.insert(c, 0, 0.0))
        s[w - 1:] = (cs[w:] - cs[:-w]) / w
        out = (c > s).astype(float)
        out[np.isnan(s)] = np.nan
        return out

    want = np.nan_to_num(hold_above_ma(mbars, 100))
    flips = np.flatnonzero(np.diff(want) != 0) + 1

    def fills_from(skip_exit_at=None):
        out = []
        for t in flips:
            side = "BUY" if want[t] > 0 else "SELL"
            if side == "SELL" and skip_exit_at is not None and t >= skip_exit_at:
                continue  # the exit that never fired
            out.append(Fill(ts=float(mbars.ts[t]), symbol="MON", side=side,
                            qty=1.0, price=float(mbars.close[t]), fee=0.0))
        return out

    faithful = LiveRecord(fills=fills_from(), name="faithful")
    a_f = replay_divergence(hold_above_ma, mbars, faithful, "MON", {"window": 100})
    ok_f = a_f.severity == "ok"
    results.append(ok_f)
    print(f"\n{BOLD}F. Bot that follows its strategy exactly{RESET}")
    print(f"  {(GREEN if ok_f else RED)}{a_f.severity.upper():<6}{RESET} {a_f.headline[:96]}"
          f"   {DIM}(expected ok){RESET}")

    broken = LiveRecord(fills=fills_from(skip_exit_at=int(flips[len(flips) // 2])),
                        name="stuck")
    a_g = replay_divergence(hold_above_ma, mbars, broken, "MON", {"window": 100})
    ok_g = a_g.severity == "alarm" and a_g.detail["extra_bars"] > 0
    results.append(ok_g)
    print(f"{BOLD}G. Bot whose exit silently stopped firing{RESET}")
    print(f"  {(GREEN if ok_g else RED)}{a_g.severity.upper():<6}{RESET} {a_g.headline[:96]}"
          f"   {DIM}(expected alarm){RESET}")

    # A deposit is not a return. Real +2%, plus a 4x funding event.
    steps = np.full(200, 1.0001)
    eq = np.concatenate([[100.0], 100.0 * np.cumprod(steps)])
    eq[120:] *= 4.0  # someone wired money in
    rec = LiveRecord(equity_ts=np.arange(len(eq)) * 300.0 + base_ts, equity=eq)
    rets, flows = rec.trading_returns()
    true_ret = float(np.prod(1.0 + rets) - 1.0)
    ok_h = len(flows) == 1 and abs(true_ret - 0.0202) < 0.005
    results.append(ok_h)
    print(f"{BOLD}H. Equity curve containing a deposit{RESET}")
    print(f"  {(GREEN if ok_h else RED)}{len(flows)} flow removed, trading return "
          f"{true_ret:+.2%}{RESET}   {DIM}(expected 1 flow, ~+2.0%; raw curve says "
          f"{eq[-1]/eq[0]-1:+.0%}){RESET}")

    print(f"\n{DIM}{'-' * 78}{RESET}")
    passed = sum(results)
    colour = GREEN if passed == len(results) else RED
    print(f"{colour}{BOLD}{passed}/{len(results)} known-answer cases correct{RESET}"
          f"  {DIM}({time.time() - t0:.1f}s){RESET}")
    print(f"{DIM}reports written to reports/{RESET}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
