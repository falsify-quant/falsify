"""The published technical-analysis canon, implemented causally, with citations.

Eighteen strategies that appear in books, courses and vendor pitches, written to do what
their sources actually say rather than what makes them look good. Each carries the
parameters its author named -- 50/200, RSI(14) at 30/70, Bollinger(20, 2) -- because
those are the numbers people trade, and they are the numbers the corpus study scores.

Three rules were followed throughout, and they are the reason this file exists rather
than a pile of one-liners:

**The shipped parameters are the ones under examination, not the best of the grid.**
Scoring the grid's winner would measure how well the search did, which is a different and
much easier question than whether the famous version of the strategy works.

**The grid is the search that produced the famous numbers, as far as it can be
reconstructed.** Brock, Lakonishok and LeBaron tested five moving-average rules; the
literature that followed tested thousands. A grid can only ever be a lower bound on the
true number of trials, so every score here is an *upper* bound on the strategy's merit.
That caveat is not a hedge, it is the main finding: the honest N for anything in this
file is the entire history of technical analysis.

**Long-only where the source is long-only.** Applying a symmetric long/short reading to
a rule its author only ever used to time an equity allocation manufactures a short book
nobody trades and a drawdown profile nobody has seen.

Citations are to the primary source where one exists. Where a rule is folklore with no
identifiable origin, it says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from falsify_quant.indicators import (
    atr,
    bollinger,
    day_of_month,
    days_in_month,
    ema,
    keltner,
    macd,
    realised_vol,
    rolling_max,
    rolling_mean,
    rolling_min,
    rsi,
    shift,
    stochastic,
    williams_r,
)
from falsify_quant.spec import Bars

__all__ = ["Candidate", "CANON", "by_name", "for_cadence"]


# --------------------------------------------------------------------------------------
# Shared machinery
# --------------------------------------------------------------------------------------


def _hold(
    n: int,
    enter_long: np.ndarray,
    exit_long: np.ndarray,
    enter_short: np.ndarray,
    exit_short: np.ndarray,
    warmup: int,
) -> np.ndarray:
    """Run entry and exit events through a position state machine.

    Rules with different entry and exit conditions -- every breakout system, every
    oscillator with a neutral zone -- are path dependent: whether today's reading is an
    exit depends on whether you are in a position, which depends on the entire history.
    They cannot be written as a pointwise function of the indicators, and trying anyway
    produces a rule that exits positions it never entered.

    Entries take priority over exits on the same bar, and an entry in the opposite
    direction flips rather than flattens, which is what the source systems do on a
    reversal signal.

    The loop only visits bars where something happens -- a few hundred out of tens of
    thousands for a breakout rule -- because this runs once per grid point per
    permutation, which is tens of thousands of times per study cell.
    """
    pos = np.full(n, np.nan)
    active = np.flatnonzero(enter_long | exit_long | enter_short | exit_short)
    active = active[active >= warmup]
    if len(active) == 0:
        pos[warmup:] = 0.0
        return pos

    states = np.empty(len(active))
    cur = 0.0
    for j, i in enumerate(active):
        if enter_long[i]:
            cur = 1.0
        elif enter_short[i]:
            cur = -1.0
        elif (cur > 0 and exit_long[i]) or (cur < 0 and exit_short[i]):
            cur = 0.0
        states[j] = cur

    slot = np.searchsorted(active, np.arange(n), side="right") - 1
    pos[warmup:] = np.where(slot[warmup:] >= 0, states[np.clip(slot[warmup:], 0, None)], 0.0)
    return pos


def _flat_when_warm(w: np.ndarray, *guards: np.ndarray) -> np.ndarray:
    """NaN out every bar where any input indicator has not warmed up yet."""
    bad = np.zeros(len(w), dtype=bool)
    for g in guards:
        bad |= ~np.isfinite(g)
    w = np.asarray(w, dtype=np.float64).copy()
    w[bad] = np.nan
    return w


# --------------------------------------------------------------------------------------
# Trend
# --------------------------------------------------------------------------------------


def golden_cross(bars: Bars, fast: float = 50, slow: float = 200) -> np.ndarray:
    """Long while the fast mean is above the slow one, flat otherwise. Long only.

    The most widely reported rule in existence, and long-only by convention -- the
    financial press announces the golden cross as a signal to be invested, never as a
    signal to short. Brock, Lakonishok and LeBaron (1992) tested exactly this family on
    ninety years of the Dow and reported returns that launched a literature.
    """
    c = bars.close
    f, s = rolling_mean(c, int(fast)), rolling_mean(c, int(slow))
    return _flat_when_warm(np.where(f > s, 1.0, 0.0), f, s)


def dual_ma(bars: Bars, fast: float = 20, slow: float = 100) -> np.ndarray:
    """The same crossover traded symmetrically: long above, short below.

    The version that appears in trading courses, and the one whose short leg does most
    of the damage once financing is charged.
    """
    c = bars.close
    f, s = rolling_mean(c, int(fast)), rolling_mean(c, int(slow))
    return _flat_when_warm(np.where(f > s, 1.0, -1.0), f, s)


def price_vs_ma(bars: Bars, n: float = 200) -> np.ndarray:
    """Long while price is above its own trailing mean. Long only.

    Faber (2007), "A Quantitative Approach to Tactical Asset Allocation", used a
    ten-month mean on monthly data -- about two hundred trading days. The single most
    widely implemented tactical overlay in retail asset allocation.
    """
    c = bars.close
    m = rolling_mean(c, int(n))
    return _flat_when_warm(np.where(c > m, 1.0, 0.0), m)


def tsmom(bars: Bars, lookback: float = 252) -> np.ndarray:
    """Long if the trailing return over `lookback` bars is positive, short if negative.

    Moskowitz, Ooi and Pedersen (2012), "Time Series Momentum", Journal of Financial
    Economics. Twelve months is their headline horizon, and unlike most of this file the
    symmetric short leg is genuinely what the paper trades.
    """
    c = np.asarray(bars.close, dtype=np.float64)
    past = shift(c, int(lookback))
    return _flat_when_warm(np.where(c > past, 1.0, -1.0), past)


def triple_ma(bars: Bars, fast: float = 5, mid: float = 20, slow: float = 60) -> np.ndarray:
    """Long only when three means are stacked in order, short when stacked inverted.

    Folklore -- no identifiable origin, universally taught. The stacking requirement makes
    it flat far more often than a two-mean crossover, which is usually sold as a virtue
    and is really just a lower exposure to whatever the underlying does.
    """
    c = bars.close
    f, m, s = rolling_mean(c, int(fast)), rolling_mean(c, int(mid)), rolling_mean(c, int(slow))
    w = np.where((f > m) & (m > s), 1.0, np.where((f < m) & (m < s), -1.0, 0.0))
    return _flat_when_warm(w, f, m, s)


def macd_cross(
    bars: Bars, fast: float = 12, slow: float = 26, signal: float = 9
) -> np.ndarray:
    """Long while the MACD histogram is positive, short while negative.

    Appel (1979). The 12/26/9 defaults are on every chart package on earth, which is
    precisely why they are worth deflating: the number of people who have searched
    around them is not small.
    """
    line, sig, hist = macd(bars.close, int(fast), int(slow), int(signal))
    return _flat_when_warm(np.where(hist > 0, 1.0, -1.0), hist)


def vol_target_trend(
    bars: Bars,
    fast: float = 50,
    slow: float = 200,
    vol_n: float = 20,
    cap: float = 3.0,
    target_n: float = 252,
) -> np.ndarray:
    """A trend signal scaled to constant risk rather than constant notional.

    The managed-futures construction: take the direction from a crossover, then size it
    so each position contributes the same volatility. Harvey et al. (2018), "The Impact
    of Volatility Targeting", is the careful treatment. Included because it is the one
    published refinement that plausibly changes a trend rule's statistics rather than
    just its parameters -- and because the leverage it takes in quiet markets is exactly
    where turnover costs go to hide.

    The target is the asset's own *trailing* average volatility over `target_n` bars, so
    a full position is one unit of the risk it has recently been running and the
    comparison against the unscaled rule is like for like.

    The first draft of this used the median volatility of the whole series, which reads
    the future: it tells bar 300 how turbulent the next decade will be. That version is
    worth describing because it is a leak that mostly cancels -- a constant multiplier
    on every weight scales returns and costs identically and leaves the Sharpe ratio
    untouched -- so it survives every check except the one that looks for it. It only
    bites through the clip, and then only sometimes. Leaks that cancel are still leaks:
    the next person to change the sizing rule inherits a live one.
    """
    c = bars.close
    f, s = rolling_mean(c, int(fast)), rolling_mean(c, int(slow))
    v = realised_vol(c, int(vol_n))
    target = rolling_mean(v, int(target_n))
    with np.errstate(divide="ignore", invalid="ignore"):
        size = np.clip(target / v, 0.0, float(cap))
    return _flat_when_warm(np.sign(f - s) * size, f, s, v, target)


# --------------------------------------------------------------------------------------
# Breakout
# --------------------------------------------------------------------------------------


def donchian(bars: Bars, entry: float = 20, exit: float = 10) -> np.ndarray:
    """Turtle System 1: enter on an N-bar channel breakout, leave on the shorter one.

    Faith (2007), "Way of the Turtle", documents the rules Richard Dennis taught in 1983.
    Twenty-day entry, ten-day exit, both sides.

    Every channel is shifted a bar. Compared against an unshifted channel a breakout can
    never fire, because today's close is inside today's window -- the rule degenerates
    into never trading and reports a flawless zero.
    """
    hi = bars.high if bars.high is not None else bars.close
    lo = bars.low if bars.low is not None else bars.close
    c = np.asarray(bars.close, dtype=np.float64)

    up_in = shift(rolling_max(hi, int(entry)), 1)
    dn_in = shift(rolling_min(lo, int(entry)), 1)
    up_out = shift(rolling_max(hi, int(exit)), 1)
    dn_out = shift(rolling_min(lo, int(exit)), 1)

    warm = int(np.nanmax([int(entry), int(exit)])) + 1
    return _hold(
        len(c),
        enter_long=c > up_in,
        exit_long=c < dn_out,
        enter_short=c < dn_in,
        exit_short=c > up_out,
        warmup=warm,
    )


def bollinger_breakout(bars: Bars, n: float = 20, k: float = 2.0) -> np.ndarray:
    """Long above the upper band, short below the lower one. Volatility breakout.

    Bollinger (2001) describes both this and its exact opposite as valid readings,
    depending on whether the market is "trending" or "ranging" -- a distinction only
    available after the fact. Both readings are in this file. They cannot both be right,
    and the interesting result is what happens when neither is told which regime it is in.
    """
    c = np.asarray(bars.close, dtype=np.float64)
    _, up, lo = bollinger(c, int(n), float(k))
    return _flat_when_warm(np.where(c > up, 1.0, np.where(c < lo, -1.0, 0.0)), up, lo)


def keltner_breakout(bars: Bars, n: float = 20, k: float = 2.0, atr_n: float = 10) -> np.ndarray:
    """The same idea with the channel width set by ATR instead of standard deviation.

    Attributed to Chester Keltner (1960) and popularised in its modern ATR form by Linda
    Raschke. Less prone than Bollinger to the band collapsing during a quiet stretch and
    then firing on the first tick of noise.
    """
    c = np.asarray(bars.close, dtype=np.float64)
    _, up, lo = keltner(bars, int(n), float(k), int(atr_n))
    return _flat_when_warm(np.where(c > up, 1.0, np.where(c < lo, -1.0, 0.0)), up, lo)


def chandelier(bars: Bars, n: float = 22, k: float = 3.0) -> np.ndarray:
    """Chuck LeBeau's chandelier exit, traded as a system.

    Long while the close holds above the highest high of the last n bars less k ATRs;
    short while it holds below the lowest low plus k ATRs. A trailing stop rather than a
    signal, which makes it the purest test of whether stop placement alone constitutes
    an edge.
    """
    c = np.asarray(bars.close, dtype=np.float64)
    hi = bars.high if bars.high is not None else bars.close
    lo = bars.low if bars.low is not None else bars.close
    a = atr(bars, int(n))
    long_stop = shift(rolling_max(hi, int(n)) - float(k) * a, 1)
    short_stop = shift(rolling_min(lo, int(n)) + float(k) * a, 1)

    warm = int(np.argmax(np.isfinite(long_stop) & np.isfinite(short_stop)))
    return _hold(
        len(c),
        enter_long=c > short_stop,
        exit_long=c < long_stop,
        enter_short=c < long_stop,
        exit_short=c > short_stop,
        warmup=max(warm, int(n) + 1),
    )


# --------------------------------------------------------------------------------------
# Mean reversion and oscillators
# --------------------------------------------------------------------------------------


def rsi_reversion(
    bars: Bars, n: float = 14, low: float = 30, high: float = 70
) -> np.ndarray:
    """Buy oversold, sell overbought, close out at the midline.

    Wilder (1978), "New Concepts in Technical Trading Systems". The 30/70 bands are his.
    Exiting at 50 rather than at the opposite band is the standard implementation and is
    the more forgiving of the two -- waiting for the opposite band means holding through
    the entire move against you.
    """
    r = rsi(bars.close, int(n))
    warm = int(n) + 1
    return _hold(
        len(r),
        enter_long=r < float(low),
        exit_long=r > 50.0,
        enter_short=r > float(high),
        exit_short=r < 50.0,
        warmup=warm,
    )


def rsi2_connors(
    bars: Bars, n: float = 2, entry: float = 5, exit: float = 70, trend: float = 200
) -> np.ndarray:
    """Connors' short-term reversal: a very fast RSI, filtered by a slow trend. Long only.

    Connors and Alvarez (2008), "Short Term Trading Strategies That Work". Buy when a
    two-period RSI drops below 5 while the close is above its 200-day mean; sell when the
    RSI recovers past 70. The book reports win rates above 70% on index ETFs.

    Worth knowing what a high win rate is and is not: it is a statement about the shape
    of the return distribution, not its mean. Reversal rules win often and lose large,
    and the arithmetic of that trade-off is what the cost and deflation tests exist to do.
    """
    c = np.asarray(bars.close, dtype=np.float64)
    r = rsi(c, int(n))
    filt = rolling_mean(c, int(trend))
    ok = np.isfinite(filt) & (c > filt)
    warm = int(trend)
    return _hold(
        len(c),
        enter_long=ok & (r < float(entry)),
        exit_long=r > float(exit),
        enter_short=np.zeros(len(c), dtype=bool),
        exit_short=np.zeros(len(c), dtype=bool),
        warmup=warm,
    )


def bollinger_reversion(bars: Bars, n: float = 20, k: float = 2.0) -> np.ndarray:
    """Fade the bands: short the upper, buy the lower, cover at the middle.

    Bollinger (2001), read the other way round from `bollinger_breakout`. Bands at two
    standard deviations of a twenty-day mean are his stated defaults.
    """
    c = np.asarray(bars.close, dtype=np.float64)
    mid, up, lo = bollinger(c, int(n), float(k))
    return _hold(
        len(c),
        enter_long=c < lo,
        exit_long=c > mid,
        enter_short=c > up,
        exit_short=c < mid,
        warmup=int(n),
    )


def stochastic_reversion(
    bars: Bars, n: float = 14, smooth: float = 3, low: float = 20, high: float = 80
) -> np.ndarray:
    """Lane's stochastic, traded as an oversold/overbought oscillator."""
    k, d = stochastic(bars, int(n), int(smooth))
    return _hold(
        len(d),
        enter_long=d < float(low),
        exit_long=d > 50.0,
        enter_short=d > float(high),
        exit_short=d < 50.0,
        warmup=int(n) + int(smooth),
    )


def williams_r_reversion(
    bars: Bars, n: float = 14, low: float = -80, high: float = -20
) -> np.ndarray:
    """Larry Williams' %R (1973), same oversold/overbought reading on a -100..0 scale."""
    r = williams_r(bars, int(n))
    return _hold(
        len(r),
        enter_long=r < float(low),
        exit_long=r > -50.0,
        enter_short=r > float(high),
        exit_short=r < -50.0,
        warmup=int(n) + 1,
    )


def n_down_days(bars: Bars, n: float = 3, trend: float = 200) -> np.ndarray:
    """Buy after n consecutive lower closes while above a trend filter; sell on any up day.

    Connors' simplest published rule, and the one most often reproduced in blog posts.
    `trend=0` disables the filter, which is how it is usually traded despite the book.
    """
    c = np.asarray(bars.close, dtype=np.float64)
    down = np.zeros(len(c), dtype=bool)
    down[1:] = np.diff(c) < 0
    streak = down.copy()
    for k in range(1, int(n)):
        streak &= shift(down.astype(np.float64), k) > 0

    if int(trend) > 0:
        filt = rolling_mean(c, int(trend))
        streak &= np.isfinite(filt) & (c > filt)
        warm = int(trend)
    else:
        warm = int(n) + 1

    up = np.zeros(len(c), dtype=bool)
    up[1:] = np.diff(c) > 0
    return _hold(
        len(c),
        enter_long=streak,
        exit_long=up,
        enter_short=np.zeros(len(c), dtype=bool),
        exit_short=np.zeros(len(c), dtype=bool),
        warmup=warm,
    )


# --------------------------------------------------------------------------------------
# Seasonal
# --------------------------------------------------------------------------------------


def turn_of_month(bars: Bars, before: float = 1, after: float = 3) -> np.ndarray:
    """Long across the month boundary, flat the rest of the time. Long only.

    Ariel (1987) and Lakonishok and Smidt (1988) both found that essentially all of the
    US equity market's positive drift accrued in a handful of days around month end.
    Included because it is a real, documented, published anomaly with a plausible
    mechanism -- pension inflows -- and therefore a useful counterweight to a file
    otherwise full of chart patterns.

    The window is defined on the *calendar*, which is public information months ahead.
    Defining it on the bar sequence instead -- "the last trading day of the month" -- is
    a lookahead, because knowing today is the last trading day requires knowing that
    tomorrow is not. It is a natural way to write it and it inflates the result.
    """
    if bars.ts is None:
        raise ValueError("turn_of_month needs timestamps")
    dom = day_of_month(bars.ts)
    dim = days_in_month(bars.ts)
    inside = (dom > dim - int(before)) | (dom <= int(after))
    return np.where(inside, 1.0, 0.0)


# --------------------------------------------------------------------------------------
# The catalogue
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One published strategy, its shipped parameters, and the search around them."""

    name: str
    title: str
    family: str  # trend | breakout | reversion | seasonal
    source: str
    # What the *source* tested it on, taken from the source and nothing else. This is the
    # rule's home turf, and it is not always available here: `futures` in particular has
    # no matching instrument in this universe, so those rules are never given a fair test.
    #
    #   equity-index   the source tested equity indices or index ETFs
    #   futures        the source tested commodity or financial futures
    #   cross-asset    the source tested a futures panel spanning several asset classes
    #   unstated       folklore, or a source that names no particular market
    #
    # Assigned by reading each citation, not by looking at scores. The labels are listed
    # in FINDINGS.md so the assignment can be disputed on its merits.
    domain: str
    fn: Callable[..., np.ndarray]
    shipped: dict[str, float]
    grid: dict[str, Sequence[float]]
    valid: Callable[[dict], bool] | None = None
    long_only: bool = False
    needs: tuple[str, ...] = ()
    cadences: tuple[str, ...] = ("daily", "hourly")
    notes: str = ""

    @property
    def n_grid(self) -> int:
        total = 1
        for v in self.grid.values():
            total *= len(v)
        if self.valid is None:
            return total
        from itertools import product

        keys = list(self.grid)
        return sum(
            1 for combo in product(*self.grid.values()) if self.valid(dict(zip(keys, combo)))
        )


def _fast_lt_slow(p: dict) -> bool:
    return p["fast"] < p["slow"]


CANON: list[Candidate] = [
    Candidate(
        name="golden-cross",
        title="Golden cross (50/200 simple moving average)",
        family="trend",
        domain="equity-index",
        source="Brock, Lakonishok & LeBaron (1992), Journal of Finance 47(5)",
        fn=golden_cross,
        shipped={"fast": 50, "slow": 200},
        grid={"fast": [1, 5, 10, 20, 50], "slow": [50, 100, 150, 200, 250]},
        valid=_fast_lt_slow,
        long_only=True,
        notes="BLL tested (1,50) (1,150) (5,150) (1,200) (2,200) on the Dow, 1897-1986.",
    ),
    Candidate(
        name="dual-ma",
        title="Dual moving average crossover, long and short",
        family="trend",
        domain="unstated",
        source="Folklore; the symmetric reading of the crossover taught in most courses",
        fn=dual_ma,
        shipped={"fast": 20, "slow": 100},
        grid={"fast": [5, 10, 20, 30, 50], "slow": [60, 100, 150, 200, 250]},
        valid=_fast_lt_slow,
    ),
    Candidate(
        name="price-vs-ma",
        title="Price above its 200-day mean (Faber tactical overlay)",
        family="trend",
        domain="equity-index",
        source="Faber (2007), 'A Quantitative Approach to Tactical Asset Allocation'",
        fn=price_vs_ma,
        shipped={"n": 200},
        grid={"n": [50, 100, 150, 200, 250, 300]},
        long_only=True,
        notes="Faber used a 10-month mean on monthly bars; 200 trading days is the "
              "daily equivalent everyone actually implements.",
    ),
    Candidate(
        name="tsmom",
        title="Time-series momentum, 12-month lookback",
        family="trend",
        domain="cross-asset",
        source="Moskowitz, Ooi & Pedersen (2012), Journal of Financial Economics 104(2)",
        fn=tsmom,
        shipped={"lookback": 252},
        grid={"lookback": [21, 63, 126, 189, 252, 504]},
    ),
    Candidate(
        name="triple-ma",
        title="Triple moving average stack (5/20/60)",
        family="trend",
        domain="unstated",
        source="Folklore; no identifiable primary source",
        fn=triple_ma,
        shipped={"fast": 5, "mid": 20, "slow": 60},
        grid={"fast": [3, 5, 10], "mid": [15, 20, 30], "slow": [50, 60, 100]},
        valid=lambda p: p["fast"] < p["mid"] < p["slow"],
    ),
    Candidate(
        name="macd",
        title="MACD histogram crossover (12/26/9)",
        family="trend",
        domain="unstated",
        source="Appel (1979), 'The Moving Average Convergence-Divergence Method'",
        fn=macd_cross,
        shipped={"fast": 12, "slow": 26, "signal": 9},
        grid={"fast": [5, 8, 12, 19], "slow": [17, 26, 39], "signal": [5, 9, 12]},
        valid=_fast_lt_slow,
    ),
    Candidate(
        name="vol-target-trend",
        title="Volatility-targeted 50/200 trend",
        family="trend",
        domain="equity-index",
        source="Harvey et al. (2018), 'The Impact of Volatility Targeting', JPM 45(1)",
        fn=vol_target_trend,
        shipped={"fast": 50, "slow": 200, "vol_n": 20, "cap": 3.0, "target_n": 252},
        grid={"fast": [20, 50], "slow": [100, 200], "vol_n": [10, 20, 60],
              "cap": [2.0, 3.0], "target_n": [126, 252]},
        valid=_fast_lt_slow,
        notes="target_n is in bars, so on hourly data 252 is about ten days rather than "
              "a year. That is deliberate: the target should track the horizon you trade.",
    ),
    Candidate(
        name="donchian",
        title="Donchian channel breakout, 20-day entry / 10-day exit (Turtle System 1)",
        family="breakout",
        domain="futures",
        source="Faith (2007), 'Way of the Turtle'; rules taught by Richard Dennis, 1983",
        fn=donchian,
        shipped={"entry": 20, "exit": 10},
        grid={"entry": [10, 20, 40, 55], "exit": [5, 10, 20]},
        valid=lambda p: p["exit"] < p["entry"],
        needs=("high", "low"),
        notes="System 2 in the same book uses 55/20, which is why 55 is in the grid.",
    ),
    Candidate(
        name="bollinger-breakout",
        title="Bollinger band breakout (20, 2.0)",
        family="breakout",
        domain="unstated",
        source="Bollinger (2001), 'Bollinger on Bollinger Bands'",
        fn=bollinger_breakout,
        shipped={"n": 20, "k": 2.0},
        grid={"n": [10, 20, 30, 50], "k": [1.5, 2.0, 2.5, 3.0]},
    ),
    Candidate(
        name="keltner-breakout",
        title="Keltner channel breakout (20, 2.0 ATR)",
        family="breakout",
        domain="futures",
        source="Keltner (1960), modern ATR formulation popularised by Linda Raschke",
        fn=keltner_breakout,
        shipped={"n": 20, "k": 2.0, "atr_n": 10},
        grid={"n": [10, 20, 40], "k": [1.5, 2.0, 3.0], "atr_n": [10, 20]},
        needs=("high", "low"),
    ),
    Candidate(
        name="chandelier",
        title="Chandelier exit traded as a system (22-day, 3 ATR)",
        family="breakout",
        domain="futures",
        source="Chuck LeBeau, 'Computer Analysis of the Futures Markets' (1992)",
        fn=chandelier,
        shipped={"n": 22, "k": 3.0},
        grid={"n": [10, 22, 40], "k": [2.0, 3.0, 4.0]},
        needs=("high", "low"),
    ),
    Candidate(
        name="rsi-reversion",
        title="RSI(14) oversold/overbought at 30/70",
        family="reversion",
        domain="futures",
        source="Wilder (1978), 'New Concepts in Technical Trading Systems'",
        fn=rsi_reversion,
        shipped={"n": 14, "low": 30, "high": 70},
        grid={"n": [7, 14, 21], "low": [20, 25, 30], "high": [70, 75, 80]},
        valid=lambda p: p["low"] < p["high"],
    ),
    Candidate(
        name="rsi2-connors",
        title="Connors RSI(2) below 5, filtered by the 200-day mean",
        family="reversion",
        domain="equity-index",
        source="Connors & Alvarez (2008), 'Short Term Trading Strategies That Work'",
        fn=rsi2_connors,
        shipped={"n": 2, "entry": 5, "exit": 70, "trend": 200},
        grid={"n": [2, 3], "entry": [2, 5, 10], "exit": [65, 70, 75], "trend": [100, 200]},
        long_only=True,
    ),
    Candidate(
        name="bollinger-reversion",
        title="Bollinger band fade (20, 2.0)",
        family="reversion",
        domain="unstated",
        source="Bollinger (2001), 'Bollinger on Bollinger Bands'",
        fn=bollinger_reversion,
        shipped={"n": 20, "k": 2.0},
        grid={"n": [10, 20, 30, 50], "k": [1.5, 2.0, 2.5, 3.0]},
        notes="Deliberately the exact opposite reading of bollinger-breakout.",
    ),
    Candidate(
        name="stochastic",
        title="Stochastic oscillator (14, 3) at 20/80",
        family="reversion",
        domain="futures",
        source="George Lane, popularised from the late 1950s",
        fn=stochastic_reversion,
        shipped={"n": 14, "smooth": 3, "low": 20, "high": 80},
        grid={"n": [9, 14, 21], "smooth": [3, 5], "low": [10, 20, 30], "high": [70, 80, 90]},
        valid=lambda p: p["low"] < p["high"],
        needs=("high", "low"),
    ),
    Candidate(
        name="williams-r",
        title="Williams %R(14) at -80/-20",
        family="reversion",
        domain="futures",
        source="Larry Williams (1973)",
        fn=williams_r_reversion,
        shipped={"n": 14, "low": -80, "high": -20},
        grid={"n": [7, 14, 28], "low": [-90, -80, -70], "high": [-30, -20, -10]},
        valid=lambda p: p["low"] < p["high"],
        needs=("high", "low"),
    ),
    Candidate(
        name="n-down-days",
        title="Buy after three down closes above the 200-day mean",
        family="reversion",
        domain="equity-index",
        source="Connors & Alvarez (2008); widely reproduced without the trend filter",
        fn=n_down_days,
        shipped={"n": 3, "trend": 200},
        grid={"n": [2, 3, 4], "trend": [0, 100, 200]},
        long_only=True,
    ),
    Candidate(
        name="turn-of-month",
        title="Turn-of-month seasonal window",
        family="seasonal",
        domain="equity-index",
        source="Ariel (1987), JFE 18(1); Lakonishok & Smidt (1988), RFS 1(4)",
        fn=turn_of_month,
        shipped={"before": 1, "after": 3},
        grid={"before": [1, 2, 3, 4], "after": [1, 2, 3, 4]},
        long_only=True,
        needs=("ts",),
        cadences=("daily",),
        notes="A documented anomaly with a mechanism, included as a counterweight to a "
              "catalogue otherwise made of chart patterns.",
    ),
]


def by_name(name: str) -> Candidate:
    for c in CANON:
        if c.name == name:
            return c
    raise KeyError(f"no strategy named {name!r}; have {[c.name for c in CANON]}")


def for_cadence(cadence: str) -> list[Candidate]:
    """The subset meaningful at a given bar size. Seasonal rules are daily-only."""
    return [c for c in CANON if cadence in c.cadences]
