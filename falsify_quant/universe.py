"""Selection bias across assets: the search you ran before you wrote the grid.

Every test in `prosecute.py` takes the asset as given. That is a large blind spot,
because choosing *what to trade* is itself a search, usually a bigger one than the
parameter sweep, and usually undocumented.

The shape of the mistake:

    You try a 200-day trend filter on eighteen crypto pairs. It looks great on two of
    them. You ship those two. The backtest for those two is honest, the parameter grid
    was small, the deflated Sharpe clears -- and the result is still an artifact,
    because the trial count that mattered was eighteen times larger than the one you
    deflated by.

So this module sweeps the whole candidate universe and asks one question: is the
chosen set's performance explainable by picking the best k of N?

That question has an exact answer. Enumerate all C(N, k) subsets, score each, and
count how many do at least as well as the set actually chosen. No asymptotics, no
distributional assumption -- just the complete null.

An honest caveat this test cannot resolve
-----------------------------------------
It measures whether the chosen assets were unusually good. It cannot tell you *why*
they were chosen. BTC and ETH are the two largest and most liquid crypto assets, and
a reasonable person would shortlist them without ever looking at a backtest. If they
rank first and second, that is consistent with selection bias AND with megacaps
genuinely trending better than altcoins. A low p-value here means "justify this
choice from something other than returns", not "you cheated".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Mapping, Sequence

import numpy as np

from .harness import sweep
from .sim import simulate
from .spec import Bars, MarketSpec
from .stats import annualise, deflated_sharpe, sharpe

__all__ = ["AssetResult", "UniverseVerdict", "run_universe", "align_common_window"]


def align_common_window(
    bars_by_symbol: Mapping[str, Bars],
    *,
    min_years: float = 0.0,
    bars_per_year: float = 365.25,
) -> tuple[dict[str, Bars], tuple[float, float], list[str]]:
    """Cut every asset down to the period they all share.

    Without this the comparison is meaningless, and meaningless in a direction that
    flatters whatever has the longest history. An asset listed in 2022 is scored over
    a different market than one listed in 2016 -- so ranking them against each other
    measures *when each asset happened to exist*, not which one the strategy works on.
    Crypto makes this especially treacherous: listing dates cluster around bull markets,
    and delistings (XRP during the SEC suit) punch holes in the middle of a series.

    `min_years` drops short-history assets *before* intersecting, which trades universe
    breadth against window length. Dropping the newest asset often buys years of common
    history back, so it is worth tuning rather than accepting.

    Returns (aligned, (start_ts, end_ts), dropped_symbols).
    """
    usable = {s: b for s, b in bars_by_symbol.items() if b.ts is not None and len(b) > 1}
    if len(usable) < 2:
        return dict(bars_by_symbol), (0.0, 0.0), []

    dropped: list[str] = []
    if min_years > 0:
        keep = {}
        for s, b in usable.items():
            years = (b.ts[-1] - b.ts[0]) / (365.25 * 86400.0)
            if years >= min_years:
                keep[s] = b
            else:
                dropped.append(s)
        usable = keep or usable

    start = max(float(b.ts[0]) for b in usable.values())
    end = min(float(b.ts[-1]) for b in usable.values())
    if end <= start:
        return dict(bars_by_symbol), (0.0, 0.0), dropped

    aligned: dict[str, Bars] = {}
    for s, b in usable.items():
        lo = int(np.searchsorted(b.ts, start, side="left"))
        hi = int(np.searchsorted(b.ts, end, side="right"))
        if hi - lo >= 50:
            aligned[s] = b.slice(lo, hi)

    return aligned, (start, end), dropped


@dataclass
class AssetResult:
    symbol: str
    bars: int
    years: float
    live_sharpe: float  # annualised, at the shipped parameters
    live_return: float  # total net return at the shipped parameters
    best_sharpe: float  # annualised, best cell of the grid
    trial_sharpes: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    failed: bool = False
    note: str = ""

    @property
    def profitable(self) -> bool:
        return self.live_return > 0


@dataclass
class UniverseVerdict:
    chosen: list[str]
    results: list[AssetResult]
    selection_p: float  # P(a random k-subset does at least as well)
    n_subsets: int
    breadth: float  # fraction of the universe profitable at live params
    median_sharpe: float  # what a randomly picked asset would have earned
    chosen_mean_sharpe: float
    universe_dsr: float  # deflated by grid x assets, not grid alone
    universe_trials: int
    grid_trials: int
    score: float
    label: str
    headline: str
    advice: str = ""
    window: tuple[float, float] = (0.0, 0.0)  # common period every asset was scored over
    dropped_short: list[str] = field(default_factory=list)

    @property
    def window_years(self) -> float:
        lo, hi = self.window
        return (hi - lo) / (365.25 * 86400.0) if hi > lo else 0.0

    @property
    def ranked(self) -> list[AssetResult]:
        return sorted([r for r in self.results if not r.failed],
                      key=lambda r: r.live_sharpe, reverse=True)

    def rank_of(self, symbol: str) -> int:
        for i, r in enumerate(self.ranked, 1):
            if r.symbol == symbol:
                return i
        return -1


def run_universe(
    strategy,
    bars_by_symbol: Mapping[str, Bars],
    spec: MarketSpec,
    grid: Mapping[str, Sequence[float]],
    chosen: Sequence[str],
    *,
    params: Mapping[str, float] | None = None,
    valid: Callable[[dict], bool] | None = None,
    min_bars: int = 300,
    align: bool = True,
    min_years: float = 0.0,
    progress: Callable[[str], None] | None = None,
) -> UniverseVerdict:
    """Sweep every candidate asset, then test the choice of assets itself.

    `chosen` are the symbols actually shipped. `params` are the shipped parameters --
    the whole point is to evaluate every asset at the SAME settings, so that what
    varies across the universe is the asset and nothing else. Letting each asset pick
    its own best cell would bake a second layer of selection into the comparison.
    """
    say = progress or (lambda _: None)
    chosen = list(chosen)

    missing = [c for c in chosen if c not in bars_by_symbol]
    if missing:
        raise ValueError(f"chosen symbols not in the universe: {missing}")

    window = (0.0, 0.0)
    dropped_short: list[str] = []
    if align:
        bars_by_symbol, window, dropped_short = align_common_window(
            bars_by_symbol, min_years=min_years, bars_per_year=spec.bars_per_year
        )
        still_missing = [c for c in chosen if c not in bars_by_symbol]
        if still_missing:
            raise ValueError(
                f"aligning to a common window dropped shipped assets {still_missing}. "
                "Lower min_years, or pass align=False and accept that assets are being "
                "compared over different periods."
            )
        say(f"aligned {len(bars_by_symbol)} assets to a common window")

    results: list[AssetResult] = []
    pooled_trials: list[np.ndarray] = []

    for sym, bars in bars_by_symbol.items():
        if len(bars) < min_bars:
            results.append(AssetResult(sym, len(bars), 0.0, 0.0, 0.0, 0.0,
                                       failed=True, note=f"only {len(bars)} bars"))
            continue

        say(f"sweeping {sym}")
        try:
            sw = sweep(strategy, bars, spec, grid, valid=valid)
        except Exception as e:
            results.append(AssetResult(sym, len(bars), 0.0, 0.0, 0.0, 0.0,
                                       failed=True, note=repr(e)))
            continue

        idx = sw.index_of(params) if params else sw.best_index
        live_net = sw.returns[:, idx].astype(np.float64)
        live = sw.result_for(idx)

        alive = sw.sharpes[~sw.failed]
        years = (len(bars) / spec.bars_per_year)

        results.append(AssetResult(
            symbol=sym,
            bars=len(bars),
            years=years,
            live_sharpe=annualise(sharpe(live_net), spec.bars_per_year),
            live_return=live.net_return,
            best_sharpe=annualise(float(np.max(alive)) if len(alive) else 0.0,
                                  spec.bars_per_year),
            trial_sharpes=alive,
        ))
        pooled_trials.append(alive)

    live_ok = [r for r in results if not r.failed]
    if len(live_ok) < 3:
        raise RuntimeError("need at least three usable assets to test selection")

    # --- the exact selection test -------------------------------------------------
    # Under the null that the assets were exchangeable at the time of choosing, the
    # shipped set is one draw from all C(N, k). Score every draw the same way and
    # count how many match or beat it.
    symbols = [r.symbol for r in live_ok]
    sharpes = {r.symbol: r.live_sharpe for r in live_ok}
    k = len([c for c in chosen if c in sharpes])
    chosen_mean = float(np.mean([sharpes[c] for c in chosen if c in sharpes]))

    subsets = list(combinations(symbols, k))
    means = np.array([np.mean([sharpes[s] for s in sub]) for sub in subsets])
    at_least_as_good = int(np.count_nonzero(means >= chosen_mean - 1e-12))
    selection_p = at_least_as_good / len(subsets)

    breadth = float(np.mean([r.profitable for r in live_ok]))
    median_sharpe = float(np.median([r.live_sharpe for r in live_ok]))

    # --- deflation by the *combined* search ---------------------------------------
    pooled = np.concatenate(pooled_trials) if pooled_trials else np.array([0.0])
    grid_trials = int(len(pooled) / max(1, len(pooled_trials)))
    universe_trials = int(len(pooled))

    chosen_net = []
    for c in chosen:
        if c not in sharpes:
            continue
        sw = sweep(strategy, bars_by_symbol[c], spec, grid, valid=valid)
        idx = sw.index_of(params) if params else sw.best_index
        chosen_net.append(sw.returns[:, idx].astype(np.float64))

    # Equal-weight the shipped basket, which is how it is actually traded.
    n = min(len(x) for x in chosen_net)
    basket = np.mean([x[-n:] for x in chosen_net], axis=0)
    universe_dsr, _ = deflated_sharpe(basket, universe_trials, float(np.var(pooled, ddof=1)))

    # --- scoring -------------------------------------------------------------------
    # Selection p-value dominates; breadth is corroboration. A strategy that works on
    # most of the universe cannot have been rescued by asset picking, whatever the rank.
    sel_score = float(np.clip(selection_p / 0.20, 0.0, 1.0))  # p >= 0.20 -> full marks
    breadth_score = float(np.clip((breadth - 0.4) / 0.4, 0.0, 1.0))  # 40% -> 0, 80% -> 1
    score = max(sel_score, breadth_score) * (0.5 + 0.5 * universe_dsr)

    rank_txt = ", ".join(
        f"{c} #{_rank(live_ok, c)}" for c in chosen if c in sharpes
    )

    if selection_p <= 0.05 and breadth < 0.5:
        label = "SELECTION BIAS"
        headline = (
            f"The shipped set looks picked. {rank_txt} out of {len(live_ok)} candidates; "
            f"only {at_least_as_good} of {len(subsets)} possible {k}-asset choices would "
            f"have done as well (p = {selection_p:.3f}), and the strategy is profitable on "
            f"just {breadth:.0%} of the universe."
        )
        advice = (
            "Deflate by the whole search, not the grid: with "
            f"{universe_trials:,} effective trials the basket's DSR is {universe_dsr:.2f}. "
            "If the choice of assets can be justified from something other than returns "
            "(liquidity, market cap, a prior), say so explicitly — otherwise this is "
            "curve fitting one level up from the parameters."
        )
    elif breadth >= 0.6:
        label = "BROAD"
        headline = (
            f"The edge is not confined to the shipped set. Profitable on {breadth:.0%} of "
            f"{len(live_ok)} candidates at the same parameters, median Sharpe "
            f"{median_sharpe:.2f}. {rank_txt}."
        )
        advice = ""
    else:
        label = "MIXED"
        headline = (
            f"{rank_txt} out of {len(live_ok)}. A random {k}-asset pick matches this "
            f"{selection_p:.0%} of the time, and the strategy is profitable on "
            f"{breadth:.0%} of the universe."
        )
        advice = (
            "Not clearly selection bias, not clearly broad. Widen the shipped set and see "
            "whether the aggregate holds up — that is the cheapest way to find out."
        )

    return UniverseVerdict(
        chosen=chosen,
        results=results,
        selection_p=selection_p,
        n_subsets=len(subsets),
        breadth=breadth,
        median_sharpe=median_sharpe,
        chosen_mean_sharpe=chosen_mean,
        universe_dsr=universe_dsr,
        universe_trials=universe_trials,
        grid_trials=grid_trials,
        score=float(np.clip(score, 0.0, 1.0)) * 100.0,
        label=label,
        headline=headline,
        advice=advice,
        window=window,
        dropped_short=dropped_short,
    )


def _rank(live_ok: list[AssetResult], symbol: str) -> int:
    order = sorted(live_ok, key=lambda r: r.live_sharpe, reverse=True)
    for i, r in enumerate(order, 1):
        if r.symbol == symbol:
            return i
    return -1
