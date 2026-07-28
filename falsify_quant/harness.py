"""The sweep harness: runs every parameter combination and remembers all of them.

This is the part that makes honest deflation possible, and it is the reason `falsify`
has to own the search loop rather than accept an uploaded backtest report.

The Deflated Sharpe Ratio needs two inputs that no report contains: N, the number of
variants actually tried, and V, the variance of their Sharpe ratios. Ask a human for N
and they will say "a few". The real answer is the size of the grid times the number of
times they re-ran it after not liking the answer. A tool that scores a single submitted
equity curve cannot deflate anything, because the search that produced it happened
somewhere the tool cannot see.

So the deal is: bring the strategy and the grid, and the harness does the searching.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .sim import simulate
from .spec import Bars, MarketSpec, Strategy
from .stats import sharpe_columns

__all__ = ["Sweep", "sweep", "grid_size"]

ParamDict = dict[str, float]


@dataclass
class Sweep:
    """Every trial that was run, not just the survivor."""

    params: list[ParamDict]
    returns: np.ndarray  # (T, N) net returns, float32
    sharpes: np.ndarray  # (N,) per-observation Sharpe
    gross: np.ndarray  # (N,) summed gross return
    churn: np.ndarray  # (N,) summed turnover
    failed: np.ndarray  # (N,) bool -- combination raised or produced nothing
    bars: Bars
    spec: MarketSpec
    strategy: Strategy = field(repr=False)
    grid: Mapping[str, Sequence[float]] = field(repr=False, default_factory=dict)

    @property
    def n_trials(self) -> int:
        return len(self.params)

    @property
    def n_bars(self) -> int:
        return self.returns.shape[0]

    @property
    def sharpe_variance(self) -> float:
        """Cross-sectional variance of trial Sharpes -- the V in expected-max-Sharpe.

        Measured over trials that actually ran. A wide spread here means the grid had
        lots of room to get lucky in, and deflation will be correspondingly brutal.
        """
        live = self.sharpes[~self.failed]
        if len(live) < 2:
            return 0.0
        return float(np.var(live, ddof=1))

    @property
    def best_index(self) -> int:
        """The trial a human would have shipped: highest in-sample Sharpe."""
        masked = np.where(self.failed, -np.inf, self.sharpes)
        return int(np.argmax(masked))

    @property
    def best_params(self) -> ParamDict:
        return self.params[self.best_index]

    def index_of(self, params: Mapping[str, float]) -> int:
        """Find the trial matching a specific parameter set, for when the user already
        picked one rather than taking the grid maximum."""
        for i, p in enumerate(self.params):
            if all(math.isclose(p.get(k, np.nan), v, rel_tol=1e-9) for k, v in params.items()):
                return i
        raise KeyError(f"no trial in this sweep used params {dict(params)}")

    def result_for(self, index: int):
        """Re-run one trial to get its full SimResult (positions, turnover, the lot)."""
        w = self.strategy(self.bars, **self.params[index])
        return simulate(self.bars, w, self.spec)


def grid_size(grid: Mapping[str, Sequence[float]]) -> int:
    n = 1
    for values in grid.values():
        n *= len(values)
    return n


def _expand(grid: Mapping[str, Sequence[float]]) -> list[ParamDict]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def sweep(
    strategy: Strategy,
    bars: Bars,
    spec: MarketSpec,
    grid: Mapping[str, Sequence[float]],
    *,
    valid: Callable[[ParamDict], bool] | None = None,
    max_trials: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Sweep:
    """Run the whole grid, keeping the net return series of every combination.

    `valid` filters out nonsensical combinations (fast >= slow, and so on) *before* they
    are counted as trials -- a combination that could never have been shipped should not
    inflate the deflation penalty.

    Combinations that raise are recorded as failures rather than crashing the sweep, so
    one bad corner of the parameter space does not cost you the run.
    """
    if len(bars) < 50:
        raise ValueError(
            f"need at least 50 bars to say anything honest, got {len(bars)}. Every "
            "statistic downstream of this is a sample estimate, and on a sample this "
            "small they are noise wearing a decimal point."
        )

    combos = _expand(grid)
    if valid is not None:
        combos = [c for c in combos if valid(c)]
    if not combos:
        raise ValueError("grid produced no valid parameter combinations")
    if max_trials is not None and len(combos) > max_trials:
        raise ValueError(
            f"grid has {len(combos)} combinations, above max_trials={max_trials}. "
            "Shrink the grid rather than raising the cap -- every extra combination is "
            "another lottery ticket the deflation has to charge you for."
        )

    T, N = len(bars), len(combos)
    returns = np.zeros((T, N), dtype=np.float32)
    gross = np.zeros(N, dtype=np.float64)
    churn = np.zeros(N, dtype=np.float64)
    failed = np.zeros(N, dtype=bool)

    for i, params in enumerate(combos):
        try:
            w = strategy(bars, **params)
            res = simulate(bars, w, spec)
        except Exception:
            failed[i] = True
            continue

        if not np.all(np.isfinite(res.net)):
            failed[i] = True
            continue

        returns[:, i] = res.net.astype(np.float32)
        gross[i] = res.gross_return
        churn[i] = float(np.sum(res.turnover))

        if progress is not None and (i % 64 == 0 or i == N - 1):
            progress(i + 1, N)

    if failed.all():
        raise RuntimeError("every parameter combination failed -- check the strategy signature")

    sharpes = sharpe_columns(returns.astype(np.float64))
    sharpes[failed] = -np.inf

    return Sweep(
        params=combos,
        returns=returns,
        sharpes=sharpes,
        gross=gross,
        churn=churn,
        failed=failed,
        bars=bars,
        spec=spec,
        strategy=strategy,
        grid=dict(grid),
    )
