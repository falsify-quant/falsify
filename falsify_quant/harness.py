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

__all__ = ["Sweep", "StrategyError", "sweep", "grid_size"]

ParamDict = dict[str, float]


class StrategyError(Exception):
    """The user's strategy could not be run, with an explanation of why.

    Separate from ValueError so the CLI can print it as advice rather than as a
    crash. Every message this carries names the actual cause and what to change.
    """


def _params_str(params: ParamDict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in params.items()) or "(no parameters)"


def _accepted_params(strategy) -> list[str] | None:
    """Parameter names `strategy` accepts, or None if it cannot be inspected."""
    try:
        import inspect

        sig = inspect.signature(strategy)
    except (TypeError, ValueError):
        return None
    return [
        p.name
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]


def _check_weights(w, n_bars: int) -> str | None:
    """Why `w` is not a usable weight series, or None if it is fine.

    Checked explicitly rather than left to fail somewhere downstream, because the
    error the engine would raise on its own describes the engine's problem and not
    the user's mistake.
    """
    if w is None:
        return ("strategy() returned None -- did you forget to `return`?")
    try:
        arr = np.asarray(w, dtype=float)
    except (TypeError, ValueError) as exc:
        return (f"strategy() returned {type(w).__name__}, which is not a sequence of "
                f"numbers ({exc}). Return one float per bar.")
    if arr.ndim != 1:
        return (f"strategy() returned an array with shape {arr.shape}. Return a flat "
                "series of one weight per bar.")
    if len(arr) != n_bars:
        return (
            f"strategy() returned {len(arr)} weights for {n_bars} bars.\n\n"
            "Return one weight per bar, aligned to `bars`. If your indicator has a\n"
            "warmup, pad the front with np.nan -- the engine reads NaN as flat --\n"
            "rather than returning a shorter array."
        )
    if arr.size and np.all(np.isnan(arr)):
        return ("strategy() returned all NaN, so the warmup never ends. Check that "
                "your longest window is shorter than the data.")
    return None


def _diagnose(reason: str, params: ParamDict, strategy, grid) -> str:
    """Turn one concrete failure into advice.

    The old behaviour was a single message blaming the signature no matter what
    actually went wrong, which misdirects three times out of four: a DataFrame-style
    subscript, a GRID key that is not a parameter, a missing `return` and a
    wrong-length array are four different mistakes with four different fixes.
    """
    hint = ""
    low = reason.lower()

    if "not subscriptable" in low:
        hint = (
            "\n`bars` is a Bars object, not a DataFrame. Use attribute access:\n\n"
            "    bars.close   bars.open   bars.high   bars.low   bars.volume\n\n"
            "Each is a plain numpy array, so pandas users can wrap it:\n"
            "    c = pd.Series(bars.close)\n"
        )
    elif "unexpected keyword argument" in low:
        accepted = _accepted_params(strategy)
        lines = ["\nGRID has a key your strategy does not accept. Every GRID key must "
                 "be a\nparameter name of strategy()."]
        lines.append(f"\n    GRID keys        : {', '.join(grid) or '(none)'}")
        if accepted is not None:
            lines.append(f"    strategy accepts : {', '.join(accepted) or '(none)'}")
            extra = [k for k in grid if k not in accepted]
            if extra:
                lines.append(f"\nNot accepted: {', '.join(extra)}. Either add "
                             "them as parameters or drop them from GRID.")
        hint = "\n".join(lines) + "\n"
    elif "missing" in low and "positional argument" in low:
        accepted = _accepted_params(strategy)
        hint = (
            "\nstrategy() has a parameter with no default that GRID does not supply.\n"
            "Give every parameter a default, or add it to GRID.\n"
            + (f"\n    strategy accepts : {', '.join(accepted)}\n" if accepted else "")
        )

    return (
        f"The first failure was at {_params_str(params)}:\n\n"
        f"    {reason}\n" + hint
    )


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
    # (params, reason) of the first combination that failed, kept so a PARTIAL
    # failure can be reported instead of passing silently. A grid where half the
    # cells failed still deflates as if all of them ran, so silence here quietly
    # corrupts the one number this tool exists to get right.
    first_failure: tuple[ParamDict, str] | None = field(repr=False, default=None)

    @property
    def n_trials(self) -> int:
        return len(self.params)

    @property
    def n_failed(self) -> int:
        return int(self.failed.sum())

    def failure_warning(self) -> str | None:
        """A note for the user when some -- but not all -- of the grid failed."""
        n = self.n_failed
        if n == 0 or self.first_failure is None:
            return None
        params, reason = self.first_failure
        return (
            f"{n} of {self.n_trials} parameter combinations failed and were skipped.\n"
            f"  first: {_params_str(params)}\n"
            f"  {reason.splitlines()[0]}\n"
            "Those cells are excluded from the deflation, so the trial count reflects "
            "what actually ran."
        )

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

    # Keep the FIRST real failure. Individual combinations are still allowed to fail
    # without crashing the run -- one bad corner of the grid should not cost you the
    # sweep -- but discarding the exception entirely is what left every mistake
    # reporting the same misleading sentence.
    first_failure: tuple[ParamDict, str] | None = None

    def note(params: ParamDict, reason: str) -> None:
        nonlocal first_failure
        if first_failure is None:
            first_failure = (params, reason)

    for i, params in enumerate(combos):
        try:
            w = strategy(bars, **params)
            bad = _check_weights(w, T)
            if bad is not None:
                raise StrategyError(bad)
            res = simulate(bars, w, spec)
        except StrategyError as exc:
            failed[i] = True
            note(params, str(exc))
            continue
        except Exception as exc:
            failed[i] = True
            note(params, f"{type(exc).__name__}: {exc}")
            continue

        if not np.all(np.isfinite(res.net)):
            failed[i] = True
            note(params, "the simulated net return contained inf or NaN")
            continue

        returns[:, i] = res.net.astype(np.float32)
        gross[i] = res.gross_return
        churn[i] = float(np.sum(res.turnover))

        if progress is not None and (i % 64 == 0 or i == N - 1):
            progress(i + 1, N)

    if failed.all():
        params, reason = first_failure
        raise StrategyError(
            f"Every one of the {N} parameter combinations failed, so there is "
            f"nothing to score.\n\n"
            + _diagnose(reason, params, strategy, grid)
        )

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
        first_failure=first_failure,
    )
