"""Sharpe ratio statistics that account for the fact that you went looking.

A Sharpe ratio computed from a backtest is not an estimate of skill. It is the maximum
of however many estimates you generated, and the maximum of N noisy draws is biased
upward by an amount that grows with N. If you swept 4,096 parameter combinations and
kept the best, roughly 3.0 of "Sharpe 3.0" can be manufactured from pure noise.

The corrections here are from Bailey & Lopez de Prado:

  * Probabilistic Sharpe Ratio      -- confidence that true SR exceeds a benchmark,
                                       adjusted for skew, fat tails and sample length
  * Expected maximum Sharpe         -- what the best of N random strategies scores
  * Deflated Sharpe Ratio           -- PSR measured against that inflated benchmark
  * Minimum Track Record Length     -- how long you must trade live to prove it

All Sharpe ratios in this module are *per observation*, not annualised. Annualisation is
a display concern and mixing the two is the most common way these formulas get misused.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

__all__ = [
    "EULER_MASCHERONI",
    "sharpe",
    "sharpe_columns",
    "annualise",
    "probabilistic_sharpe",
    "expected_max_sharpe",
    "deflated_sharpe",
    "min_track_record_length",
    "max_drawdown",
]

EULER_MASCHERONI = 0.5772156649015329


def sharpe(returns: np.ndarray) -> float:
    """Per-observation Sharpe ratio. Zero risk-free rate (bar-level rf is noise)."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    sd = float(np.std(r, ddof=1))
    if sd < 1e-15:
        return 0.0
    return float(np.mean(r)) / sd


def sharpe_columns(matrix: np.ndarray) -> np.ndarray:
    """Per-observation Sharpe of every column of a (T, N) returns matrix.

    Vectorised because the overfitting tests call this on tens of thousands of columns.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError(f"expected a 2-D (T, N) matrix, got shape {m.shape}")
    mu = np.mean(m, axis=0)
    sd = np.std(m, axis=0, ddof=1)
    out = np.zeros_like(mu)
    live = sd > 1e-15
    out[live] = mu[live] / sd[live]
    return out


def annualise(sr: float, bars_per_year: float) -> float:
    """Scale a per-observation Sharpe to annual terms."""
    return sr * np.sqrt(bars_per_year)


def probabilistic_sharpe(
    returns: np.ndarray,
    benchmark_sr: float = 0.0,
    *,
    observed_sr: float | None = None,
) -> float:
    """P(true Sharpe > benchmark), corrected for skew, kurtosis and sample length.

    Negative skew and fat tails -- the signature of strategies that sell insurance and
    look wonderful right up until they do not -- widen the error bars and push this down.

    `benchmark_sr` and `observed_sr` are per-observation.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 3:
        return 0.5

    # A constant series -- typically a variant that never opens a position -- has no
    # third or fourth moment, and scipy returns NaN. There is no evidence of skill in a
    # flat line, so say so rather than letting NaN propagate into the verdict.
    if float(np.std(r, ddof=1)) < 1e-15:
        return 0.5

    sr = sharpe(r) if observed_sr is None else observed_sr
    skew = float(sps.skew(r, bias=False))
    kurt = float(sps.kurtosis(r, fisher=False, bias=False))  # non-excess: normal == 3
    if not (np.isfinite(skew) and np.isfinite(kurt)):
        skew, kurt = 0.0, 3.0  # fall back to the Gaussian assumption

    # Variance of the Sharpe estimator under non-normality (Mertens / Bailey-LdP).
    var = 1.0 - skew * sr + 0.25 * (kurt - 1.0) * sr**2
    if not np.isfinite(var) or var <= 0:
        # Can go negative for wild higher moments on short samples; the statistic is
        # not meaningful there and claiming confidence would be the exact failure mode
        # this library exists to catch.
        return 0.5

    z = (sr - benchmark_sr) * np.sqrt(T - 1) / np.sqrt(var)
    out = float(sps.norm.cdf(z))
    return out if np.isfinite(out) else 0.5


def expected_max_sharpe(n_trials: int, variance_of_trial_sharpes: float) -> float:
    """The Sharpe the *best* of N genuinely worthless strategies is expected to post.

    This is the benchmark your backtest actually has to beat. It rises with the number
    of variants you tried, which is why the trial count has to be recorded honestly by
    the harness rather than remembered by the human.
    """
    n = int(n_trials)
    v = float(variance_of_trial_sharpes)
    if n <= 1 or v <= 0:
        return 0.0

    # E[max of n standard normals], Bailey & Lopez de Prado's two-term approximation.
    a = sps.norm.ppf(1.0 - 1.0 / n)
    b = sps.norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(np.sqrt(v) * ((1.0 - EULER_MASCHERONI) * a + EULER_MASCHERONI * b))


def deflated_sharpe(
    returns: np.ndarray,
    n_trials: int,
    variance_of_trial_sharpes: float,
    *,
    observed_sr: float | None = None,
) -> tuple[float, float]:
    """Return (DSR, deflated benchmark SR).

    DSR is a probability in [0, 1]: the confidence that the strategy's true Sharpe beats
    what the luckiest of `n_trials` coin flips would have produced. Below ~0.95 there is
    no evidence of skill; below ~0.5 the backtest is worse than the null.
    """
    sr0 = expected_max_sharpe(n_trials, variance_of_trial_sharpes)
    dsr = probabilistic_sharpe(returns, benchmark_sr=sr0, observed_sr=observed_sr)
    return dsr, sr0


def min_track_record_length(
    returns: np.ndarray,
    benchmark_sr: float = 0.0,
    confidence: float = 0.95,
) -> float:
    """Bars of *live* trading needed before the record itself proves SR > benchmark.

    Returns inf when the observed Sharpe does not exceed the benchmark at all, which is
    the honest answer: no amount of waiting proves something that is not there.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return float("inf")

    sr = sharpe(r)
    if sr <= benchmark_sr:
        return float("inf")

    skew = float(sps.skew(r, bias=False))
    kurt = float(sps.kurtosis(r, fisher=False, bias=False))
    var = 1.0 - skew * sr + 0.25 * (kurt - 1.0) * sr**2
    if var <= 0:
        return float("inf")

    z = sps.norm.ppf(confidence)
    return float(1.0 + var * (z / (sr - benchmark_sr)) ** 2)


def max_drawdown(returns: np.ndarray) -> float:
    """Worst peak-to-trough decline of the compounded equity curve, as a positive fraction."""
    r = np.asarray(returns, dtype=np.float64)
    if len(r) == 0:
        return 0.0
    equity = np.cumprod(1.0 + np.nan_to_num(r))
    peak = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / peak))
