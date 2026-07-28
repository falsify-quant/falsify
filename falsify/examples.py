"""Reference strategies and synthetic markets, used by the self-test.

Two of the strategies here are deliberately broken and one of the markets deliberately
contains a real, known edge. Together they pin down the two ways `falsify` could be
worthless: by failing to catch a fake, or by refusing to pass a genuine result.
"""

from __future__ import annotations

import numpy as np

from .indicators import rolling_mean, rolling_std
from .spec import Bars

__all__ = [
    "rolling_mean",
    "rolling_std",
    "ma_crossover",
    "ma_crossover_leaky",
    "zscore_threshold_leaky",
    "zscore_reversion",
    "random_market",
    "momentum_market",
    "trending_market",
]


# `rolling_mean` and `rolling_std` now live in `falsify.indicators` alongside the rest of
# the causal primitives, and are re-exported here so the self-test and the example
# strategies keep importing them from the place they always did.


# --------------------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------------------


def ma_crossover(bars: Bars, fast: float = 10, slow: float = 50) -> np.ndarray:
    """Long when the fast trailing mean is above the slow one, short otherwise.

    Correct by construction: every value is computed from trailing windows, and the
    execution lag is left to the engine.
    """
    c = bars.close
    f = rolling_mean(c, int(fast))
    s = rolling_mean(c, int(slow))
    w = np.where(f > s, 1.0, -1.0)
    w[np.isnan(f) | np.isnan(s)] = np.nan  # warmup: engine reads NaN as flat
    return w


def ma_crossover_leaky(bars: Bars, fast: float = 10, slow: float = 50) -> np.ndarray:
    """The same idea, smoothed with a *centred* window. This is a real lookahead.

    `mode="same"` on a convolution centres the kernel, so the value at bar i is built
    from bars i-n/2 .. i+n/2. Half the window is in the future. The equivalents in other
    stacks are `pandas.rolling(..., center=True)` and `scipy.signal.savgol_filter`, and
    they appear in retail strategy code constantly, usually added to "reduce whipsaw".

    A note on a leak that is *not* here: normalising by full-sample mean and standard
    deviation, the other classic suspect, is invisible to a `fast > slow` comparison,
    because an affine transform applied to both sides cancels. It only bites when the
    normalised value is compared against a fixed threshold. Whether preprocessing leaks
    depends on what you do with it downstream -- which is exactly why the causality test
    checks behaviour rather than inspecting code.
    """
    c = np.asarray(bars.close, dtype=np.float64)
    n = int(fast)
    kernel = np.ones(n) / n
    f = np.convolve(c, kernel, mode="same")  # <-- centred: reads n/2 bars ahead
    s = rolling_mean(c, int(slow))
    w = np.where(f > s, 1.0, -1.0)
    w[np.isnan(s)] = np.nan
    return w


def zscore_threshold_leaky(bars: Bars, lookback: float = 20, entry: float = 1.0) -> np.ndarray:
    """Full-sample normalisation that *does* leak, because a fixed threshold sees it.

    The deviation from the trailing mean is causal. Dividing it by its own standard
    deviation over the *whole* sample is not: it tells bar 30 how volatile the next ten
    years are going to be. Scaling a signal "so the threshold means something" is a very
    natural thing to write, which is what makes this leak so common.

    Contrast `ma_crossover_leaky`, where an identical-looking normalisation is harmless
    because the comparison is scale-invariant. Here it is compared to a constant, so the
    scale factor survives and carries future information into every decision.
    """
    c = np.asarray(bars.close, dtype=np.float64)
    dev = c - rolling_mean(c, int(lookback))
    z = dev / (np.nanstd(dev) or 1.0)  # <-- the leak: full-sample scale
    w = np.where(z > float(entry), -1.0, np.where(z < -float(entry), 1.0, 0.0))
    w[np.isnan(z)] = np.nan
    return w


def zscore_reversion(bars: Bars, lookback: float = 20, entry: float = 1.0) -> np.ndarray:
    """Fade deviations from a trailing mean. Causal; trades a lot, which is the point --
    it is the archetype of a strategy that looks fine until costs are applied."""
    c = bars.close
    mu = rolling_mean(c, int(lookback))
    sd = rolling_std(c, int(lookback))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (c - mu) / sd
    w = np.clip(-z / float(entry), -1.0, 1.0)
    w[~np.isfinite(w)] = np.nan
    return w


# --------------------------------------------------------------------------------------
# Synthetic markets
# --------------------------------------------------------------------------------------


def random_market(n: int = 3000, *, vol: float = 0.015, drift: float = 0.0,
                  seed: int = 0, p0: float = 100.0, symbol: str = "NOISE") -> Bars:
    """Geometric Brownian motion. Contains no exploitable structure of any kind.

    Any strategy that scores well here is measuring its own search, not the market.
    """
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    return Bars(close=p0 * np.cumprod(1.0 + r), symbol=symbol)


def momentum_market(n: int = 3000, *, phi: float = 0.25, vol: float = 0.012,
                    seed: int = 0, p0: float = 100.0, symbol: str = "AR1") -> Bars:
    """AR(1) returns: r[t] = phi*r[t-1] + noise.

    A real edge, but one that lives almost entirely at lag 1 -- autocorrelation decays
    as phi^k, so at phi=0.35 it is 0.35 at lag 1 and 0.004 by lag 5. A moving-average
    crossover trading a 20/90 horizon structurally cannot see it. Useful precisely
    because it separates "there is no signal" from "the strategy is the wrong instrument
    for the signal", which are different diagnoses that look identical on an equity curve.
    """
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, vol, n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + eps[i]
    return Bars(close=p0 * np.cumprod(1.0 + r), symbol=symbol)


def trending_market(n: int = 5000, *, half_life: float = 100.0, trend_vol: float = 0.0022,
                    noise_vol: float = 0.011, seed: int = 0, p0: float = 100.0,
                    symbol: str = "TREND") -> Bars:
    """Returns with a slow, persistent, stochastic drift -- genuine multi-month trends.

    The drift follows its own AR(1) with a long half-life, so the market spends stretches
    of roughly `half_life` bars leaning one way. This is a real edge at the horizon a
    moving-average crossover actually trades, which makes it the honest control: the
    signal and the instrument are matched.

    This is a calibration weight, not a simulation. The trends are stronger and cleaner
    than any real market's. That is the point -- if `falsify` cannot pass this, it is
    not measuring anything, and a tool that answers "overfit" to everything is exactly
    as informative as a stopped clock while looking far more rigorous.
    """
    rng = np.random.default_rng(seed)
    rho = 0.5 ** (1.0 / half_life)
    shock = np.sqrt(1.0 - rho**2) * trend_vol

    mu = np.zeros(n)
    z = rng.normal(0.0, 1.0, n)
    for i in range(1, n):
        mu[i] = rho * mu[i - 1] + shock * z[i]

    r = mu + rng.normal(0.0, noise_vol, n)
    return Bars(close=p0 * np.cumprod(1.0 + r), symbol=symbol)
