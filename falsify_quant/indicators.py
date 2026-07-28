"""Causal technical indicators. Every one of them, by construction.

These exist because the corpus study needed reference implementations of the published
canon, and a reference implementation that leaks is worse than none at all. The contract
is uniform and strict:

    out[i] depends on x[:i+1] and nothing else.
    Warmup is NaN, never a backfilled or forward-propagated value.

That second clause is not fussiness. `fillna(method="bfill")` is the single most common
way a lookahead bug enters a strategy, and it enters through the indicator layer, not the
signal layer. An indicator that quietly fills its warmup with the first real value has
told bar 0 what bar 20 looks like, and every downstream statistic is then fiction.

The smoothed families (EMA, Wilder) are IIR filters, so they are computed with
`scipy.signal.lfilter` rather than a Python loop -- same recursion, same numbers, about
two orders of magnitude faster, which matters when the permutation test re-runs an entire
sweep a hundred times.

Warmup convention for the recursive filters follows the textbooks: seed at index n-1 with
the simple average of the first n samples, then recurse. Different libraries disagree
here (some start the recursion at index 0, some at n), and the difference persists for
several multiples of n. It is documented rather than assumed so that a disagreement with
your charting package is explainable instead of alarming.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from .spec import Bars

__all__ = [
    "shift",
    "rolling_mean",
    "rolling_std",
    "rolling_max",
    "rolling_min",
    "rolling_sum",
    "ema",
    "wilder",
    "true_range",
    "atr",
    "rsi",
    "macd",
    "bollinger",
    "keltner",
    "stochastic",
    "williams_r",
    "realised_vol",
    "day_of_month",
    "days_in_month",
]


# --------------------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------------------


def shift(x: np.ndarray, k: int = 1) -> np.ndarray:
    """Move a series `k` bars into the future: `out[i] = x[i-k]`. NaN at the start.

    Refuses negative `k`. A negative shift is a lookahead by definition, and the one
    place it reliably appears is someone "fixing" an off-by-one by flipping the sign
    until the equity curve looks better. There is no legitimate use for it inside a
    strategy, so this function will not write the bug for you.
    """
    k = int(k)
    if k < 0:
        raise ValueError(
            f"shift({k}) reads {-k} bar(s) into the future. If you are trying to correct "
            "an off-by-one, the engine already applies the execution lag -- shifting "
            "backwards here double-counts it in the wrong direction."
        )
    out = np.full(len(x), np.nan)
    if k == 0:
        out[:] = x
    elif k < len(x):
        out[k:] = np.asarray(x, dtype=np.float64)[:-k]
    return out


# --------------------------------------------------------------------------------------
# Trailing windows
# --------------------------------------------------------------------------------------


def _windows(x: np.ndarray, n: int) -> np.ndarray | None:
    """Sliding trailing windows as a view. None if the window does not fit."""
    n = int(n)
    if n < 1 or n > len(x):
        return None
    return np.lib.stride_tricks.sliding_window_view(np.asarray(x, dtype=np.float64), n)


def _prefix_sums(x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Windowed sums of x and x**2, plus a mask of windows that contained a NaN.

    The cumulative-sum trick that makes these O(N) has one sharp edge: a single NaN
    anywhere in the input poisons every prefix sum after it, so one missing print at bar
    3 turns the whole rest of the series into NaN rather than just the windows that
    actually touch it. Nothing about the output looks wrong -- it looks like a longer
    warmup -- which is how it survives a visual check.

    NaNs are therefore zeroed out of the sums and counted separately, so exactly the
    windows that contain one come back NaN and the rest are unaffected.
    """
    n = int(n)
    if n < 1 or n > len(x):
        return None
    x = np.asarray(x, dtype=np.float64)
    bad = ~np.isfinite(x)
    z = np.where(bad, 0.0, x)

    c1 = np.cumsum(np.insert(z, 0, 0.0))
    c2 = np.cumsum(np.insert(z * z, 0, 0.0))
    cb = np.cumsum(np.insert(bad.astype(np.float64), 0, 0.0))
    return c1[n:] - c1[:-n], c2[n:] - c2[:-n], (cb[n:] - cb[:-n]) > 0


def rolling_sum(x: np.ndarray, n: int) -> np.ndarray:
    """Trailing sum over n bars. `out[i]` uses x[i-n+1 .. i]."""
    out = np.full(len(x), np.nan)
    sums = _prefix_sums(x, n)
    if sums is None:
        return out
    s1, _, bad = sums
    out[int(n) - 1:] = np.where(bad, np.nan, s1)
    return out


def rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Trailing mean over n bars. `out[i]` uses x[i-n+1 .. i] only. NaN during warmup."""
    out = np.full(len(x), np.nan)
    sums = _prefix_sums(x, n)
    if sums is None:
        return out
    s1, _, bad = sums
    out[int(n) - 1:] = np.where(bad, np.nan, s1 / int(n))
    return out


def rolling_std(x: np.ndarray, n: int) -> np.ndarray:
    """Trailing sample standard deviation over n bars, same warmup convention."""
    n = int(n)
    out = np.full(len(x), np.nan)
    if n < 2:
        return out
    sums = _prefix_sums(x, n)
    if sums is None:
        return out
    s1, s2, bad = sums
    var = np.maximum((s2 - s1 * s1 / n) / (n - 1), 0.0)
    out[n - 1:] = np.where(bad, np.nan, np.sqrt(var))
    return out


def rolling_max(x: np.ndarray, n: int) -> np.ndarray:
    """Trailing maximum over n bars, *including the current one*.

    Breakout strategies almost always want `shift(rolling_max(high, n), 1)`: the channel
    formed by the prior n bars, compared against today. Without the shift, `close >
    rolling_max(close, n)` can only ever be an equality, because today's close is inside
    today's window -- a silent no-trade strategy that reports a perfect zero.
    """
    w = _windows(x, n)
    out = np.full(len(x), np.nan)
    if w is None:
        return out
    out[int(n) - 1:] = w.max(axis=1)
    return out


def rolling_min(x: np.ndarray, n: int) -> np.ndarray:
    """Trailing minimum over n bars, including the current one. See `rolling_max`."""
    w = _windows(x, n)
    out = np.full(len(x), np.nan)
    if w is None:
        return out
    out[int(n) - 1:] = w.min(axis=1)
    return out


# --------------------------------------------------------------------------------------
# Recursive smoothers
# --------------------------------------------------------------------------------------


def _recursive(x: np.ndarray, alpha: float, n: int) -> np.ndarray:
    """y[t] = alpha*x[t] + (1-alpha)*y[t-1], seeded with the mean of the first n samples.

    Runs as a first-order IIR through `lfilter`. The initial condition `zi` for direct
    form II transposed is `(1-alpha) * y_seed`, which reproduces the recursion exactly
    from the first output sample onwards.

    A leading run of NaN is skipped rather than fatal, because inputs that begin with one
    are normal: a return series has no return on its first bar, and a true range has no
    previous close to gap from. Without this, `atr` on a close-only history is NaN from
    end to end, and a channel strategy built on it silently never trades.

    An *interior* NaN is still fatal to everything after it. That is a property of the
    recursion, not a shortcut -- a running average with a missing term cannot be resumed
    without inventing the term. Clean gaps out of your data before smoothing it.
    """
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    out = np.full(N, np.nan)
    n = int(n)
    if n < 1 or n > N:
        return out

    finite = np.isfinite(x)
    if not finite.any():
        return out
    start = int(np.argmax(finite))
    if N - start < n:
        return out

    seed = float(np.mean(x[start: start + n]))
    if not np.isfinite(seed):
        return out

    end = start + n
    out[end - 1] = seed
    if end < N:
        tail, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], x[end:], zi=[(1.0 - alpha) * seed])
        out[end:] = tail
    return out


def ema(x: np.ndarray, n: int) -> np.ndarray:
    """Exponential moving average with the conventional smoothing factor 2/(n+1)."""
    n = int(n)
    if n < 1:
        return np.full(len(x), np.nan)
    return _recursive(x, 2.0 / (n + 1.0), n)


def wilder(x: np.ndarray, n: int) -> np.ndarray:
    """Wilder's smoothing: alpha = 1/n, the one used by RSI, ATR and ADX.

    Equivalent to an EMA of period 2n-1, which is why an "RSI(14)" reacts like a 27-day
    average and not a 14-day one. Charting packages that implement RSI with a standard
    EMA produce a visibly different, faster line; both get called RSI(14).
    """
    n = int(n)
    if n < 1:
        return np.full(len(x), np.nan)
    return _recursive(x, 1.0 / n, n)


# --------------------------------------------------------------------------------------
# Range and volatility
# --------------------------------------------------------------------------------------


def true_range(bars: Bars) -> np.ndarray:
    """Wilder's true range. Falls back to |close-to-close| when there is no OHLC.

    The fallback is not equivalent -- it misses the intrabar range entirely and so
    understates the true range, typically by a third -- but it lets ATR-scaled strategies
    run on close-only histories rather than silently producing NaN everywhere.
    """
    c = np.asarray(bars.close, dtype=np.float64)
    if bars.high is None or bars.low is None:
        tr = np.empty(len(c))
        tr[0] = np.nan
        tr[1:] = np.abs(np.diff(c))
        return tr

    h = np.asarray(bars.high, dtype=np.float64)
    lo = np.asarray(bars.low, dtype=np.float64)
    prev = shift(c, 1)
    tr = np.maximum(h - lo, np.maximum(np.abs(h - prev), np.abs(lo - prev)))
    tr[0] = h[0] - lo[0]  # no previous close to gap from
    return tr


def atr(bars: Bars, n: int = 14) -> np.ndarray:
    """Average true range, Wilder-smoothed. In price units, not percent."""
    return wilder(true_range(bars), n)


def realised_vol(close: np.ndarray, n: int = 20) -> np.ndarray:
    """Trailing standard deviation of simple returns, per bar (not annualised)."""
    c = np.asarray(close, dtype=np.float64)
    r = np.full(len(c), np.nan)
    r[1:] = np.diff(c) / c[:-1]
    return rolling_std(r, n)


# --------------------------------------------------------------------------------------
# Named oscillators
# --------------------------------------------------------------------------------------


def rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    """Wilder's Relative Strength Index, 0-100. First valid value at index n.

    Gains and losses are smoothed with Wilder's alpha = 1/n, seeded on the first n
    deltas.

    Two degenerate cases have to be decided rather than left to floating point. No losses
    at all is 100, which is the definition's limit and what every implementation agrees
    on. No losses *and* no gains -- a perfectly flat window -- is 50 here, not 100: there
    is no directional pressure to report, and 100 would fire the short leg of every
    reversion strategy on a market that had not moved. Flat stretches are common in thin
    instruments and in any series carrying stale or padded prints, so this is not a
    theoretical corner.
    """
    c = np.asarray(close, dtype=np.float64)
    N = len(c)
    out = np.full(N, np.nan)
    n = int(n)
    if n < 1 or N < n + 1:
        return out

    delta = np.diff(c)
    up = wilder(np.maximum(delta, 0.0), n)
    down = wilder(np.maximum(-delta, 0.0), n)

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = up / down
    val = 100.0 - 100.0 / (1.0 + rs)
    warm = np.isfinite(up) & np.isfinite(down)
    val = np.where(warm & (down == 0) & (up > 0), 100.0, val)
    val = np.where(warm & (down == 0) & (up == 0), 50.0, val)

    out[1:] = val  # delta[j] belongs to bar j+1
    return out


def macd(
    close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Appel's MACD: (line, signal, histogram).

    The signal line is an EMA of the MACD line, so its warmup compounds -- the first
    honest histogram value is at roughly `slow + signal` bars, not `signal`.
    """
    line = ema(close, fast) - ema(close, slow)
    sig = np.full(len(line), np.nan)
    valid = np.isfinite(line)
    if valid.any():
        start = int(np.argmax(valid))
        sig[start:] = ema(line[start:], signal)
    return line, sig, line - sig


def bollinger(
    close: np.ndarray, n: int = 20, k: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger bands: (middle, upper, lower). Middle is the simple mean."""
    mid = rolling_mean(close, n)
    sd = rolling_std(close, n)
    return mid, mid + k * sd, mid - k * sd


def keltner(
    bars: Bars, n: int = 20, k: float = 2.0, atr_n: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keltner channel: EMA centre, width set by ATR rather than standard deviation."""
    mid = ema(bars.close, n)
    a = atr(bars, atr_n)
    return mid, mid + k * a, mid - k * a


def stochastic(bars: Bars, n: int = 14, smooth: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Lane's stochastic oscillator: (%K, %D), both 0-100.

    %K places the close within the trailing n-bar range; %D is its `smooth`-bar mean. A
    flat range (high == low across the whole window) gives 50 rather than a divide by
    zero -- the honest reading of "the close is nowhere in particular".
    """
    hi = rolling_max(bars.high if bars.high is not None else bars.close, n)
    lo = rolling_min(bars.low if bars.low is not None else bars.close, n)
    c = np.asarray(bars.close, dtype=np.float64)
    span = hi - lo
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 100.0 * (c - lo) / span
    k = np.where(np.isfinite(span) & (span <= 0), 50.0, k)
    return k, rolling_mean(k, smooth)


def williams_r(bars: Bars, n: int = 14) -> np.ndarray:
    """Williams %R, -100 to 0. The stochastic %K measured from the top of the range."""
    k, _ = stochastic(bars, n, smooth=1)
    return k - 100.0


# --------------------------------------------------------------------------------------
# Calendar
# --------------------------------------------------------------------------------------


def _as_datetime64(ts: np.ndarray) -> np.ndarray:
    return np.asarray(ts, dtype=np.float64).astype("datetime64[s]")


def day_of_month(ts: np.ndarray) -> np.ndarray:
    """1-31 for each bar, from unix timestamps."""
    d = _as_datetime64(ts)
    return (d.astype("datetime64[D]") - d.astype("datetime64[M]")).astype(np.int64) + 1


def days_in_month(ts: np.ndarray) -> np.ndarray:
    """Length of each bar's calendar month, in days. Leap years included."""
    m = _as_datetime64(ts).astype("datetime64[M]")
    nxt = m + np.timedelta64(1, "M")
    return (nxt.astype("datetime64[D]") - m.astype("datetime64[D]")).astype(np.int64)
