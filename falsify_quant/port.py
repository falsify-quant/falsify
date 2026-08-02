"""Getting a strategy you already have into the shape falsify scores.

falsify wants a **target weight per bar**. Almost nobody writes strategies that way.
vectorbt's core call is `from_signals(entries, exits)`, backtrader users write
`buy()`/`sell()`, Pine users write `strategy.entry`/`strategy.close`, and anyone
describing an idea in English says "buy when X, sell when Y". Those are all the same
model -- discrete events -- and the translation to a continuous position series is
the single most common thing standing between someone's working backtest and a
verdict.

It is also genuinely easy to get wrong. The event model leaves three questions
unanswered that a weight series has to answer: what happens on a bar where entry and
exit both fire, what happens to an entry while already in a position, and what the
position is before the first event ever occurs. Getting the second one wrong is how a
port silently trades twice.

    from falsify_quant.port import from_signals

    def strategy(bars, fast=20, slow=100):
        f, s = sma(bars.close, fast), sma(bars.close, slow)
        return from_signals(entries=f > s, exits=f < s, warmup=int(slow))

Nothing here computes a signal for you. It converts the signal you already have.
"""

from __future__ import annotations

import numpy as np

__all__ = ["from_signals", "from_positions"]


def _as_bool(x, n: int, name: str) -> np.ndarray:
    if x is None:
        return np.zeros(n, dtype=bool)
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a flat array, got shape {arr.shape}")
    if len(arr) != n:
        raise ValueError(
            f"{name} has {len(arr)} values but entries has {n}. Every signal array "
            "must be the same length as the bars."
        )
    # NaN is not a signal. Treat it as "no event" rather than letting it become True.
    if arr.dtype.kind == "f":
        arr = np.nan_to_num(arr, nan=0.0)
    return arr.astype(bool)


def from_signals(
    entries,
    exits=None,
    *,
    short_entries=None,
    short_exits=None,
    size: float = 1.0,
    warmup: int = 0,
) -> np.ndarray:
    """Entry/exit events -> one target weight per bar.

    The rule is **last event wins**: the weight at bar `i` is set by the most recent
    event at or before `i`. That gives the three answers the event model leaves open,
    and gives the ones people actually mean:

      * an entry while already long is a no-op, not a second position;
      * an exit while already flat is a no-op;
      * before the first event the position is flat.

    On a bar carrying conflicting signals the priority is **exits > short entries >
    long entries**, so the most conservative reading wins. If that ever changes your
    result, the signal logic is ambiguous on that bar and the fix belongs upstream of
    here, not in the tie-break.

    `warmup` marks the first N bars NaN, which the engine reads as flat. Use it for
    the span where your indicator had insufficient data, so those bars are not scored
    as a deliberate decision to be out of the market.

    Causal by construction: the state at bar `i` reads only events at or before `i`.
    Do not shift the result yourself -- falsify applies the execution lag.

    >>> from_signals(entries=[False, True, False, False],
    ...              exits=  [False, False, False, True])
    array([0., 1., 1., 0.])
    """
    ent = np.asarray(entries)
    if ent.ndim != 1:
        raise ValueError(f"entries must be a flat array, got shape {ent.shape}")
    n = len(ent)
    if n == 0:
        return np.zeros(0, dtype=float)

    ent = _as_bool(entries, n, "entries")
    ex = _as_bool(exits, n, "exits")
    sent = _as_bool(short_entries, n, "short_entries")
    sex = _as_bool(short_exits, n, "short_exits")

    if not np.isfinite(size):
        raise ValueError(f"size must be finite, got {size}")

    # Later assignment wins, so apply in ascending priority: long, short, then the
    # two flattening events.
    target = np.zeros(n, dtype=float)
    event = np.zeros(n, dtype=bool)

    for mask, value in ((ent, float(size)), (sent, -float(size)),
                        (sex, 0.0), (ex, 0.0)):
        if mask.any():
            target[mask] = value
            event |= mask

    # Forward-fill the most recent event. `maximum.accumulate` over event indices is
    # the whole trick, and it only ever looks backwards.
    idx = np.where(event, np.arange(n), -1)
    idx = np.maximum.accumulate(idx)
    w = np.where(idx >= 0, target[idx], 0.0)

    if warmup > 0:
        w[: min(int(warmup), n)] = np.nan
    return w


def from_positions(positions, *, warmup: int = 0) -> np.ndarray:
    """A position series you already have -> a validated weight array.

    For when the port is only a type change: a pandas Series, a list, a column out of
    a DataFrame. Coerces to float and checks the shape, so a mistake surfaces here
    with a clear message instead of as a failed sweep.

    >>> from_positions([0, 1, 1, 0])
    array([0., 1., 1., 0.])
    """
    try:
        w = np.asarray(positions, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"positions is not a sequence of numbers ({exc}). Pass a list, a numpy "
            "array, or a pandas Series."
        ) from None
    if w.ndim != 1:
        raise ValueError(
            f"positions must be a flat series, got shape {w.shape}. If this is a "
            "DataFrame, select the one column you mean."
        )
    w = w.copy()
    if warmup > 0:
        w[: min(int(warmup), len(w))] = np.nan
    return w
