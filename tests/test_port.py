"""The event-model -> weight-series conversion.

These matter more than their size suggests: every one of them is a question the
entry/exit model leaves unanswered, and a port that answers one differently from the
original backtest produces a verdict about a strategy nobody runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from falsify_quant.port import from_positions, from_signals


def test_basic_entry_then_exit():
    w = from_signals(entries=[False, True, False, False],
                     exits=[False, False, False, True])
    assert list(w) == [0.0, 1.0, 1.0, 0.0]


def test_flat_before_the_first_event():
    """No event yet is not a position."""
    w = from_signals(entries=[False, False, True], exits=[False, False, False])
    assert list(w[:2]) == [0.0, 0.0]


def test_repeated_entry_while_already_long_is_a_noop():
    """The one that silently doubles a position in a hand-written port."""
    w = from_signals(entries=[True, True, True], exits=[False, False, False])
    assert list(w) == [1.0, 1.0, 1.0]
    assert w.max() == 1.0


def test_exit_while_flat_is_a_noop():
    w = from_signals(entries=[False, False], exits=[True, True])
    assert list(w) == [0.0, 0.0]


def test_exit_wins_on_a_conflicting_bar():
    """Documented tie-break: the conservative reading."""
    w = from_signals(entries=[True, True], exits=[False, True])
    assert list(w) == [1.0, 0.0]


def test_short_side():
    w = from_signals(entries=[False, False, False, False],
                     exits=[False, False, False, False],
                     short_entries=[False, True, False, False],
                     short_exits=[False, False, False, True])
    assert list(w) == [0.0, -1.0, -1.0, 0.0]


def test_flip_from_long_to_short():
    w = from_signals(entries=[True, False, False],
                     exits=[False, False, False],
                     short_entries=[False, True, False])
    assert list(w) == [1.0, -1.0, -1.0]


def test_size_scales_both_sides():
    w = from_signals(entries=[True, False], short_entries=[False, True], size=0.5)
    assert list(w) == [0.5, -0.5]


def test_warmup_is_nan_not_zero():
    """NaN and 0.0 both trade flat, but only NaN says 'no decision was possible'."""
    w = from_signals(entries=[True] * 5, warmup=3)
    assert np.isnan(w[:3]).all()
    assert not np.isnan(w[3:]).any()


def test_nan_in_a_signal_is_not_a_true_signal():
    """A float signal array with NaN warmup must not read as an entry."""
    ent = np.array([np.nan, np.nan, 1.0, 0.0])
    w = from_signals(entries=ent)
    assert list(w) == [0.0, 0.0, 1.0, 1.0]


def test_length_mismatch_is_rejected_with_a_clear_message():
    with pytest.raises(ValueError, match="same length"):
        from_signals(entries=[True, False, True], exits=[False, True])


def test_empty_input():
    assert len(from_signals(entries=[])) == 0


# --------------------------------------------------------------------------------------
# The property that the whole tool rests on
# --------------------------------------------------------------------------------------


def test_conversion_is_causal():
    """Weights up to bar k must not change when later signals are withheld.

    from_signals only ever forward-fills, so this holds by construction -- which is
    exactly the kind of claim that should be checked rather than asserted, since it is
    the property falsify would otherwise fail the user's strategy for.
    """
    rng = np.random.default_rng(3)
    n = 300
    ent = rng.random(n) < 0.05
    ex = rng.random(n) < 0.05
    full = from_signals(entries=ent, exits=ex)

    for k in (17, 50, 123, 299):
        trunc = from_signals(entries=ent[:k], exits=ex[:k])
        assert np.array_equal(trunc, full[:k]), f"conversion looked ahead at k={k}"


def test_it_survives_a_real_sweep_and_scores_causal():
    """End to end: a from_signals strategy must pass falsify's own causality test."""
    from falsify_quant.harness import sweep
    from falsify_quant.prosecute import check_causality
    from falsify_quant.spec import Bars, MarketSpec

    spec = MarketSpec(name="t", asset_class="equity", bars_per_year=252,
                      fee=0.0005, half_spread=0.0)
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 600)))
    bars = Bars(close=close, ts=np.arange(600) * 86400.0 + 1_700_000_000)

    def sma(x, n):
        out = np.full(len(x), np.nan)
        c = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1:] = (c[n:] - c[:-n]) / n
        return out

    def strategy(bars, fast=10, slow=50):
        f, s = sma(bars.close, int(fast)), sma(bars.close, int(slow))
        return from_signals(entries=f > s, exits=f < s, warmup=int(slow))

    sw = sweep(strategy, bars, spec, {"fast": [10, 20], "slow": [50, 100]})
    assert sw.n_failed == 0
    assert check_causality(sw, sw.best_index).score == 1.0


# --------------------------------------------------------------------------------------
# from_positions
# --------------------------------------------------------------------------------------


def test_from_positions_coerces_and_validates():
    assert list(from_positions([0, 1, 1, 0])) == [0.0, 1.0, 1.0, 0.0]


def test_from_positions_rejects_2d_with_a_useful_message():
    with pytest.raises(ValueError, match="one column"):
        from_positions(np.zeros((10, 2)))


def test_from_positions_accepts_a_series_like_object():
    """Anything numpy can read, including a pandas Series, without importing pandas."""
    class SeriesLike:
        def __init__(self, v): self.v = v
        def __array__(self, dtype=None, copy=None):
            a = np.asarray(self.v, dtype=dtype)
            return a

    assert list(from_positions(SeriesLike([0, 1, 0]))) == [0.0, 1.0, 0.0]


# --------------------------------------------------------------------------------------
# pandas
#
# The porting section sells this path explicitly -- "that's vectorbt's
# from_signals(entries, exits)", "a pandas column", "pd.Series(bars.close) included" --
# and it is the shape most people arriving with an existing backtest actually have.
# It had no test. falsify does not depend on pandas at runtime and should not; it is in
# the dev extra so the promise is checked rather than assumed.
# --------------------------------------------------------------------------------------


def test_from_signals_takes_the_boolean_series_vectorbt_hands_you():
    pd = pytest.importorskip("pandas")

    close = pd.Series([10.0, 11, 12, 11, 10, 9, 10, 11])
    entries, exits = close > close.shift(1), close < close.shift(1)

    w = from_signals(entries=entries, exits=exits, warmup=1)
    assert isinstance(w, np.ndarray)
    assert np.isnan(w[0])
    assert w[1] == 1.0 and w[4] == 0.0


def test_from_positions_takes_a_pandas_column():
    pd = pytest.importorskip("pandas")

    w = from_positions(pd.Series([0, 0, 1, 1, 0, 1]), warmup=2)
    assert np.array_equal(w[2:], [1.0, 1.0, 0.0, 1.0])
    assert np.isnan(w[:2]).all()


def test_the_index_is_ignored_because_bars_do_not_have_one():
    """Positional, not aligned -- and it has to be, since `bars` carries no index.

    A DatetimeIndex is what every real pandas user has, and a filtered frame can
    carry one that is neither sorted nor unique. Any of them must give the same
    answer as the plain array, or a port would silently trade a shuffled signal.
    """
    pd = pytest.importorskip("pandas")

    values = [False, True, True, False, False, True]
    plain = from_signals(entries=np.array(values), exits=~np.array(values))

    for index in (pd.date_range("2020-01-01", periods=6, freq="D"),
                  [5, 4, 3, 2, 1, 0],
                  [1, 1, 2, 2, 3, 3]):
        s = pd.Series(values, index=index)
        got = from_signals(entries=s, exits=~s)
        assert np.array_equal(got, plain, equal_nan=True), f"index {list(index)[:3]} changed it"


def test_a_series_wrapping_bars_close_round_trips():
    """The README tells pandas users to do exactly this."""
    pd = pytest.importorskip("pandas")

    close = np.array([10.0, 11, 12, 11, 10, 9])
    assert np.array_equal(pd.Series(close).to_numpy(), close)
