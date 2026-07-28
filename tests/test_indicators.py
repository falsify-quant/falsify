"""Indicator arithmetic, checked against definitions rather than against itself.

The load-bearing test in this file is `test_every_indicator_is_causal`. Everything in
`falsify.indicators` exists to be a *reference* implementation -- the corpus study reads
its conclusions about the published canon straight out of these functions -- so a leak
here would not produce an obvious failure. It would produce a study whose headline number
was quietly, defensibly wrong.

The rest checks the recursions against literal loops and the oscillators against their
definitions, because an EMA that is off by one seed value still looks exactly like an EMA.
"""

from __future__ import annotations

import numpy as np
import pytest

from falsify.indicators import (
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
    rolling_std,
    rolling_sum,
    rsi,
    shift,
    stochastic,
    true_range,
    wilder,
    williams_r,
)
from falsify.spec import Bars


def _bars(n: int = 400, seed: int = 0) -> Bars:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.012, n))
    wiggle = np.abs(rng.normal(0.0, 0.006, n)) * close
    high = close + wiggle
    low = close - np.abs(rng.normal(0.0, 0.006, n)) * close
    op = low + (high - low) * rng.random(n)
    ts = np.arange(n, dtype=np.float64) * 86400.0 + 1_600_000_000.0
    return Bars(open=op, high=high, low=low, close=close,
                volume=rng.random(n) * 1e6, ts=ts, symbol="SYNTH")


# --------------------------------------------------------------------------------------
# The one that matters
# --------------------------------------------------------------------------------------

# Every public indicator, reduced to a single bars -> array callable.
CAUSAL_CASES = {
    "rolling_sum": lambda b: rolling_sum(b.close, 20),
    "rolling_mean": lambda b: rolling_mean(b.close, 20),
    "rolling_std": lambda b: rolling_std(b.close, 20),
    "rolling_max": lambda b: rolling_max(b.high, 20),
    "rolling_min": lambda b: rolling_min(b.low, 20),
    "shift": lambda b: shift(b.close, 3),
    "ema": lambda b: ema(b.close, 21),
    "wilder": lambda b: wilder(b.close, 14),
    "true_range": lambda b: true_range(b),
    "atr": lambda b: atr(b, 14),
    "realised_vol": lambda b: realised_vol(b.close, 20),
    "rsi": lambda b: rsi(b.close, 14),
    "macd_line": lambda b: macd(b.close, 12, 26, 9)[0],
    "macd_signal": lambda b: macd(b.close, 12, 26, 9)[1],
    "macd_hist": lambda b: macd(b.close, 12, 26, 9)[2],
    "bollinger_upper": lambda b: bollinger(b.close, 20, 2.0)[1],
    "bollinger_lower": lambda b: bollinger(b.close, 20, 2.0)[2],
    "keltner_upper": lambda b: keltner(b, 20, 2.0, 10)[1],
    "stochastic_k": lambda b: stochastic(b, 14, 3)[0],
    "stochastic_d": lambda b: stochastic(b, 14, 3)[1],
    "williams_r": lambda b: williams_r(b, 14),
}


def _truncate(b: Bars, k: int) -> Bars:
    return Bars(open=b.open[:k], high=b.high[:k], low=b.low[:k], close=b.close[:k],
                volume=b.volume[:k], ts=b.ts[:k], symbol=b.symbol)


@pytest.mark.parametrize("name", sorted(CAUSAL_CASES))
@pytest.mark.parametrize("k", [80, 137, 250, 399])
def test_every_indicator_is_causal(name, k):
    """Deleting the future must not change the past.

    This is the same truncation argument the lookahead test uses on whole strategies,
    applied one level down. If an indicator reads ahead -- a centred window, a backfill,
    a full-sample normalisation -- then the value it reports at bar i changes when bars
    after i are removed. If it does not read ahead, the two runs agree exactly, to the
    bit, because they executed the same arithmetic on the same inputs.
    """
    b = _bars()
    full = CAUSAL_CASES[name](b)
    trunc = CAUSAL_CASES[name](_truncate(b, k))

    assert len(trunc) == k
    a, c = full[:k], trunc[:k]
    both_nan = np.isnan(a) & np.isnan(c)
    assert np.array_equal(np.isnan(a), np.isnan(c)), f"{name}: warmup moved when truncated"
    np.testing.assert_array_equal(a[~both_nan], c[~both_nan], err_msg=f"{name} reads ahead")


def test_shift_refuses_to_look_forward():
    with pytest.raises(ValueError, match="future"):
        shift(np.arange(10.0), -1)


def test_shift_moves_values_forward():
    x = np.arange(5.0)
    out = shift(x, 2)
    assert np.isnan(out[:2]).all()
    np.testing.assert_array_equal(out[2:], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(shift(x, 0), x)


# --------------------------------------------------------------------------------------
# Recursions against literal loops
# --------------------------------------------------------------------------------------


def _loop_recursive(x, alpha, n):
    """The recursion written the obvious slow way, as the thing lfilter must reproduce."""
    out = np.full(len(x), np.nan)
    out[n - 1] = np.mean(x[:n])
    for t in range(n, len(x)):
        out[t] = alpha * x[t] + (1.0 - alpha) * out[t - 1]
    return out


@pytest.mark.parametrize("n", [2, 5, 14, 26, 50])
def test_ema_matches_the_loop(n):
    x = _bars().close
    np.testing.assert_allclose(ema(x, n), _loop_recursive(x, 2.0 / (n + 1.0), n), rtol=1e-12)


@pytest.mark.parametrize("n", [2, 14, 21])
def test_wilder_matches_the_loop(n):
    x = _bars().close
    np.testing.assert_allclose(wilder(x, n), _loop_recursive(x, 1.0 / n, n), rtol=1e-12)


def test_wilder_is_a_slower_ema_than_its_period_suggests():
    """Wilder(n) has the same alpha as EMA(2n-1). This is why RSI(14) feels like a month.

    They are the same filter with different seeds, so they agree only asymptotically:
    alpha = 1/14 has a half-life of about nine bars, and the seeds are thirteen bars
    apart. The gap starts near 0.3% and decays by half every nine bars; two hundred bars
    in it is down to parts per hundred million.
    """
    x = _bars().close
    np.testing.assert_allclose(wilder(x, 14)[200:], ema(x, 27)[200:], rtol=1e-7)
    assert abs(wilder(x, 14)[30] - ema(x, 27)[30]) > 1e-6  # ... but not right after warmup


def test_smoothers_do_not_emit_before_warmup():
    x = _bars(100).close
    for f, n in [(ema, 21), (wilder, 14)]:
        out = f(x, n)
        assert np.isnan(out[: n - 1]).all()
        assert np.isfinite(out[n - 1:]).all()
        assert out[n - 1] == pytest.approx(np.mean(x[:n]))


def test_window_longer_than_history_is_all_nan():
    x = np.arange(10.0)
    for out in (rolling_mean(x, 20), rolling_std(x, 20), rolling_max(x, 20),
                ema(x, 20), wilder(x, 20), rsi(x, 20)):
        assert np.isnan(out).all()


# --------------------------------------------------------------------------------------
# Trailing windows against naive loops
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 3, 20])
def test_rolling_extremes_match_naive(n):
    x = _bars(120).close
    want_hi = np.full(len(x), np.nan)
    want_lo = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        want_hi[i] = x[i - n + 1: i + 1].max()
        want_lo[i] = x[i - n + 1: i + 1].min()
    np.testing.assert_array_equal(rolling_max(x, n), want_hi)
    np.testing.assert_array_equal(rolling_min(x, n), want_lo)


def test_rolling_window_includes_the_current_bar():
    """The convention breakout strategies have to shift around. Pinned deliberately."""
    x = np.array([1.0, 5.0, 2.0, 9.0])
    np.testing.assert_array_equal(rolling_max(x, 2), [np.nan, 5.0, 5.0, 9.0])
    # A breakout compared against an unshifted channel can never fire.
    assert not np.any(x > rolling_max(x, 2))


def test_rolling_mean_and_std_agree_with_numpy():
    x = _bars(200).close
    n = 30
    for i in (n - 1, 100, 199):
        w = x[i - n + 1: i + 1]
        assert rolling_mean(x, n)[i] == pytest.approx(w.mean())
        assert rolling_std(x, n)[i] == pytest.approx(w.std(ddof=1))


def test_rolling_sum_is_mean_times_n():
    x = _bars(80).close
    np.testing.assert_allclose(rolling_sum(x, 12)[11:], rolling_mean(x, 12)[11:] * 12, rtol=1e-12)


def test_one_nan_does_not_poison_every_later_window():
    """The cumulative-sum trick's sharp edge, pinned.

    A single missing print used to turn the entire remainder of the series into NaN,
    because the prefix sums it was built from were themselves NaN from that point on.
    That reads as a longer warmup, not as a bug, which is why it needs a test.
    """
    x = np.arange(1.0, 21.0)
    x[4] = np.nan
    for f in (rolling_sum, rolling_mean, rolling_std):
        out = f(x, 3)
        assert np.isnan(out[4:7]).all(), f"{f.__name__}: windows touching the gap must be NaN"
        assert np.isfinite(out[7:]).all(), f"{f.__name__}: windows past the gap were poisoned"

    np.testing.assert_allclose(rolling_mean(x, 3)[7:], np.arange(7.0, 20.0))


def test_smoothers_skip_a_leading_nan_run():
    """A return series has no return on bar zero. That must not blank the whole output."""
    x = np.concatenate([[np.nan], np.arange(1.0, 41.0)])
    for f in (ema, wilder):
        out = f(x, 10)
        assert np.isnan(out[:10]).all()  # one NaN bar plus a ten-bar seed window
        assert np.isfinite(out[10:]).all()
    assert ema(x, 10)[10] == pytest.approx(np.mean(np.arange(1.0, 11.0)))


def test_atr_works_without_ohlc():
    """The close-only fallback puts a NaN in bar zero of the true range."""
    c = _bars(200).close
    out = atr(Bars(close=c), 14)
    assert np.isfinite(out[15:]).all()
    assert np.nanmedian(out) > 0.0


# --------------------------------------------------------------------------------------
# RSI
# --------------------------------------------------------------------------------------


def test_rsi_pins_at_100_when_nothing_falls():
    x = np.cumsum(np.full(60, 1.0)) + 100.0
    out = rsi(x, 14)
    assert np.isnan(out[:14]).all()
    np.testing.assert_allclose(out[14:], 100.0)


def test_rsi_pins_at_0_when_nothing_rises():
    x = 200.0 - np.cumsum(np.full(60, 1.0))
    np.testing.assert_allclose(rsi(x, 14)[14:], 0.0)


def test_rsi_of_a_flat_market_is_50_not_100():
    """No gains and no losses is not maximum strength. Documented in `rsi`."""
    np.testing.assert_allclose(rsi(np.full(60, 100.0), 14)[14:], 50.0)


def test_rsi_stays_in_range_and_starts_where_promised():
    out = rsi(_bars(500).close, 14)
    assert np.isnan(out[:14]).all()
    assert np.isfinite(out[14:]).all()
    assert out[14:].min() >= 0.0 and out[14:].max() <= 100.0


def test_rsi_matches_the_definition_at_its_seed():
    """At the seed bar, Wilder's smoothing is just the simple average of the first n."""
    x = _bars(60).close
    d = np.diff(x)
    up, down = np.maximum(d, 0.0)[:14].mean(), np.maximum(-d, 0.0)[:14].mean()
    assert rsi(x, 14)[14] == pytest.approx(100.0 - 100.0 / (1.0 + up / down))


# --------------------------------------------------------------------------------------
# Range, channels, oscillators
# --------------------------------------------------------------------------------------


def test_true_range_by_hand():
    b = Bars(high=np.array([10.0, 12.0, 11.0]),
             low=np.array([9.0, 11.5, 8.0]),
             close=np.array([9.5, 11.8, 8.5]))
    #  bar 0: no previous close -> plain range          10.0 - 9.0  = 1.0
    #  bar 1: gapped up; |high - prev close| = 12 - 9.5             = 2.5
    #  bar 2: gapped down; |low - prev close| = |8 - 11.8|          = 3.8
    np.testing.assert_allclose(true_range(b), [1.0, 2.5, 3.8])


def test_true_range_falls_back_to_close_to_close():
    b = Bars(close=np.array([10.0, 11.0, 9.0]))
    tr = true_range(b)
    assert np.isnan(tr[0])
    np.testing.assert_allclose(tr[1:], [1.0, 2.0])


def test_atr_is_wilder_of_true_range():
    b = _bars()
    np.testing.assert_allclose(atr(b, 14), wilder(true_range(b), 14), equal_nan=True)


def test_bollinger_bands_are_symmetric_about_the_mean():
    c = _bars().close
    mid, up, lo = bollinger(c, 20, 2.0)
    np.testing.assert_allclose(mid, rolling_mean(c, 20), equal_nan=True)
    np.testing.assert_allclose(up - mid, mid - lo, equal_nan=True)
    np.testing.assert_allclose((up - lo)[19:], 4.0 * rolling_std(c, 20)[19:])


def test_keltner_width_comes_from_atr_not_deviation():
    b = _bars()
    mid, up, lo = keltner(b, 20, 2.0, 10)
    np.testing.assert_allclose((up - lo)[25:], 4.0 * atr(b, 10)[25:])
    np.testing.assert_allclose(mid, ema(b.close, 20), equal_nan=True)


def test_stochastic_endpoints():
    n = 5
    close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1.0])
    b = Bars(high=np.maximum(close, 5.0), low=np.minimum(close, 1.0), close=close)
    k, _ = stochastic(b, n, smooth=1)
    assert k[4] == pytest.approx(100.0)  # close sits at the top of the range
    assert k[5] == pytest.approx(0.0)  # and then at the bottom


def test_stochastic_of_a_flat_range_is_50():
    flat = np.full(20, 7.0)
    k, _ = stochastic(Bars(high=flat, low=flat, close=flat), 14, smooth=3)
    np.testing.assert_allclose(k[13:], 50.0)


def test_williams_r_is_the_stochastic_shifted():
    b = _bars()
    k, _ = stochastic(b, 14, smooth=1)
    np.testing.assert_allclose(williams_r(b, 14), k - 100.0, equal_nan=True)
    r = williams_r(b, 14)
    assert np.nanmin(r) >= -100.0 and np.nanmax(r) <= 0.0


def test_macd_signal_warmup_compounds():
    """The histogram is not honest until both EMAs and the signal EMA have filled."""
    line, sig, hist = macd(_bars().close, 12, 26, 9)
    assert np.isnan(line[:25]).all() and np.isfinite(line[25:]).all()
    first_signal = int(np.argmax(np.isfinite(sig)))
    assert first_signal == 25 + 8  # slow warmup, then nine more bars of signal EMA
    assert np.isnan(hist[:first_signal]).all()


def test_realised_vol_recovers_a_known_sigma():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0, 0.02, 20_000)
    c = 100.0 * np.cumprod(1.0 + r)
    assert np.nanmedian(realised_vol(c, 250)) == pytest.approx(0.02, rel=0.05)


# --------------------------------------------------------------------------------------
# Calendar
# --------------------------------------------------------------------------------------


def test_calendar_fields_on_known_dates():
    ts = np.array([d.astype("datetime64[s]").astype(np.int64) for d in
                   np.array(["2024-02-01", "2024-02-29", "2023-02-28", "2024-12-31"],
                            dtype="datetime64[D]")], dtype=np.float64)
    np.testing.assert_array_equal(day_of_month(ts), [1, 29, 28, 31])
    np.testing.assert_array_equal(days_in_month(ts), [29, 29, 28, 31])  # 2024 is a leap year
