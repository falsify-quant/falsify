"""Hand-computed checks on the fill simulator.

Every cost verdict falsify has produced rests on this module -- the IBKR bot scored
0/100 largely because its edge per unit of turnover came out below its cost per unit of
turnover, and that comparison is only as trustworthy as the accounting underneath it.

So: small cases, computed by hand in the comments, asserted exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from falsify.sim import simulate
from falsify.spec import Bars, MarketSpec

FREE = MarketSpec(name="frictionless", asset_class="crypto", bars_per_year=365,
                  fee=0.0, half_spread=0.0)
FLAT_10BPS = MarketSpec(name="10bps", asset_class="crypto", bars_per_year=365,
                        fee=0.0010, half_spread=0.0)


def bars_from_returns(rets, p0=100.0):
    """Build a price series whose bar returns are exactly `rets` (after a leading 0)."""
    close = p0 * np.cumprod(1.0 + np.concatenate([[0.0], np.asarray(rets, float)]))
    n = len(close)
    return Bars(close=close, ts=np.arange(n) * 86400.0 + 1_700_000_000, symbol="T")


def const_bars(n=60, r=0.01):
    return bars_from_returns(np.full(n - 1, r))


# ----------------------------------------------------------------------------------
# Execution timing -- the convention everything else depends on
# ----------------------------------------------------------------------------------


def test_weight_earns_the_following_bar():
    """w[t] is decided at the close of bar t and must earn bar t+1's return, not t's.

    Booking the same bar's return is the classic free-money bug: it lets a strategy
    that reacts to a move collect the move it reacted to.
    """
    bars = const_bars(10, r=0.01)
    w = np.zeros(len(bars))
    w[3] = 1.0  # long for exactly one bar, decided at the close of bar 3

    res = simulate(bars, w, FREE)

    assert res.gross[3] == pytest.approx(0.0)   # the bar it was decided on pays nothing
    assert res.gross[4] == pytest.approx(0.01)  # the next bar pays
    assert res.gross[5] == pytest.approx(0.0)


def test_a_prescient_strategy_cannot_earn_the_current_bar():
    """Setting w[t] from r[t] earns nothing, because the lag is applied by the engine.

    If this ever passes with a profit, the lag has been removed and every backtest in
    the library is invalid.
    """
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.02, 200)
    bars = bars_from_returns(rets)
    w = np.sign(bars.returns)  # "knows" the current bar

    res = simulate(bars, w, FREE)
    # Correlation with the bar it supposedly predicted should be nil.
    assert abs(np.corrcoef(res.gross[2:], bars.returns[2:])[0, 1]) < 0.25


def test_genuine_foresight_earns_absolute_returns():
    """w[t] = sign(r[t+1]) is real foresight and must collect |r| on every bar."""
    rng = np.random.default_rng(1)
    rets = rng.normal(0, 0.02, 200)
    bars = bars_from_returns(rets)

    w = np.zeros(len(bars))
    w[:-1] = np.sign(bars.returns[1:])  # decided at t, knowing t+1

    res = simulate(bars, w, FREE)
    assert np.all(res.gross[1:] >= -1e-15)
    assert res.gross_return == pytest.approx(np.sum(np.abs(bars.returns[1:])))


def test_extra_lag_shifts_by_whole_bars():
    bars = const_bars(12, r=0.01)
    w = np.zeros(len(bars))
    w[3] = 1.0

    assert simulate(bars, w, FREE, extra_lag=0).gross[4] == pytest.approx(0.01)
    assert simulate(bars, w, FREE, extra_lag=1).gross[5] == pytest.approx(0.01)
    assert simulate(bars, w, FREE, extra_lag=2).gross[6] == pytest.approx(0.01)


# ----------------------------------------------------------------------------------
# Cost accounting, computed by hand
# ----------------------------------------------------------------------------------


def test_costs_hand_computed():
    """close = 100, 110, 121, 121  ->  r = 0, 0.1, 0.1, 0
       w    = 1, 1, 0, 0           ->  p = lag(w) = 0, 1, 1, 0
       gross            = 0, 0.10, 0.10, 0
       turnover = |p - lag(p)|  = 0, 1, 0, 1
       net at 10bps     = 0, 0.099, 0.10, -0.001
    """
    bars = Bars(close=np.array([100.0, 110.0, 121.0, 121.0]),
                ts=np.arange(4) * 86400.0 + 1_700_000_000)
    w = np.array([1.0, 1.0, 0.0, 0.0])

    res = simulate(bars, w, FLAT_10BPS)

    assert res.gross == pytest.approx([0.0, 0.10, 0.10, 0.0])
    assert res.turnover == pytest.approx([0.0, 1.0, 0.0, 1.0])
    assert res.net == pytest.approx([0.0, 0.099, 0.10, -0.001])
    assert res.total_cost == pytest.approx(0.002)


def test_flip_long_to_short_costs_double():
    """Going from fully long to fully short moves 200% of capital and pays twice."""
    bars = const_bars(8, r=0.0)
    w = np.array([1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])

    res = simulate(bars, w, FLAT_10BPS)
    assert res.turnover.max() == pytest.approx(2.0)
    assert res.total_cost == pytest.approx(2.0 * 0.0010 + 1.0 * 0.0010)  # entry + flip


def test_cost_is_linear_in_the_rate():
    """Net return must be exactly linear in the cost rate -- the breakeven calculation
    in test_costs() relies on it being exact rather than searched."""
    rng = np.random.default_rng(2)
    bars = bars_from_returns(rng.normal(0, 0.01, 300))
    w = np.sign(rng.normal(0, 1, len(bars)))

    a = simulate(bars, w, FLAT_10BPS.scaled(1.0))
    b = simulate(bars, w, FLAT_10BPS.scaled(3.0))
    churn = float(np.sum(a.turnover))

    assert b.net_return == pytest.approx(a.net_return - 2 * 0.0010 * churn)


def test_edge_per_turnover_is_the_breakeven_rate():
    """Charging exactly the edge per unit of turnover must leave net P&L at zero."""
    rng = np.random.default_rng(3)
    bars = bars_from_returns(rng.normal(0.0005, 0.01, 400))
    w = (rng.normal(0, 1, len(bars)) > 0).astype(float)

    base = simulate(bars, w, FREE)
    breakeven = base.edge_per_turnover

    spec = MarketSpec(name="breakeven", asset_class="crypto", bars_per_year=365,
                      fee=breakeven, half_spread=0.0)
    assert simulate(bars, w, spec).net_return == pytest.approx(0.0, abs=1e-12)


# ----------------------------------------------------------------------------------
# Carry
# ----------------------------------------------------------------------------------


def test_equity_borrow_charges_shorts_only():
    bars = const_bars(6, r=0.0)
    spec = MarketSpec(name="borrow", asset_class="equity", bars_per_year=252,
                      carry=0.001, carry_on="short")

    long_only = simulate(bars, np.ones(len(bars)), spec)
    short_only = simulate(bars, -np.ones(len(bars)), spec)

    assert long_only.total_cost == pytest.approx(0.0)
    assert short_only.total_cost > 0.0


def test_perp_funding_charges_both_sides():
    bars = const_bars(6, r=0.0)
    spec = MarketSpec(name="funding", asset_class="crypto", bars_per_year=365 * 24,
                      carry=0.001, carry_on="abs")

    long_cost = simulate(bars, np.ones(len(bars)), spec).total_cost
    short_cost = simulate(bars, -np.ones(len(bars)), spec).total_cost

    assert long_cost > 0
    assert long_cost == pytest.approx(short_cost)


def test_retiming_scales_carry_but_not_fees():
    """A six-hour bar owes six times an hourly bar's funding; the fee per trade is
    unchanged by how long a bar happens to last."""
    hourly = MarketSpec(name="h", asset_class="crypto", bars_per_year=365.25 * 24,
                        fee=0.0005, half_spread=0.0001, carry=0.00002)
    six_hourly = hourly.at_bars_per_year(365.25 * 4)

    assert six_hourly.fee == pytest.approx(hourly.fee)
    assert six_hourly.half_spread == pytest.approx(hourly.half_spread)
    assert six_hourly.carry == pytest.approx(hourly.carry * 6.0)


# ----------------------------------------------------------------------------------
# Warmup and validation
# ----------------------------------------------------------------------------------


def test_nan_weights_are_treated_as_flat():
    bars = const_bars(10, r=0.01)
    w = np.full(len(bars), np.nan)
    w[5:] = 1.0

    res = simulate(bars, w, FREE)
    assert res.position[:5] == pytest.approx(0.0)
    assert res.gross[:6] == pytest.approx(0.0)
    assert res.gross[6] == pytest.approx(0.01)


def test_infinite_weights_are_rejected():
    bars = const_bars(10)
    w = np.zeros(len(bars))
    w[2] = np.inf
    with pytest.raises(ValueError, match="inf"):
        simulate(bars, w, FREE)


def test_wrong_length_weights_are_rejected():
    bars = const_bars(10)
    with pytest.raises(ValueError, match="shape"):
        simulate(bars, np.ones(5), FREE)


def test_equity_curve_compounds():
    bars = const_bars(4, r=0.0)
    w = np.zeros(len(bars))
    res = simulate(bars, w, FREE)
    assert res.equity == pytest.approx(np.ones(len(bars)))
