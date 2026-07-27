"""Behavioural checks on the six prosecution tests.

These assert what each test must say in situations where the right answer is known by
construction, including the two the library gets asked about most: PBO under a true null
(it must land near 0.5, not near 0) and the plateau case that textbook PBO scores as a
failure.
"""

from __future__ import annotations

import numpy as np
import pytest

from falsify.harness import Sweep
from falsify.prosecute import (
    Finding,
    check_causality,
    check_costs,
    check_pbo,
    check_regime,
)
from falsify.spec import Bars, MarketSpec
from falsify.stats import sharpe_columns

SPEC = MarketSpec(name="test", asset_class="crypto", bars_per_year=365,
                  fee=0.0005, half_spread=0.0)


def make_sweep(returns: np.ndarray, bars: Bars | None = None, strategy=None) -> Sweep:
    """Wrap a (T, N) returns matrix as a Sweep so the statistics can be driven directly."""
    T, N = returns.shape
    if bars is None:
        close = 100.0 * np.cumprod(1.0 + np.concatenate([[0.0], np.zeros(T - 1)]))
        bars = Bars(close=close, ts=np.arange(T) * 86400.0 + 1_700_000_000)
    return Sweep(
        params=[{"i": float(i)} for i in range(N)],
        returns=returns.astype(np.float32),
        sharpes=sharpe_columns(returns),
        gross=returns.sum(axis=0),
        churn=np.ones(N),
        failed=np.zeros(N, dtype=bool),
        bars=bars,
        spec=SPEC,
        strategy=strategy or (lambda b, **p: np.zeros(len(b))),
        grid={"i": list(range(N))},
    )


# ----------------------------------------------------------------------------------
# PBO / CSCV
# ----------------------------------------------------------------------------------


def test_pbo_under_a_true_null_is_about_one_half():
    """Exchangeable worthless strategies: the in-sample winner must rank at random.

    A PBO near 0 here would mean the statistic is finding structure in noise, which is
    the failure that would make every "selection holds up" verdict worthless.
    """
    rng = np.random.default_rng(0)
    sweep = make_sweep(rng.normal(0.0, 0.01, (2000, 60)))

    f = check_pbo(sweep)
    assert f.detail["pbo"] == pytest.approx(0.5, abs=0.2)
    assert f.score < 0.4


def test_pbo_near_zero_when_one_strategy_genuinely_dominates():
    rng = np.random.default_rng(1)
    m = rng.normal(0.0, 0.01, (2000, 40))
    m[:, 7] += 0.004  # a real, large, persistent edge in exactly one column

    f = check_pbo(make_sweep(m))
    assert f.detail["pbo"] < 0.05
    assert f.score > 0.9


def test_pbo_plateau_is_not_scored_as_failure():
    """Every variant carries the same real edge.

    The ranking among them is then pure noise, so textbook PBO climbs toward 0.5 and
    calls it overfitting -- while the thing that matters, "does the variant I picked
    still make money out of sample", is emphatically yes. falsify measures the absolute
    out-of-sample level alongside the rank so a plateau reads as the good case it is.
    """
    rng = np.random.default_rng(2)
    m = rng.normal(0.0015, 0.01, (2000, 40))  # all columns identical in distribution

    f = check_pbo(make_sweep(m))
    assert f.detail["pbo"] > 0.2                      # rank is uninformative
    assert f.detail["oos_profitable_rate"] > 0.9      # but the pick still makes money
    assert f.score > 0.8                              # and the finding must reflect that
    assert "flat parameter surface" in f.headline.lower()


def test_pbo_handles_a_grid_too_small_to_rank():
    rng = np.random.default_rng(3)
    f = check_pbo(make_sweep(rng.normal(0, 0.01, (500, 2))))
    assert f.score == pytest.approx(0.5)


# ----------------------------------------------------------------------------------
# Causality
# ----------------------------------------------------------------------------------


def _bars(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + np.concatenate([[0.0], rng.normal(0, 0.01, n - 1)]))
    return Bars(close=close, ts=np.arange(n) * 86400.0 + 1_700_000_000)


def test_causality_passes_a_trailing_strategy():
    bars = _bars()

    def trailing(b, **_):
        c = np.asarray(b.close)
        out = np.zeros(len(c))
        out[20:] = (c[20:] > c[:-20]).astype(float)
        return out

    sweep = make_sweep(np.zeros((len(bars), 1)), bars=bars, strategy=trailing)
    f = check_causality(sweep, 0)
    assert f.score == 1.0
    assert not f.fatal


def test_causality_catches_a_future_reading_strategy():
    bars = _bars()

    def peeks(b, **_):
        c = np.asarray(b.close)
        out = np.zeros(len(c))
        out[:-1] = (c[1:] > c[:-1]).astype(float)  # reads the next bar
        return out

    sweep = make_sweep(np.zeros((len(bars), 1)), bars=bars, strategy=peeks)
    f = check_causality(sweep, 0)
    assert f.fatal
    assert f.score == 0.0
    assert "LOOKAHEAD" in f.headline


def test_causality_catches_full_sample_scaling_against_a_threshold():
    """The subtle one: normalising by a whole-sample statistic, then comparing to a
    constant. The scale factor carries future information into every decision."""
    bars = _bars()

    def scaled(b, **_):
        c = np.asarray(b.close, float)
        z = (c - c.mean()) / c.std()
        return (z > 0.5).astype(float)

    sweep = make_sweep(np.zeros((len(bars), 1)), bars=bars, strategy=scaled)
    assert check_causality(sweep, 0).fatal


def test_causality_flags_a_wrong_length_return():
    bars = _bars()
    sweep = make_sweep(np.zeros((len(bars), 1)), bars=bars,
                       strategy=lambda b, **_: np.zeros(len(b) // 2))
    f = check_causality(sweep, 0)
    assert f.fatal
    assert "one weight per bar" in f.headline


def test_causality_notes_a_strategy_that_never_trades():
    bars = _bars()
    sweep = make_sweep(np.zeros((len(bars), 1)), bars=bars,
                       strategy=lambda b, **_: np.zeros(len(b)))
    f = check_causality(sweep, 0)
    assert f.detail.get("always_flat") is True


# ----------------------------------------------------------------------------------
# Costs
# ----------------------------------------------------------------------------------


def test_costs_reports_negative_edge_without_a_ratio():
    """A negative gross edge has no meaningful cost ratio. An earlier version divided by
    an epsilon and printed 'you are paying 9093785952.7x your edge'."""
    bars = _bars(seed=4)

    def alternating(b, **_):
        w = np.zeros(len(b))
        w[::2] = 1.0
        return w

    sweep = make_sweep(np.full((len(bars), 1), -0.001), bars=bars, strategy=alternating)
    f = check_costs(sweep, 0)
    if f.detail.get("edge_per_turnover_bps", 0.0) <= 0:
        assert "before costs" in f.headline
        assert "x your edge" not in f.headline
        assert f.score == 0.0


def test_costs_flags_a_variant_that_never_trades():
    bars = _bars(seed=5)
    sweep = make_sweep(np.zeros((len(bars), 1)), bars=bars,
                       strategy=lambda b, **_: np.zeros(len(b)))
    f = check_costs(sweep, 0)
    assert f.score == 0.0
    assert "never trades" in f.headline


# ----------------------------------------------------------------------------------
# Regime
# ----------------------------------------------------------------------------------


def test_regime_flags_a_single_lucky_window():
    T = 1200
    r = np.zeros((T, 1))
    r[200:260, 0] = 0.02  # the entire result, earned in one stretch
    f = check_regime(make_sweep(r), 0)
    assert f.score < 0.6
    assert f.detail["periods_profitable"] <= 2


def test_regime_rewards_evenly_spread_pnl():
    rng = np.random.default_rng(6)
    r = (rng.normal(0.0012, 0.004, (1200, 1)))
    f = check_regime(make_sweep(r), 0)
    assert f.detail["consistency"] == 1.0
    assert f.score > 0.7


# ----------------------------------------------------------------------------------
# Finding invariants
# ----------------------------------------------------------------------------------


def test_finding_fails_closed_on_nan():
    """An uncomputable test has established nothing and must never read as a pass."""
    assert Finding("x", "X", float("nan"), "").score == 0.0
    assert Finding("x", "X", 2.5, "").score == 1.0
    assert Finding("x", "X", -3.0, "").score == 0.0
