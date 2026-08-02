"""Verify the Sharpe statistics against references rather than against themselves.

`selftest.py` checks that falsify *behaves* correctly -- noise scores low, real signal
scores high. That is necessary and not sufficient: a wrong constant in the deflation
formula would leave every one of those cases passing while making every reported number
subtly wrong, and the entire product proposition is "trust this number".

So these tests check the arithmetic against things that are true independently of the
implementation: Monte Carlo simulation of the quantity being approximated, closed-form
values, and self-consistency round trips.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from falsify_quant.stats import (
    annualise,
    deflated_sharpe,
    expected_max_sharpe,
    max_drawdown,
    min_track_record_length,
    probabilistic_sharpe,
    sharpe,
    sharpe_columns,
)


# ----------------------------------------------------------------------------------
# Sharpe
# ----------------------------------------------------------------------------------


def test_sharpe_hand_computed():
    r = np.array([0.01, -0.02, 0.03, 0.00, 0.01])
    expected = np.mean(r) / np.std(r, ddof=1)
    assert sharpe(r) == pytest.approx(expected)


def test_sharpe_is_scale_invariant():
    """Doubling every return doubles mean and std, so the ratio must not move."""
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 500)
    assert sharpe(r) == pytest.approx(sharpe(r * 2.0))


def test_sharpe_of_constant_series_is_zero():
    assert sharpe(np.full(100, 0.01)) == 0.0


def test_sharpe_columns_matches_scalar():
    rng = np.random.default_rng(1)
    m = rng.normal(0.0005, 0.01, (400, 7))
    got = sharpe_columns(m)
    for j in range(m.shape[1]):
        assert got[j] == pytest.approx(sharpe(m[:, j]))


def test_annualise():
    assert annualise(0.05, 252) == pytest.approx(0.05 * np.sqrt(252))


# ----------------------------------------------------------------------------------
# Expected maximum Sharpe -- checked against simulation of the thing it approximates
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [5, 10, 50, 200, 1000])
def test_expected_max_sharpe_matches_monte_carlo(n):
    """The formula approximates E[max of n draws]. So draw them and take the max.

    This is the single most load-bearing number in the library: it is the benchmark a
    backtest has to beat, and if it is wrong then every Deflated Sharpe is wrong in the
    same direction. Bailey & Lopez de Prado's two-term expression is an approximation,
    so a few percent of disagreement is expected -- a wrong constant would not be a few
    percent.
    """
    rng = np.random.default_rng(42)
    draws = rng.standard_normal((40_000, n))
    empirical = float(np.mean(np.max(draws, axis=1)))
    formula = expected_max_sharpe(n, 1.0)
    assert formula == pytest.approx(empirical, rel=0.05)


def test_expected_max_sharpe_scales_with_sqrt_variance():
    """It is sqrt(V) times a shape term, so quadrupling V must double the result."""
    a = expected_max_sharpe(100, 1.0)
    b = expected_max_sharpe(100, 4.0)
    assert b == pytest.approx(2.0 * a)


def test_expected_max_sharpe_grows_with_trials():
    vals = [expected_max_sharpe(n, 1.0) for n in (2, 10, 100, 1000, 10_000)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_expected_max_sharpe_degenerate_cases():
    assert expected_max_sharpe(1, 1.0) == 0.0  # one trial is not a maximum
    assert expected_max_sharpe(0, 1.0) == 0.0
    assert expected_max_sharpe(100, 0.0) == 0.0  # no dispersion, no luck to extract


# ----------------------------------------------------------------------------------
# Probabilistic Sharpe -- invariants that catch sign errors
# ----------------------------------------------------------------------------------


def test_psr_is_half_at_its_own_sharpe():
    """Benchmarking a track record against itself is a coin flip, exactly."""
    rng = np.random.default_rng(3)
    r = rng.normal(0.001, 0.01, 1000)
    assert probabilistic_sharpe(r, benchmark_sr=sharpe(r)) == pytest.approx(0.5, abs=1e-9)


def test_psr_rises_with_sample_length():
    """Same Sharpe, more evidence, more confidence."""
    rng = np.random.default_rng(4)
    base = rng.normal(0.0008, 0.01, 200)
    short = probabilistic_sharpe(base)
    long = probabilistic_sharpe(np.tile(base, 10))  # identical moments, 10x the data
    assert long > short


def test_psr_penalises_negative_skew():
    """The one that matters most.

    Negative skew is the signature of a strategy that collects premiums and blows up.
    A sign error on the skew term would make falsify *reward* exactly that shape, which
    is the most expensive mistake this library could make. Two series with matched mean
    and standard deviation, opposite skew: the left-tailed one must score lower.
    """
    rng = np.random.default_rng(5)
    g = rng.standard_gamma(2.0, 4000)
    right = (g - g.mean()) / g.std()          # positive skew
    left = -right                              # same mean and sd, mirrored

    assert sps.skew(left) < 0 < sps.skew(right)
    scale = 0.01
    mu = 0.0005
    r_right = mu + scale * right
    r_left = mu + scale * left
    assert sharpe(r_right) == pytest.approx(sharpe(r_left), rel=1e-9)

    assert probabilistic_sharpe(r_left) < probabilistic_sharpe(r_right)


def test_psr_penalises_fat_tails():
    """Matched Sharpe, heavier tails -> less confidence."""
    rng = np.random.default_rng(6)
    thin = rng.standard_normal(4000)
    fat = rng.standard_t(4, 4000)
    thin = (thin - thin.mean()) / thin.std()
    fat = (fat - fat.mean()) / fat.std()
    assert sps.kurtosis(fat, fisher=False) > sps.kurtosis(thin, fisher=False)

    r_thin, r_fat = 0.0005 + 0.01 * thin, 0.0005 + 0.01 * fat
    assert sharpe(r_thin) == pytest.approx(sharpe(r_fat), rel=1e-9)
    assert probabilistic_sharpe(r_fat) < probabilistic_sharpe(r_thin)


def test_psr_matches_closed_form_for_gaussian():
    """For skew 0 and kurtosis 3 the estimator variance is 1 + SR^2/2, so the whole
    statistic collapses to a normal CDF that can be written out by hand."""
    rng = np.random.default_rng(7)
    r = rng.standard_normal(3000) * 0.01 + 0.0006
    sr = sharpe(r)
    skew = float(sps.skew(r, bias=False))
    kurt = float(sps.kurtosis(r, fisher=False, bias=False))
    var = 1.0 - skew * sr + 0.25 * (kurt - 1.0) * sr**2
    expected = float(sps.norm.cdf(sr * np.sqrt(len(r) - 1) / np.sqrt(var)))
    assert probabilistic_sharpe(r) == pytest.approx(expected, rel=1e-9)


def test_psr_degenerate_series_returns_half():
    """A flat line is not evidence of skill, and must not read as certainty."""
    assert probabilistic_sharpe(np.zeros(500)) == 0.5
    assert probabilistic_sharpe(np.full(500, 0.01)) == 0.5


# ----------------------------------------------------------------------------------
# Deflated Sharpe
# ----------------------------------------------------------------------------------


def test_dsr_falls_as_trial_count_rises():
    rng = np.random.default_rng(8)
    r = rng.normal(0.0009, 0.01, 2000)
    v = 0.0004
    vals = [deflated_sharpe(r, n, v)[0] for n in (2, 50, 1000, 100_000)]
    assert all(b <= a for a, b in zip(vals, vals[1:]))


def test_dsr_equals_psr_when_nothing_was_searched():
    rng = np.random.default_rng(9)
    r = rng.normal(0.0009, 0.01, 1500)
    dsr, sr0 = deflated_sharpe(r, 1, 0.001)
    assert sr0 == 0.0
    assert dsr == pytest.approx(probabilistic_sharpe(r, 0.0))


def test_dsr_on_best_of_many_noise_strategies_is_not_confident():
    """Pick the luckiest of 200 worthless strategies. Deflation must not bless it.

    This is the exact situation the statistic exists for, so it is worth checking end to
    end rather than trusting the algebra.
    """
    rng = np.random.default_rng(10)
    trials = rng.normal(0.0, 0.01, (1500, 200))
    srs = sharpe_columns(trials)
    best = int(np.argmax(srs))

    plain = probabilistic_sharpe(trials[:, best])
    dsr, _ = deflated_sharpe(trials[:, best], 200, float(np.var(srs, ddof=1)))

    assert plain > 0.9      # naively it looks significant
    assert dsr < 0.6        # deflated, it does not


# ----------------------------------------------------------------------------------
# Minimum track record length -- round trip against PSR
# ----------------------------------------------------------------------------------


def test_mintrl_round_trips_through_psr():
    """MinTRL answers "how many observations to reach `confidence`". Feed that many
    observations with the same moments back into PSR and it must return `confidence`.

    A closed loop through two separately written formulas -- if either has a wrong term,
    they will not agree.
    """
    rng = np.random.default_rng(11)
    r = rng.normal(0.0007, 0.01, 4000)
    conf = 0.95

    t_star = min_track_record_length(r, benchmark_sr=0.0, confidence=conf)
    assert np.isfinite(t_star)

    sr = sharpe(r)
    skew = float(sps.skew(r, bias=False))
    kurt = float(sps.kurtosis(r, fisher=False, bias=False))
    var = 1.0 - skew * sr + 0.25 * (kurt - 1.0) * sr**2
    z = sr * np.sqrt(t_star - 1) / np.sqrt(var)
    assert float(sps.norm.cdf(z)) == pytest.approx(conf, abs=1e-6)


def test_mintrl_infinite_when_no_edge():
    rng = np.random.default_rng(12)
    r = rng.normal(-0.0005, 0.01, 1000)  # losing
    assert min_track_record_length(r) == float("inf")


def test_mintrl_shrinks_as_sharpe_grows():
    rng = np.random.default_rng(13)
    noise = rng.standard_normal(3000) * 0.01
    weak = min_track_record_length(noise + 0.0005)
    strong = min_track_record_length(noise + 0.004)
    assert strong < weak


# ----------------------------------------------------------------------------------
# Drawdown
# ----------------------------------------------------------------------------------


def test_max_drawdown_hand_computed():
    # +100% then -50% -> equity 1, 2, 1. Peak 2, trough 1, drawdown exactly one half.
    assert max_drawdown(np.array([1.0, -0.5])) == pytest.approx(0.5)


def test_max_drawdown_of_monotonic_rise_is_zero():
    assert max_drawdown(np.full(50, 0.01)) == pytest.approx(0.0)


def test_max_drawdown_is_positive_fraction():
    rng = np.random.default_rng(14)
    dd = max_drawdown(rng.normal(0, 0.02, 1000))
    assert 0.0 < dd < 1.0


# ----------------------------------------------------------------------------------
# What deflation is actually worth, measured
#
# The existing test above checks one draw in the right direction. These two measure the
# rate, because the claim the whole tool rests on is a rate: search hard enough and the
# undeflated number blesses noise almost every time, and deflation has to take that away
# without also taking away every real finding.
#
# One thing NOT to assert here: DSR is not a uniform p-value under the null and should
# not be made into one. It is the probability the true Sharpe beats the expected maximum
# of N worthless trials, and the observed maximum concentrates tightly around that
# expectation -- so DSR concentrates near 0.5 rather than spreading over [0, 1]. Testing
# it for uniformity rejects, and the rejection means nothing.
# ----------------------------------------------------------------------------------


def _best_of_n_noise(reps=60, T=800, N=60, edge=0.0, seed=0):
    """(DSR, PSR) for the luckiest of N worthless strategies, `reps` times over.

    Deterministic: fixed seed, so these assertions cannot flake.
    """
    rng = np.random.default_rng(seed)
    dsr, psr = [], []
    for _ in range(reps):
        trials = rng.normal(0.0, 0.01, (T, N))
        if edge:
            trials[:, 0] += edge / np.sqrt(252.0) * 0.01
        srs = sharpe_columns(trials)
        best = int(np.argmax(srs))
        dsr.append(deflated_sharpe(trials[:, best], N, float(np.var(srs, ddof=1)))[0])
        psr.append(probabilistic_sharpe(trials[:, best]))
    return np.asarray(dsr), np.asarray(psr)


def test_deflation_removes_the_false_positives_the_search_manufactured():
    """The central claim, as a rate rather than an anecdote.

    Pick the best of 60 worthless strategies and the undeflated PSR calls it
    significant nearly every time. That is not a subtle effect and it is exactly what
    someone sweeping a grid does without noticing. Deflation has to take it away.
    """
    dsr, psr = _best_of_n_noise()

    assert (psr > 0.95).mean() > 0.75, "the undeflated statistic should be fooled here"
    assert (dsr > 0.95).mean() < 0.05, "deflation blessed pure noise"
    assert 0.3 < dsr.mean() < 0.7, "the best of N should sit near the expected best"


@pytest.mark.parametrize("edge,floor", [(0.0, 0.0), (1.5, 0.55), (3.0, 0.85)])
def test_deflation_still_finds_a_real_edge(edge, floor):
    """The other half. A test that only ever says no is not evidence of anything.

    CONTRIBUTING asks for both directions on every check, and this is the direction
    that is easy to lose: a deflation term that grew too fast would look excellent
    against noise and quietly reject everything real.
    """
    dsr, _ = _best_of_n_noise(edge=edge, seed=1)

    assert dsr.mean() >= floor, (
        f"a true annual Sharpe of {edge} deflated to a mean DSR of {dsr.mean():.3f}"
    )


def test_dsr_rises_monotonically_with_the_size_of_the_real_edge():
    means = [_best_of_n_noise(reps=40, edge=e, seed=2)[0].mean()
             for e in (0.0, 1.0, 2.0, 3.0)]
    assert means == sorted(means), f"not monotone in the true edge: {means}"
