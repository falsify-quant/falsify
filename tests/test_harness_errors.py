"""What `sweep` says when the user's strategy is wrong.

This is the first thing a new user hits, and for a long time it was the same
sentence -- "check the strategy signature" -- for four different mistakes, only
one of which was a signature problem, delivered as a raw traceback. These pin the
diagnosis to the actual cause, because a wrong error message costs more than no
error message: it sends someone to look in the wrong place.

Each case here is a mistake taken from writing the tool's own first-run
experience, not an invented one.
"""

from __future__ import annotations

import numpy as np
import pytest

from falsify_quant.harness import StrategyError, sweep
from falsify_quant.spec import Bars, MarketSpec

SPEC = MarketSpec(name="test", asset_class="equity", bars_per_year=252,
                  fee=0.0005, half_spread=0.0)
GRID = {"fast": [5, 10], "slow": [50, 100]}


def bars(n: int = 400) -> Bars:
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    return Bars(close=close, ts=np.arange(n) * 86400.0 + 1_700_000_000)


def message(strategy, grid=None) -> str:
    with pytest.raises(StrategyError) as exc:
        sweep(strategy, bars(), SPEC, grid or GRID)
    return str(exc.value)


# --------------------------------------------------------------------------------------
# Each mistake must name ITSELF, not a generic signature complaint
# --------------------------------------------------------------------------------------


def test_dataframe_subscript_says_use_attributes():
    """The pandas habit: `bars["close"]`. The old message blamed the signature."""
    def strategy(bars, fast=10, slow=50):
        return np.ones(len(bars["close"]))

    m = message(strategy)
    assert "bars.close" in m
    assert "not a DataFrame" in m
    assert "subscriptable" in m


def test_grid_key_that_is_not_a_parameter_is_named():
    def strategy(bars, fast=10, slow=50):
        return np.ones(len(bars.close))

    m = message(strategy, {"fast": [5], "slow": [50], "lookback": [20]})
    assert "lookback" in m
    # and it must show what the strategy DOES accept, so the fix is obvious
    assert "fast, slow" in m


def test_missing_return_is_diagnosed_as_such():
    def strategy(bars, fast=10, slow=50):
        _ = np.ones(len(bars.close))          # no return

    m = message(strategy)
    assert "returned None" in m
    assert "return" in m


def test_wrong_length_reports_both_counts_and_the_nan_padding_fix():
    def strategy(bars, fast=10, slow=50):
        return np.ones(len(bars.close) - 7)

    m = message(strategy)
    assert "393 weights for 400 bars" in m
    assert "np.nan" in m          # the actual fix, not just the complaint


def test_all_nan_says_the_warmup_never_ends():
    def strategy(bars, fast=10, slow=50):
        return np.full(len(bars.close), np.nan)

    m = message(strategy)
    assert "warmup never ends" in m


def test_non_numeric_return_is_rejected_clearly():
    def strategy(bars, fast=10, slow=50):
        return "long"

    m = message(strategy)
    assert "strategy()" in m


def test_missing_positional_argument_names_the_parameters():
    def strategy(bars, window):                # no default, not in GRID
        return np.ones(len(bars.close))

    m = message(strategy)
    assert "window" in m


def test_the_message_no_longer_blames_the_signature_indiscriminately():
    """Regression on the specific defect: three of these are NOT signature errors."""
    def subscript(bars, fast=10, slow=50):
        return np.ones(len(bars["close"]))

    def no_return(bars, fast=10, slow=50):
        _ = 1

    def short(bars, fast=10, slow=50):
        return np.ones(3)

    for fn in (subscript, no_return, short):
        assert "check the strategy signature" not in message(fn)


# --------------------------------------------------------------------------------------
# Partial failure must be surfaced, because it changes the trial count
# --------------------------------------------------------------------------------------


def test_partial_failure_still_returns_a_sweep_but_warns():
    """One bad corner of the grid must not cost the run -- but must not be silent.

    Cells that failed are excluded from the deflation, which is correct. A user who
    is not told has a different N in their head than the one that was charged.
    """
    def strategy(bars, fast=10, slow=50):
        if slow > 60:
            raise ValueError("window longer than my data")
        c = np.asarray(bars.close)
        return np.where(c > np.mean(c[:50]), 1.0, 0.0)

    sw = sweep(strategy, bars(), SPEC, GRID)
    assert sw.n_failed == 2                      # slow=100 for both fast values
    warning = sw.failure_warning()
    assert warning is not None
    assert "2 of 4" in warning
    assert "window longer than my data" in warning


def test_no_warning_when_everything_ran():
    def strategy(bars, fast=10, slow=50):
        return np.ones(len(bars.close))

    sw = sweep(strategy, bars(), SPEC, GRID)
    assert sw.n_failed == 0
    assert sw.failure_warning() is None


def test_failure_is_a_StrategyError_not_a_bare_RuntimeError():
    """The CLI catches StrategyError to print advice instead of a traceback."""
    def strategy(bars, fast=10, slow=50):
        raise RuntimeError("boom")

    with pytest.raises(StrategyError):
        sweep(strategy, bars(), SPEC, GRID)


# --------------------------------------------------------------------------------------
# The --new template. It is the first code most users will read, and if it ever
# stops running, the onboarding path is worse than having no template at all.
# --------------------------------------------------------------------------------------


def test_template_is_a_working_causal_strategy():
    """Execute the shipped template and score it, exactly as `--new` promises."""
    from falsify_quant.cli import TEMPLATE
    from falsify_quant.prosecute import check_causality

    ns: dict = {}
    exec(compile(TEMPLATE, "template.py", "exec"), ns)

    assert "strategy" in ns and "GRID" in ns, "template must define both names"

    b = bars(600)
    sw = sweep(ns["strategy"], b, SPEC, ns["GRID"], valid=ns.get("valid"))

    assert sw.n_failed == 0, "the template must not fail any of its own grid"
    assert sw.failure_warning() is None
    # The contract it teaches must be the contract it obeys.
    assert check_causality(sw, sw.best_index).score == 1.0


def test_template_returns_one_weight_per_bar_with_nan_warmup():
    from falsify_quant.cli import TEMPLATE

    ns: dict = {}
    exec(compile(TEMPLATE, "template.py", "exec"), ns)
    b = bars(300)
    w = np.asarray(ns["strategy"](b, fast=10, slow=50), dtype=float)

    assert len(w) == len(b.close), "one weight per bar"
    assert np.isnan(w[:49]).all(), "warmup must be NaN, not backfilled or truncated"
    assert not np.isnan(w[60:]).any()


def test_template_grid_stays_small():
    """The tool's whole thesis is that trials are expensive; the template must model it."""
    from falsify_quant.cli import TEMPLATE
    from falsify_quant.harness import grid_size

    ns: dict = {}
    exec(compile(TEMPLATE, "template.py", "exec"), ns)
    assert grid_size(ns["GRID"]) <= 16, "a template that ships a huge grid teaches the opposite"
