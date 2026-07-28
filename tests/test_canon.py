"""The published canon, checked for the things that would invalidate the corpus study.

Every strategy in `strategies/canon.py` is a claim about what somebody published. Four
ways that claim could be worthless, all tested here:

  the implementation reads the future, so the study measures a bug;
  the shipped parameters are not in the grid, so the study silently scores something else;
  the rule never trades, so it reports a flawless zero it did not earn;
  a long-only rule shorts, so the study attributes losses its source never took.

The causality test is run over every candidate at several truncation points. It is the
same argument `falsify` makes about whole strategies, applied here as a unit test so a
leak shows up as a red build rather than as a surprisingly good study result.
"""

from __future__ import annotations

import numpy as np
import pytest

from falsify_quant.spec import Bars
from strategies.canon import CANON, Candidate, by_name, for_cadence, turn_of_month

NAMES = [c.name for c in CANON]


def _market(n: int = 1200, seed: int = 7) -> Bars:
    """Daily-ish OHLC with real trends in it, so path-dependent rules actually fire."""
    rng = np.random.default_rng(seed)
    rho = 0.5 ** (1.0 / 60.0)
    mu = np.zeros(n)
    z = rng.normal(0.0, 1.0, n)
    for i in range(1, n):
        mu[i] = rho * mu[i - 1] + np.sqrt(1 - rho**2) * 0.0025 * z[i]
    r = mu + rng.normal(0.0, 0.011, n)
    close = 100.0 * np.cumprod(1.0 + r)
    high = close * (1.0 + np.abs(rng.normal(0.0, 0.006, n)))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.006, n)))
    op = low + (high - low) * rng.random(n)
    ts = 1_500_000_000.0 + np.arange(n) * 86400.0
    return Bars(open=op, high=high, low=low, close=close,
                volume=rng.random(n) * 1e6, ts=ts, symbol="SYNTH")


def _truncate(b: Bars, k: int) -> Bars:
    return Bars(open=b.open[:k], high=b.high[:k], low=b.low[:k], close=b.close[:k],
                volume=b.volume[:k], ts=b.ts[:k], symbol=b.symbol)


def _run(c: Candidate, b: Bars, **over) -> np.ndarray:
    return c.fn(b, **{**c.shipped, **over})


# --------------------------------------------------------------------------------------
# Causality
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("k", [400, 700, 1199])
def test_canon_strategy_is_causal(name, k):
    """Deleting the future must not change any weight in the past.

    A strategy that peeks exactly one bar ahead differs from its truncated self at
    precisely one index -- the last one -- so the comparison runs to the end of the
    truncated range rather than stopping short of it.
    """
    c = by_name(name)
    b = _market()
    full = _run(c, b)
    trunc = _run(c, _truncate(b, k))

    assert len(trunc) == k
    a, d = full[:k], trunc[:k]
    assert np.array_equal(np.isnan(a), np.isnan(d)), f"{name}: warmup moved under truncation"
    both = ~np.isnan(a)
    np.testing.assert_array_equal(a[both], d[both], err_msg=f"{name} reads ahead")


def test_the_seasonal_trap_would_have_been_caught():
    """The bar-sequence reading of 'last trading day of the month' is a lookahead.

    `turn_of_month` uses the calendar, which is knowable years ahead. Defining the window
    by where the month changes in the bar series is the natural alternative and it fails
    truncation, because the final bar of a truncated series cannot yet know whether it is
    the last one of its month. Pinned here so the distinction stays visible.
    """
    b = _market()

    def leaky(bars, before=1, after=3):
        months = bars.ts.astype("datetime64[s]").astype("datetime64[M]")
        changes = np.zeros(len(months), dtype=bool)
        changes[:-1] = months[1:] != months[:-1]  # <-- reads the next bar
        w = np.zeros(len(months))
        for i in np.flatnonzero(changes):
            w[max(0, i - int(before) + 1): i + int(after) + 1] = 1.0
        return w

    # Truncate *on* a month boundary. That is the bar whose status the leak invents:
    # anywhere mid-month both versions agree, which is why the leak is easy to miss.
    months = b.ts.astype("datetime64[s]").astype("datetime64[M]")
    boundaries = np.flatnonzero(months[1:] != months[:-1])
    k = int(boundaries[boundaries > 400][0]) + 1

    honest_full, honest_trunc = turn_of_month(b), turn_of_month(_truncate(b, k))
    np.testing.assert_array_equal(honest_full[:k], honest_trunc[:k])

    leaky_full, leaky_trunc = leaky(b), leaky(_truncate(b, k))
    assert leaky_full[k - 1] == 1.0 and leaky_trunc[k - 1] == 0.0
    assert not np.array_equal(leaky_full[:k], leaky_trunc[:k])


# --------------------------------------------------------------------------------------
# Catalogue integrity
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_shipped_parameters_are_reachable_in_the_grid(name):
    """`falsify` scores the shipped variant by looking it up in the sweep by value.

    A shipped parameter that is not on its grid axis does not raise -- it silently
    resolves to something else, and the study reports a verdict on a strategy nobody
    published.
    """
    c = by_name(name)
    assert set(c.shipped) == set(c.grid), f"{name}: shipped keys and grid axes disagree"
    for k, v in c.shipped.items():
        assert v in list(c.grid[k]), f"{name}: shipped {k}={v} is not on its grid axis"


@pytest.mark.parametrize("name", NAMES)
def test_shipped_parameters_pass_the_validity_filter(name):
    c = by_name(name)
    if c.valid is not None:
        assert c.valid(dict(c.shipped)), f"{name}: its own shipped parameters are excluded"


@pytest.mark.parametrize("name", NAMES)
def test_grid_is_a_usable_size(name):
    """Big enough to represent a real search, small enough that the study finishes."""
    c = by_name(name)
    assert 4 <= c.n_grid <= 200, f"{name}: {c.n_grid} valid grid points"


def test_catalogue_has_no_duplicate_names():
    assert len(NAMES) == len(set(NAMES))


def test_every_candidate_cites_something():
    for c in CANON:
        assert len(c.source) > 10, f"{c.name} has no source"
        assert c.family in {"trend", "breakout", "reversion", "seasonal"}


def test_seasonal_rules_are_daily_only():
    """A turn-of-month window means nothing on hourly bars."""
    hourly = {c.name for c in for_cadence("hourly")}
    assert "turn-of-month" not in hourly
    assert "golden-cross" in hourly


def test_by_name_rejects_unknown():
    with pytest.raises(KeyError, match="no strategy named"):
        by_name("not-a-strategy")


# --------------------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_weights_are_finite_after_warmup_and_bounded(name):
    c = by_name(name)
    w = _run(c, _market())
    assert len(w) == 1200

    valid = ~np.isnan(w)
    assert valid.sum() > 400, f"{name}: only {valid.sum()} usable bars out of 1200"
    assert np.isfinite(w[valid]).all(), f"{name}: non-finite weights outside the warmup"
    assert np.abs(w[valid]).max() <= 3.0, f"{name}: weight beyond any documented leverage"

    # Warmup must be a prefix. A NaN in the middle means an indicator broke.
    first = int(np.argmax(valid))
    assert valid[first:].all(), f"{name}: NaN weights appear after trading has started"


@pytest.mark.parametrize("name", NAMES)
def test_strategy_actually_trades(name):
    """A rule that never changes position reports a perfect zero it did not earn.

    This is how the unshifted-channel bug presents: `close > rolling_max(close, n)` is
    never true, so the breakout is flat forever, costs nothing, and scores as harmless.
    """
    c = by_name(name)
    w = _run(c, _market())
    v = w[~np.isnan(w)]
    flips = int((np.diff(v) != 0).sum())
    # Two is the bar, not a typo. A 50/200 crossover on a thousand bars genuinely turns
    # over three or four times -- that is the whole selling point of a slow trend rule.
    # This test is here to catch zero, not to enforce activity.
    assert flips >= 2, f"{name}: only {flips} position changes in 1200 bars"
    assert len(np.unique(np.sign(v))) > 1, f"{name}: holds one constant position throughout"


@pytest.mark.parametrize("name", [c.name for c in CANON if c.long_only])
def test_long_only_rules_never_short(name):
    c = by_name(name)
    w = _run(c, _market())
    assert np.nanmin(w) >= 0.0, f"{name} is documented long-only but took a short position"


@pytest.mark.parametrize("name", [c.name for c in CANON if not c.long_only
                                 and c.family != "seasonal"])
def test_symmetric_rules_use_both_sides(name):
    c = by_name(name)
    w = _run(c, _market())
    assert np.nanmin(w) < 0.0, f"{name} is documented long/short but never went short"
    assert np.nanmax(w) > 0.0, f"{name} never went long"


@pytest.mark.parametrize("name", NAMES)
def test_runs_on_a_close_only_history(name):
    """Not every source has OHLC. Rules that need it declare it; the rest must cope."""
    c = by_name(name)
    b = _market()
    close_only = Bars(close=b.close, ts=b.ts, symbol=b.symbol)
    w = _run(c, close_only)
    assert np.isfinite(w[~np.isnan(w)]).all()
    if not c.needs:
        assert (~np.isnan(w)).sum() > 400, f"{name} declares no OHLC need but produced nothing"


def test_donchian_would_be_dead_without_the_channel_shift():
    """The bug the shift exists to prevent, demonstrated rather than described."""
    from falsify_quant.indicators import rolling_max

    b = _market()
    assert not np.any(b.close > rolling_max(b.close, 20))  # unshifted: never fires
    w = _run(by_name("donchian"), b)
    assert (np.diff(w[~np.isnan(w)]) != 0).sum() > 20  # shifted: trades


def test_the_two_bollinger_readings_disagree():
    """Breakout and fade are opposite readings of the same bands. They must not agree."""
    b = _market()
    up = _run(by_name("bollinger-breakout"), b)
    fade = _run(by_name("bollinger-reversion"), b)
    both = ~np.isnan(up) & ~np.isnan(fade)
    active = both & ((up != 0) | (fade != 0))
    assert active.sum() > 50
    assert np.corrcoef(up[active], fade[active])[0, 1] < 0.0


# --------------------------------------------------------------------------------------
# Home turf
# --------------------------------------------------------------------------------------


def test_every_rule_declares_where_its_source_tested_it():
    """The `Home turf` section of the study is only meaningful if this is complete."""
    allowed = {"equity-index", "futures", "cross-asset", "unstated"}
    for c in CANON:
        assert c.domain in allowed, f"{c.name} has domain {c.domain!r}"


def test_the_domain_labels_are_pinned_to_the_citations():
    """Frozen deliberately.

    These labels decide which cells count as a fair test, and they were written down
    after the scores existed. That is the exact situation in which a label quietly drifts
    toward whatever makes the result cleaner. Changing one now means changing this list,
    in a diff, next to the citation it is supposed to come from.
    """
    expected = {
        "golden-cross": "equity-index",       # Brock/Lakonishok/LeBaron tested the DJIA
        "price-vs-ma": "equity-index",        # Faber, broad asset-class indices
        "vol-target-trend": "equity-index",   # Harvey et al, equity indices
        "rsi2-connors": "equity-index",       # Connors & Alvarez, equity index ETFs
        "n-down-days": "equity-index",        # Connors & Alvarez
        "turn-of-month": "equity-index",      # Ariel; Lakonishok & Smidt
        "donchian": "futures",                # Turtles, commodity/financial futures
        "keltner-breakout": "futures",        # Keltner (1960), commodities
        "chandelier": "futures",              # LeBeau, futures markets
        "rsi-reversion": "futures",           # Wilder (1978), commodities
        "stochastic": "futures",              # Lane, commodity futures
        "williams-r": "futures",              # Larry Williams, commodities
        "tsmom": "cross-asset",               # Moskowitz et al, 58-futures panel
        "dual-ma": "unstated",
        "triple-ma": "unstated",
        "macd": "unstated",
        "bollinger-breakout": "unstated",
        "bollinger-reversion": "unstated",
    }
    assert {c.name: c.domain for c in CANON} == expected


def test_no_rule_is_labelled_by_how_well_it_scored():
    """A label assigned from the outcome would put every winner on home turf.

    If `equity-index` had been reverse-engineered from the scores, the futures group
    would be a leftover bin of losers. It is not: it contains rules from named primary
    sources across four decades, and the split cuts across the family taxonomy rather
    than following it.
    """
    futures = {c.family for c in CANON if c.domain == "futures"}
    equity = {c.family for c in CANON if c.domain == "equity-index"}
    assert len(futures) > 1 and len(equity) > 1
    assert futures & equity, "the two groups must share at least one family"
