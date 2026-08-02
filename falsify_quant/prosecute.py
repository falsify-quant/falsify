"""The prosecution. Six independent attempts to prove the strategy is nothing.

Each test returns a `Finding` with a health score in [0, 1], where 1 means the test
failed to kill the strategy and 0 means it did. Nothing here tries to make a strategy
look better; there is no knob to turn and no parameter to improve. That is the point.

The tests, roughly in order of how often they land:

  causality   -- does the strategy read bars it could not have seen?  (a bug, not a weakness)
  costs       -- is the edge per unit of turnover bigger than the cost per unit?
  deflation   -- does the Sharpe survive being charged for the size of the search?
  pbo         -- does the in-sample winner keep winning out of sample?
  permutation -- does the same search find this much on data with no signal in it?
  regime      -- is the P&L spread across time, or is it one lucky quarter?
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
from scipy.stats import norm as _norm

from .harness import Sweep, sweep as run_sweep
from .sim import simulate
from .spec import Bars
from .stats import (
    annualise,
    deflated_sharpe,
    max_drawdown,
    min_track_record_length,
    sharpe,
    sharpe_columns,
)

__all__ = [
    "Finding",
    "check_causality",
    "check_costs",
    "check_deflation",
    "check_pbo",
    "check_permutation",
    "check_regime",
]


@dataclass
class Finding:
    name: str
    title: str
    score: float  # 1.0 = survived this test, 0.0 = killed by it
    headline: str  # the sentence a human will actually read
    detail: dict[str, Any] = field(default_factory=dict)
    fatal: bool = False  # a bug rather than a weakness; gates the whole verdict
    advice: str = ""

    def __post_init__(self) -> None:
        # A test that could not be computed has established nothing. Fail closed: in a
        # tool whose entire job is scepticism, "uncomputable" must never read as "passed".
        if not np.isfinite(self.score):
            self.score = 0.0
        self.score = float(np.clip(self.score, 0.0, 1.0))


def _squash(x: float, midpoint: float) -> float:
    """Map [0, inf) to [0, 1) with 0.5 at `midpoint`. Used to turn ratios into scores."""
    if x <= 0:
        return 0.0
    return float(x / (x + midpoint))


# --------------------------------------------------------------------------------------
# 1. Causality -- the only test that finds bugs rather than weaknesses
# --------------------------------------------------------------------------------------


def check_causality(sweep_: Sweep, index: int, *, cuts: tuple[float, ...] = (0.6, 0.75, 0.9)) -> Finding:
    """Re-run the strategy on truncated history and check its past decisions are unchanged.

    If `w[t]` computed from bars[0:k] differs from `w[t]` computed from bars[0:T] for any
    t < k, the strategy is reading the future. This is exact, not statistical -- there is
    no p-value to argue with.

    It catches the classic accidents: full-sample z-scores and min/max normalisation,
    centred rolling windows, `bfill()` on a gap, resampling that leaks the closing bar,
    and any indicator computed on the whole array before slicing.
    """
    params = sweep_.params[index]
    bars = sweep_.bars
    T = len(bars)

    w_full = np.nan_to_num(np.asarray(sweep_.strategy(bars, **params), dtype=np.float64))

    worst_leak = 0.0
    first_leak_at: int | None = None
    leaking_cuts = 0

    for frac in cuts:
        k = int(T * frac)
        if k < 60:
            continue
        w_trunc = np.nan_to_num(
            np.asarray(sweep_.strategy(bars.slice(0, k), **params), dtype=np.float64)
        )
        if len(w_trunc) != k:
            # A strategy that does not return one weight per bar is broken in a way that
            # makes every other test meaningless.
            return Finding(
                name="causality",
                title="Causality",
                score=0.0,
                headline=(
                    f"Strategy returned {len(w_trunc)} weights for {k} bars. It must return "
                    "exactly one weight per bar."
                ),
                fatal=True,
                advice="Fix the return shape before interpreting any other result.",
                detail={"expected": k, "got": len(w_trunc)},
            )

        # Compare the WHOLE truncated range, final bar included.
        #
        # An earlier version skipped the last bar, reasoning that a weight decided from
        # the final close is never acted on inside the window. That reasoning is wrong,
        # and it opened the worst possible blind spot: for a causal strategy w[t] depends
        # only on close[0..t], so truncation cannot change *any* bar up to k-1 including
        # k-1 itself -- there is nothing to forgive. Meanwhile a strategy peeking exactly
        # one bar ahead differs from its truncated self at precisely one index: the last
        # one. Skipping it made the single most common lookahead bug invisible.
        diff = np.abs(w_full[:k] - w_trunc[:k])
        scale = max(1e-9, float(np.max(np.abs(w_full[:k]))) or 1.0)
        rel = diff / scale

        if np.any(rel > 1e-6):
            leaking_cuts += 1
            worst_leak = max(worst_leak, float(np.max(rel)))
            idx = int(np.argmax(rel > 1e-6))
            first_leak_at = idx if first_leak_at is None else min(first_leak_at, idx)

    if leaking_cuts == 0:
        if float(np.max(np.abs(w_full))) <= 1e-12:
            # Nothing was ever decided, so nothing could have leaked. Passing this
            # silently would let a do-nothing variant look clean.
            return Finding(
                name="causality",
                title="Causality",
                score=1.0,
                headline="Not applicable — this variant never takes a position.",
                detail={"cuts_tested": list(cuts), "always_flat": True},
            )
        return Finding(
            name="causality",
            title="Causality",
            score=1.0,
            headline="No lookahead. Past decisions are unchanged when future bars are removed.",
            detail={"cuts_tested": list(cuts)},
        )

    return Finding(
        name="causality",
        title="Causality",
        score=0.0,
        headline=(
            f"LOOKAHEAD. Decisions change by up to {worst_leak:.1%} when future bars are "
            f"withheld — first at bar {first_leak_at}. The strategy is reading data it "
            "would not have had."
        ),
        fatal=True,
        advice=(
            "Look for full-sample normalisation (z-scores, min/max scaling), centred rolling "
            "windows, backward fills, or any indicator computed over the entire array before "
            "slicing. Every other number in this report is meaningless until this is fixed."
        ),
        detail={"max_relative_change": worst_leak, "first_leak_bar": first_leak_at,
                "cuts_leaking": leaking_cuts, "cuts_tested": list(cuts)},
    )


# --------------------------------------------------------------------------------------
# 2. Costs -- usually the whole story
# --------------------------------------------------------------------------------------


def check_costs(sweep_: Sweep, index: int) -> Finding:
    """Compare the edge earned per unit of turnover against the cost of that turnover.

    Because net return is linear in the cost rate, the breakeven cost is exact rather
    than searched: it is simply gross P&L divided by total turnover. Most retail
    strategies die here and the arithmetic takes one line.
    """
    res = sweep_.result_for(index)
    spec = sweep_.spec
    bpy = spec.bars_per_year

    if float(np.sum(res.turnover)) <= 1e-12:
        # On data with no signal, a grid's highest-Sharpe cell is often one that never
        # opens a position: a Sharpe of exactly zero beats every losing variant. Worth
        # saying out loud, because it looks like a clean pass on several other tests.
        return Finding(
            name="costs",
            title="Cost survival",
            score=0.0,
            headline=(
                "This variant never trades. It scored highest in the grid because a flat "
                "line beat every variant that actually took positions."
            ),
            advice=(
                "Widen the entry thresholds until the strategy trades, or accept that the "
                "grid found nothing worth acting on."
            ),
            detail={"n_trades": 0, "total_turnover": 0.0},
        )

    edge = res.edge_per_turnover  # gross return per unit of capital churned
    cost = spec.cost_per_turnover  # what that churn costs
    margin = edge / cost if cost > 0 else math.inf

    total_churn = float(np.sum(res.turnover))
    breakeven = edge  # by construction: the cost rate at which net P&L is exactly zero

    # Sharpe as a function of cost multiplier, for the report's curve.
    curve = []
    for mult in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        r = simulate(sweep_.bars, sweep_.strategy(sweep_.bars, **sweep_.params[index]),
                     spec.scaled(mult))
        curve.append({"multiple": mult, "sharpe": annualise(sharpe(r.net), bpy),
                      "total_return": r.net_return})

    turns_per_year = res.avg_turnover * bpy
    score = _squash(max(0.0, margin - 1.0), 1.5)  # margin 2.5x -> 0.5, margin 1x -> 0

    if edge <= 0:
        # No cost ratio is meaningful here -- there is nothing to divide. The strategy
        # is losing money before a single fee is charged, so costs are not the problem
        # and cheaper execution cannot fix it.
        headline = (
            f"Loses money before costs. The gross edge is {edge*1e4:.1f} bps per unit of "
            f"turnover — negative — so the {cost*1e4:.1f} bps of trading cost is beside the "
            "point. The signal itself is backwards or absent."
        )
        advice = (
            "Cheaper execution cannot rescue a negative gross edge. Check whether the "
            "signal has the sign you intended before touching fees or turnover."
        )
    elif margin <= 1.0:
        headline = (
            f"Dead on costs. The edge is {edge*1e4:.1f} bps per unit of turnover and trading "
            f"costs {cost*1e4:.1f} bps. You are paying {cost/edge:.1f}x your edge "
            "to collect it."
        )
        advice = (
            "No parameter will fix this. Either trade far less often, get maker fills instead "
            "of taker, or find a bigger edge. Turnover is the lever with the most leverage."
        )
    elif margin < 2.0:
        headline = (
            f"Thin. Edge is {edge*1e4:.1f} bps per unit turnover against {cost*1e4:.1f} bps of "
            f"cost — a {margin:.2f}x margin. A fee-tier change or a wider spread ends this."
        )
        advice = "Model your real fills before sizing up. A 2x margin is one bad week from zero."
    else:
        headline = (
            f"Survives costs. Edge of {edge*1e4:.1f} bps per unit turnover against "
            f"{cost*1e4:.1f} bps of cost — a {margin:.2f}x margin."
        )
        advice = ""

    return Finding(
        name="costs",
        title="Cost survival",
        score=score,
        headline=headline,
        advice=advice,
        detail={
            "edge_per_turnover_bps": edge * 1e4,
            "cost_per_turnover_bps": cost * 1e4,
            "margin": margin,
            "breakeven_cost_bps": breakeven * 1e4,
            "total_turnover": total_churn,
            "turnover_per_year": turns_per_year,
            "n_trades": res.n_trades,
            "gross_return": res.gross_return,
            "net_return": res.net_return,
            "cost_drag": res.total_cost,
            "curve": curve,
        },
    )


# --------------------------------------------------------------------------------------
# 3. Deflation -- charging the Sharpe for the size of the search
# --------------------------------------------------------------------------------------


def check_deflation(
    sweep_: Sweep, index: int, *, prior_sharpes: np.ndarray | None = None
) -> Finding:
    """Deflated Sharpe Ratio: does the result beat the best of N coin flips?

    `prior_sharpes` carries the trials from *earlier searches of the same question*, so
    they can be charged for too. A grid is one search; a person who runs a grid, dislikes
    the answer, adjusts it and runs again has run two, and the honest N is the total. The
    library cannot know that on its own -- each call sees one sweep -- so anything that
    does know, such as a session that has watched you re-run four times, passes the
    earlier trial Sharpes in and they are pooled into both N and V.
    """
    net = sweep_.returns[:, index].astype(np.float64)
    bpy = sweep_.spec.bars_per_year
    live = sweep_.sharpes[~sweep_.failed]

    if prior_sharpes is not None and len(prior_sharpes):
        pooled = np.concatenate([np.asarray(prior_sharpes, dtype=np.float64), live])
        pooled = pooled[np.isfinite(pooled)]
    else:
        pooled = live[np.isfinite(live)]

    n_trials = int(len(pooled))
    var_trials = float(np.var(pooled, ddof=1)) if n_trials > 1 else 0.0

    observed = sharpe(net)
    dsr, sr0 = deflated_sharpe(net, n_trials, var_trials, observed_sr=observed)

    ann_obs = annualise(observed, bpy)
    ann_deflated_bar = annualise(sr0, bpy)
    mintrl = min_track_record_length(net, benchmark_sr=sr0)

    if dsr >= 0.95:
        headline = (
            f"Sharpe {ann_obs:.2f} survives deflation for {n_trials:,} trials "
            f"(DSR {dsr:.2f}). The luckiest of {n_trials:,} nothings would have scored "
            f"{ann_deflated_bar:.2f}."
        )
        advice = ""
    elif dsr >= 0.5:
        headline = (
            f"Sharpe {ann_obs:.2f} is not clearly better than luck. Searching {n_trials:,} "
            f"variants is expected to produce {ann_deflated_bar:.2f} from noise alone; "
            f"confidence that this is real is only {dsr:.0%}."
        )
        advice = (
            f"Either shrink the search space or extend the sample. At this Sharpe you need "
            f"{mintrl:,.0f} bars of live trading to prove skill."
            if math.isfinite(mintrl) else "Shrink the search space or extend the sample."
        )
    else:
        headline = (
            f"Sharpe {ann_obs:.2f} is worse than the search alone explains. Trying "
            f"{n_trials:,} variants is expected to yield {ann_deflated_bar:.2f} on pure "
            f"noise. Confidence this is real: {dsr:.0%}."
        )
        advice = (
            "The result is inside the noise floor of your own search. Cut the grid down to "
            "parameters you can justify from first principles and re-run."
        )

    return Finding(
        name="deflation",
        title="Deflated Sharpe",
        score=dsr,
        headline=headline,
        advice=advice,
        detail={
            "dsr": dsr,
            "n_trials": n_trials,
            "observed_sharpe_annual": ann_obs,
            "deflated_benchmark_annual": ann_deflated_bar,
            "trial_sharpe_variance": var_trials,
            "min_track_record_bars": mintrl,
            "max_drawdown": max_drawdown(net),
        },
    )


# --------------------------------------------------------------------------------------
# 4. PBO -- does the in-sample winner keep winning?
# --------------------------------------------------------------------------------------


def check_pbo(sweep_: Sweep, *, n_blocks: int = 16, chunk: int = 256) -> Finding:
    """Combinatorially Symmetric Cross-Validation (Bailey, Borwein, Lopez de Prado, Zhu).

    Chop the history into S blocks. For each of the C(S, S/2) ways to split them into
    equal in-sample and out-of-sample halves, pick the best variant in-sample and see
    where it ranks out-of-sample. PBO is how often the winner lands in the bottom half.

    A genuine edge keeps winning: PBO near 0. A curve fit does not: PBO near 0.5 means
    the selection carries no information at all, and above 0.5 means picking the
    in-sample best is actively worse than picking at random.

    Implemented on block sums rather than raw slices so the C(16,8) = 12,870 splits stay
    a few matrix multiplies instead of a hundred million Sharpe computations.
    """
    live = ~sweep_.failed
    M = sweep_.returns[:, live].astype(np.float64)
    T, N = M.shape

    if N < 4:
        return Finding(
            name="pbo", title="Out-of-sample rank", score=0.5,
            headline="Too few surviving variants to measure overfitting by selection.",
            detail={"n_variants": N},
        )

    if n_blocks % 2:
        n_blocks -= 1
    while n_blocks > 4 and T // n_blocks < 20:
        n_blocks -= 2  # blocks must be long enough for a Sharpe to mean anything
    if n_blocks < 4:
        return Finding(
            name="pbo", title="Out-of-sample rank", score=0.5,
            headline=f"Only {T} bars — too short to split into meaningful train/test blocks.",
            detail={"n_bars": T},
        )

    edges = np.linspace(0, T, n_blocks + 1).astype(int)
    counts = np.diff(edges).astype(np.float64)  # (S,)
    b_sum = np.stack([M[edges[i]:edges[i + 1]].sum(axis=0) for i in range(n_blocks)])  # (S,N)
    b_sq = np.stack([(M[edges[i]:edges[i + 1]] ** 2).sum(axis=0) for i in range(n_blocks)])

    half = n_blocks // 2
    all_combos = list(combinations(range(n_blocks), half))

    def block_sharpe(mask: np.ndarray) -> np.ndarray:
        """Sharpe of every variant over the blocks selected by `mask` (C, S) -> (C, N)."""
        n = mask @ counts  # (C,)
        s1 = mask @ b_sum  # (C, N)
        s2 = mask @ b_sq  # (C, N)
        n_c = n[:, None]
        mean = s1 / n_c
        var = (s2 - s1 * s1 / n_c) / (n_c - 1.0)
        sd = np.sqrt(np.maximum(var, 0.0))
        out = np.zeros_like(mean)
        ok = sd > 1e-15
        out[ok] = mean[ok] / sd[ok]
        return out

    logits: list[float] = []
    chosen_oos_sr: list[float] = []
    for start in range(0, len(all_combos), chunk):
        batch = all_combos[start:start + chunk]
        mask = np.zeros((len(batch), n_blocks), dtype=np.float64)
        for j, combo in enumerate(batch):
            mask[j, list(combo)] = 1.0

        sr_is = block_sharpe(mask)
        sr_oos = block_sharpe(1.0 - mask)

        best = np.argmax(sr_is, axis=1)  # (C,)
        rows = np.arange(len(batch))
        chosen_oos = sr_oos[rows, best]
        chosen_oos_sr.extend(chosen_oos.tolist())

        # Relative rank of the chosen variant within the OOS Sharpe distribution.
        rank = (sr_oos < chosen_oos[:, None]).sum(axis=1)
        omega = (rank + 0.5) / N
        omega = np.clip(omega, 1e-6, 1 - 1e-6)
        logits.extend(np.log(omega / (1.0 - omega)).tolist())

    lam = np.asarray(logits)
    oos = np.asarray(chosen_oos_sr)
    pbo = float(np.mean(lam <= 0.0))
    median_rank = float(np.mean(1.0 / (1.0 + np.exp(-lam))))

    # PBO asks whether the *ranking* generalises, which is not the same question as
    # whether the strategy works. If every variant in the grid carries a real edge, the
    # ranking among them is close to random and PBO climbs toward 0.5 -- while the thing
    # that actually matters, "does the variant I picked still make money out of sample",
    # is a resounding yes. That is a plateau, and a plateau is the good case: it means
    # the result does not depend on landing on one lucky cell.
    #
    # Textbook PBO scores that situation as a failure. So measure the absolute level too,
    # and let either form of robustness carry the finding.
    oos_win_rate = float(np.mean(oos > 0.0))
    median_oos = float(np.median(oos))

    rank_score = float(np.clip(1.0 - 2.0 * pbo, 0.0, 1.0))  # pbo 0 -> 1, pbo >= 0.5 -> 0
    level_score = float(np.clip(2.0 * oos_win_rate - 1.0, 0.0, 1.0))  # 50% -> 0, 100% -> 1
    score = max(rank_score, level_score)
    plateau = pbo >= 0.15 and level_score > rank_score
    bpy = sweep_.spec.bars_per_year

    if plateau:
        headline = (
            f"Flat parameter surface, but it holds up. The in-sample winner ranks near "
            f"random out-of-sample (PBO {pbo:.2f}) yet still made money in "
            f"{oos_win_rate:.0%} of {len(lam):,} splits — the variants are largely "
            "interchangeable, which is the good kind of insensitivity."
        )
        advice = (
            "Since the grid is a plateau rather than a peak, take the middle of the region "
            "rather than the top cell. You give up nothing and stop depending on the rank."
        )
    elif pbo < 0.15:
        headline = (
            f"Selection holds up. The in-sample winner stayed in the top half out-of-sample "
            f"in {1-pbo:.0%} of {len(lam):,} splits (PBO {pbo:.2f})."
        )
        advice = ""
    elif pbo < 0.35:
        headline = (
            f"Selection is shaky. The in-sample winner dropped to the bottom half "
            f"out-of-sample in {pbo:.0%} of {len(lam):,} splits (PBO {pbo:.2f}), and made "
            f"money in only {oos_win_rate:.0%} of them."
        )
        advice = (
            "Prefer a parameter plateau over the single best cell — pick the middle of a "
            "broad region, not a peak."
        )
    else:
        headline = (
            f"Selection is noise. The in-sample winner fell to the bottom half out-of-sample "
            f"in {pbo:.0%} of {len(lam):,} splits (PBO {pbo:.2f}) and was profitable in only "
            f"{oos_win_rate:.0%}. Choosing the best backtest is no better than choosing at "
            "random."
        )
        advice = (
            "This is the signature of a curve fit: the parameter surface has no stable "
            "structure, so whatever wins in one period loses in the next."
        )

    return Finding(
        name="pbo",
        title="Out-of-sample rank",
        score=score,
        headline=headline,
        advice=advice,
        detail={"pbo": pbo, "oos_profitable_rate": oos_win_rate,
                "median_oos_sharpe_annual": annualise(median_oos, bpy),
                "n_splits": len(lam), "n_blocks": n_blocks,
                "n_variants": N, "mean_oos_percentile": median_rank},
    )


# --------------------------------------------------------------------------------------
# 5. Permutation -- run the same search on data with nothing in it
# --------------------------------------------------------------------------------------


def check_permutation(
    sweep_: Sweep,
    *,
    n_runs: int = 100,
    method: str = "iid",
    block_size: int | None = None,
    seed: int = 0,
) -> Finding:
    """Monte Carlo permutation test over the *whole search*, not one strategy.

    Manufacture price paths with the same drift and volatility as the real series but no
    exploitable structure, then run the identical parameter sweep on each. The p-value is
    how often the search found something at least as good on noise.

    This is the test people find hardest to argue with, because it makes no assumption
    about the strategy at all. It measures the search procedure itself.

    Two nulls, and the choice matters:

      method="iid"    (default) shuffles returns freely, destroying all serial structure.
                      Rejecting means the market had exploitable structure and the search
                      found it. This isolates the search, which is the primary question.

      method="block"  resamples contiguous runs, preserving volatility clustering and
                      short-horizon autocorrelation. A strictly harder null: rejecting
                      means the edge is more than generic serial dependence. A strategy
                      whose entire edge *is* lag-1 autocorrelation will pass under "iid"
                      and fail under "block", and both of those facts are worth knowing.
    """
    rng = np.random.default_rng(seed)
    bars = sweep_.bars
    r = bars.returns[1:]  # drop the leading zero
    T = len(r)
    if block_size is None:
        block_size = max(5, int(np.sqrt(T)))

    real_best = float(np.max(sweep_.sharpes[~sweep_.failed]))
    p0 = float(bars.close[0])

    null_best: list[float] = []
    first_failure: str | None = None
    for _ in range(n_runs):
        if method == "iid":
            synth = rng.permutation(r)
        else:
            n_blocks = int(np.ceil(T / block_size))
            starts = rng.integers(0, max(1, T - block_size), size=n_blocks)
            synth = np.concatenate([r[s:s + block_size] for s in starts])[:T]

        prices = p0 * np.cumprod(1.0 + np.concatenate([[0.0], synth]))
        synth_bars = bars.with_close(prices)

        try:
            s = run_sweep(sweep_.strategy, synth_bars, sweep_.spec, sweep_.grid)
        except Exception as exc:  # noqa: BLE001 -- one bad synthetic path is not fatal
            if first_failure is None:
                first_failure = f"{type(exc).__name__}: {exc}"
            continue
        alive = s.sharpes[~s.failed]
        if len(alive):
            null_best.append(float(np.max(alive)))

    if not null_best:
        # Saying only "could not run" leaves the user to guess, which is the thing this
        # project exists not to do. The cause is nearly always the same one: synthetic
        # paths are manufactured from returns, so `with_close` gives them a close series
        # and nothing else. A strategy reading bars.volume or bars.high works perfectly
        # on the real series and raises on every null run -- and this check is one of the
        # six that are weighted, so a silent 0.5 moves the score with no explanation.
        # The nested failure is a whole multi-line diagnosis. The headline gets one
        # sentence of it; the rest belongs in the advice, where there is room.
        cause = ""
        if first_failure:
            root = [ln.strip() for ln in first_failure.splitlines() if ln.strip()][-1]
            cause = f" Every synthetic run failed with {root}"
        return Finding(
            name="permutation", title="Search on noise", score=0.5,
            headline="Permutation test could not run on this strategy." + cause,
            detail={"n_runs": 0, "first_failure": first_failure},
            advice=(
                "Synthetic price paths carry a close series and nothing else: they are "
                "built from permuted returns, and there is no honest way to manufacture "
                "a matching high, low or volume to go with them. A strategy that reads "
                "any of those works on your real data and raises on every null path, "
                "which is what happened here. Express the entry rule in terms of "
                "bars.close to get this check.\n\nThe failure was:\n" + first_failure
                if first_failure else
                "The sweep produced no scorable result on any synthetic path."
            ),
        )

    null = np.asarray(null_best)
    beaten = int(np.count_nonzero(null >= real_best))
    p_value = float((1 + beaten) / (len(null) + 1))
    floor = 1.0 / (len(null) + 1)  # best p-value this many runs can resolve
    bpy = sweep_.spec.bars_per_year

    score = float(np.clip(1.0 - p_value / 0.10, 0.0, 1.0))  # p <= 0 -> 1, p >= 0.10 -> 0

    if beaten == 0:
        headline = (
            f"Beats noise. On {len(null)} synthetic histories with no signal, the same search "
            f"never once found anything this good — p is at this test's resolution floor "
            f"of {floor:.3f}."
        )
        advice = ""
    elif p_value <= 0.02:
        headline = (
            f"Beats noise. The same search reached this Sharpe on only {beaten} of "
            f"{len(null)} structureless histories (p = {p_value:.3f})."
        )
        advice = ""
    elif p_value <= 0.10:
        headline = (
            f"Marginal. The same search on structureless data reached this Sharpe "
            f"{p_value:.0%} of the time (p = {p_value:.3f})."
        )
        advice = "Suggestive, not established. Extend the sample before committing capital."
    else:
        headline = (
            f"Indistinguishable from noise. Running this exact search on {len(null)} "
            f"structureless histories produced a result this good {p_value:.0%} of the time "
            f"(p = {p_value:.2f})."
        )
        advice = (
            "The search finds this much on data that contains nothing. The result is a "
            "property of the search, not of the market."
        )

    return Finding(
        name="permutation",
        title="Search on noise",
        score=score,
        headline=headline,
        advice=advice,
        detail={
            "p_value": p_value,
            "p_value_floor": floor,
            "n_runs": len(null),
            "n_beaten": beaten,
            "method": method,
            "block_size": block_size,
            "real_best_sharpe_annual": annualise(real_best, bpy),
            "null_best_median_annual": annualise(float(np.median(null)), bpy),
            "null_best_p95_annual": annualise(float(np.percentile(null, 95)), bpy),
        },
    )


# --------------------------------------------------------------------------------------
# 6. Regime -- is the P&L spread out, or is it one lucky window?
# --------------------------------------------------------------------------------------


def _bulk_scale(net: np.ndarray, min_active: int = 30) -> float:
    """A scale estimate for the body of the return distribution, not its tails.

    The concentration benchmark asks what share of P&L the top bars would carry if the
    series were ordinary. Estimating "ordinary" with the sample standard deviation lets
    the outliers set their own benchmark: ten bars carrying a decade of profit inflate the
    variance, which inflates the expected share, which hides the ten bars. Testing for
    outliers with a statistic they contaminate is the standard version of this mistake.

    So the scale comes from the median absolute deviation, which the tails cannot move,
    rescaled by 1.4826 to match the standard deviation of a normal.

    Bars where the strategy was flat are excluded first. A rule that is out of the market
    four days in five has a median absolute deviation of exactly zero, and its scale would
    otherwise be estimated as "no variation at all" from the bars where, by construction,
    nothing could have happened.
    """
    active = net[net != 0.0]
    if len(active) < min_active:
        sd = float(np.std(net, ddof=1)) if len(net) > 1 else 0.0
        return sd
    mad = float(np.median(np.abs(active - np.median(active))))
    if mad > 0:
        return 1.4826 * mad
    return float(np.std(active, ddof=1)) if len(active) > 1 else 0.0


def check_regime(sweep_: Sweep, index: int, *, n_chunks: int = 6) -> Finding:
    """Split the history into contiguous periods and look for a strategy that only ever
    worked once, plus the handful of bars carrying the entire result."""
    net = sweep_.returns[:, index].astype(np.float64)
    bpy = sweep_.spec.bars_per_year
    T = len(net)

    edges = np.linspace(0, T, n_chunks + 1).astype(int)
    chunks = []
    for i in range(n_chunks):
        seg = net[edges[i]:edges[i + 1]]
        chunks.append({
            "index": i,
            "sharpe": annualise(sharpe(seg), bpy),
            "total_return": float(np.sum(seg)),
            "bars": len(seg),
        })

    positive = sum(1 for c in chunks if c["total_return"] > 0)
    consistency = positive / n_chunks

    total = float(np.sum(net))
    top_n = max(1, int(0.01 * T))
    if total > 0:
        top_sum = float(np.sum(np.sort(net)[-top_n:]))
        concentration = top_sum / total
    else:
        concentration = math.inf

    # How concentrated *should* the P&L be? Comparing the raw share against a fixed
    # threshold looks like a test of lumpiness and is really a test of Sharpe wearing a
    # disguise: the numerator is set by the noise and the denominator by the edge, so the
    # ratio is roughly one over the signal-to-noise no matter how evenly the profit
    # arrived. Run at scale that showed up immediately -- against a 1.5 cutoff the leg
    # failed nine cells in ten, and since the verdict is a geometric mean it was pulling
    # every score down by a near-constant factor while looking like a finding.
    #
    # The scale-free question is whether the concentration exceeds what an ordinary series
    # with this mean and this variance would produce anyway. For T iid draws from
    # N(mu, sd), the expected sum of the top q-fraction is k*mu + T*sd*phi(z), with
    # z = Phi^-1(1-q) -- so the expected share divides that by T*mu. Coming in at the
    # expected share means the profit arrived exactly as unremarkably as noise would
    # deliver it, which is the honest null. Genuine lumpiness -- one trade carrying a
    # decade -- still lands far above it.
    expected = math.inf
    if total > 0 and T > 1:
        mu = float(np.mean(net))
        sd = _bulk_scale(net)
        if mu > 0 and sd > 0:
            q = top_n / T
            z = float(_norm.isf(q))
            expected = (top_n * mu + T * sd * float(_norm.pdf(z))) / (T * mu)

    if math.isfinite(concentration) and math.isfinite(expected) and expected > 0:
        excess = concentration / expected
        # Fat tails are normal in returns, so the band is wide: at one and a half times
        # the null nothing is said, and it takes four times before the leg is failed
        # outright.
        conc_score = float(np.clip((4.0 - excess) / 2.5, 0.0, 1.0))
    else:
        excess = math.inf
        conc_score = 0.0

    # Both legs matter: consistent across periods, and not carried by a few bars.
    score = float(np.sqrt(max(consistency, 1e-6) * max(conc_score, 1e-6)))

    bits = [f"Profitable in {positive}/{n_chunks} periods"]
    if math.isfinite(concentration):
        bits.append(
            f"the best 1% of bars ({top_n}) carry {concentration:.0%} of all P&L"
            + (f", {excess:.1f}x what this return distribution alone would give"
               if math.isfinite(excess) else "")
        )
    headline = "; ".join(bits) + "."

    if consistency <= 0.5:
        headline = (
            f"Worked in only {positive} of {n_chunks} periods. " +
            (f"The best 1% of bars carry {concentration:.0%} of the P&L. "
             if math.isfinite(concentration) else "") +
            "This is one regime, not an edge."
        )
        advice = "Check what was happening in the winning window. That is the only thing this trades."
    elif math.isfinite(concentration) and concentration > 0.8:
        advice = (
            "Almost all the profit comes from a handful of bars. Miss those fills and the "
            "strategy is flat — which is what usually happens live."
        )
    else:
        advice = ""

    return Finding(
        name="regime",
        title="Regime spread",
        score=score,
        headline=headline,
        advice=advice,
        detail={
            "consistency": consistency,
            "periods_profitable": positive,
            "n_periods": n_chunks,
            "top1pct_pnl_share": concentration,
            "expected_top1pct_share": expected,
            "excess_concentration": excess,
            "chunks": chunks,
            "max_drawdown": max_drawdown(net),
        },
    )
