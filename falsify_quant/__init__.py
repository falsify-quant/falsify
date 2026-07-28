"""falsify -- prove your trading strategy is nothing, before the market does.

Every other backtesting tool helps you search for parameters that look good. This one
assumes you already did that, and tries to establish that what you found is an artefact
of the search. It has no optimiser and no knobs that improve a score.

    import falsify_quant
    from falsify_quant.spec import Bars, CRYPTO_PERP_TAKER

    def momentum(bars, fast=10, slow=50):
        c = bars.close
        f = moving_average(c, int(fast))
        s = moving_average(c, int(slow))
        return np.where(f > s, 1.0, -1.0)

    verdict = falsify_quant.run(
        momentum,
        bars=Bars(close=prices, ts=timestamps, symbol="BTC-USD"),
        spec=CRYPTO_PERP_TAKER,
        grid={"fast": [5, 10, 20, 40], "slow": [50, 100, 200]},
        valid=lambda p: p["fast"] < p["slow"],
    )
    print(verdict.score, verdict.label)
    falsify_quant.write_report(verdict, "verdict.html")

The strategy must return one target weight per bar, using only information available at
that bar. `falsify` applies the execution lag, charges the costs, and checks the
causality claim rather than taking your word for it.

Copyright (C) 2026 Joseph Langstroth

This program is free software: you can redistribute it and/or modify it under the terms
of the GNU Affero General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this
program. If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

from .attest import Attestation, attest, read_attestation, verify, write_attestation
from .harness import Sweep, sweep
from .prosecute import (
    Finding,
    check_causality,
    check_costs,
    check_deflation,
    check_pbo,
    check_permutation,
    check_regime,
)
from .report import write_report
from .score import Verdict, score_findings
from .spec import PRESETS, Bars, MarketSpec
from .stats import annualise, sharpe
from .universe import UniverseVerdict, run_universe

__version__ = "0.1.0"

__all__ = [
    "run",
    "run_on_sweep",
    "run_universe",
    "UniverseVerdict",
    "write_report",
    "attest",
    "verify",
    "Attestation",
    "read_attestation",
    "write_attestation",
    "Bars",
    "MarketSpec",
    "PRESETS",
    "Sweep",
    "Verdict",
    "Finding",
    "sweep",
]


def run(
    strategy,
    bars: Bars,
    spec: MarketSpec,
    grid: Mapping[str, Sequence[float]],
    *,
    valid: Callable[[dict], bool] | None = None,
    params: Mapping[str, float] | None = None,
    n_permutations: int = 100,
    permutation_method: str = "iid",
    n_blocks: int = 16,
    n_chunks: int = 6,
    seed: int = 0,
    strict_calendar: bool = True,
    progress: Callable[[str], None] | None = None,
) -> Verdict:
    """Run the full prosecution and return a verdict.

    `params` names the specific variant under examination. Leave it out and `falsify`
    examines the one you would have shipped: the highest in-sample Sharpe in the grid.

    `n_permutations` is the expensive knob -- each one re-runs the entire sweep on a
    synthetic history. A hundred resolves a p-value to 0.01.

    `strict_calendar` checks the bar spacing in your data against the one in your cost
    spec. Turn it off only if you know why they differ.
    """
    say = progress or (lambda _: None)
    _check_calendar(bars, spec, strict_calendar)

    say(f"sweeping {len(grid) and '×'.join(str(len(v)) for v in grid.values()) or '1'} grid")
    sw = sweep(strategy, bars, spec, grid, valid=valid)

    index = sw.index_of(params) if params else sw.best_index
    return run_on_sweep(
        sw,
        index,
        n_permutations=n_permutations,
        permutation_method=permutation_method,
        n_blocks=n_blocks,
        n_chunks=n_chunks,
        seed=seed,
        progress=progress,
    )


def run_on_sweep(
    sw: Sweep,
    index: int,
    *,
    n_permutations: int = 100,
    permutation_method: str = "iid",
    n_blocks: int = 16,
    n_chunks: int = 6,
    seed: int = 0,
    prior_sharpes=None,
    progress: Callable[[str], None] | None = None,
) -> Verdict:
    """Prosecute one variant of an already-completed sweep.

    `prior_sharpes` charges for searches you already ran on the same question. See
    `check_deflation` -- the short version is that a grid you ran, disliked and adjusted
    is still a grid you ran.
    """
    say = progress or (lambda _: None)
    findings: list[Finding] = []

    say("checking causality")
    causality = check_causality(sw, index)
    findings.append(causality)

    if causality.fatal and causality.score <= 0.0:
        # Everything downstream is computed from a P&L series that could not have existed.
        # Reporting a Sharpe here would be actively misleading.
        return score_findings(findings, meta=_meta(sw, index))

    say("checking costs")
    findings.append(check_costs(sw, index))

    say("deflating Sharpe")
    findings.append(check_deflation(sw, index, prior_sharpes=prior_sharpes))

    say("cross-validating selection")
    findings.append(check_pbo(sw, n_blocks=n_blocks))

    say(f"running the search on {n_permutations} synthetic histories")
    findings.append(
        check_permutation(sw, n_runs=n_permutations, method=permutation_method, seed=seed)
    )

    say("checking regime spread")
    findings.append(check_regime(sw, index, n_chunks=n_chunks))

    return score_findings(findings, meta=_meta(sw, index))


def _check_calendar(bars: Bars, spec: MarketSpec, strict: bool) -> None:
    """Catch a bar size that does not match the cost spec's calendar.

    Every annualised number -- Sharpe, the deflation benchmark, turnover per year -- is
    scaled by `spec.bars_per_year`. Load daily bars while the spec says hourly and the
    reported Sharpe is inflated by about six times, with nothing anywhere looking wrong.
    It is a one-line mistake that survives code review, so it is checked rather than
    trusted.
    """
    actual = bars.inferred_bars_per_year
    if actual is None:
        return  # no timestamps, nothing to check against

    ratio = actual / spec.bars_per_year
    if 0.6 < ratio < 1.7:
        return

    msg = (
        f"Bar spacing does not match the cost spec.\n"
        f"  data       ~{actual:,.0f} bars/year (from {len(bars):,} timestamps)\n"
        f"  spec       {spec.bars_per_year:,.0f} bars/year ({spec.name})\n"
        f"Every annualised figure would be off by about {max(ratio, 1/ratio):.1f}x. "
        f"Pick a preset matching your bars, or pass strict_calendar=False if this is "
        f"deliberate."
    )
    if strict:
        raise ValueError(msg)
    import warnings

    warnings.warn(msg, RuntimeWarning, stacklevel=3)


def _meta(sw: Sweep, index: int) -> dict:
    net = sw.returns[:, index].astype(float)
    return {
        "symbol": sw.bars.symbol,
        "market": sw.spec.name,
        "asset_class": sw.spec.asset_class,
        "bars": len(sw.bars),
        "bars_per_year": sw.spec.bars_per_year,
        "years": len(sw.bars) / sw.spec.bars_per_year,
        "n_trials": int((~sw.failed).sum()),
        "params": sw.params[index],
        "grid": {k: list(v) for k, v in sw.grid.items()},
        "sharpe_annual": annualise(sharpe(net), sw.spec.bars_per_year),
        "equity": (1.0 + net).cumprod().tolist(),
        "ts": sw.bars.ts.tolist() if sw.bars.ts is not None else None,
    }
