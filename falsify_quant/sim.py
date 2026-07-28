"""The fill simulator: target weights -> the returns you would actually have earned.

This module is small on purpose. It is the single place where execution timing and cost
accounting live, so there is exactly one thing to audit and no way for a strategy to
quietly cheat on either.

Timing convention
-----------------
    w[t]        weight chosen at the close of bar t, from information <= t
    p[t]        position actually held during bar t   =  w[t-1]
    r[t]        return of bar t
    gross[t]    p[t] * r[t]
    turnover[t] |p[t] - p[t-1]|      the trade done at the close of bar t-1
    net[t]      gross[t] - turnover[t]*cost - carry(p[t])

The one-bar lag between deciding and holding is applied here, once. This is the most
common source of fake alpha in retail backtests: deciding on bar t's close and booking
bar t's return.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spec import Bars, MarketSpec

__all__ = ["SimResult", "simulate"]


@dataclass(frozen=True)
class SimResult:
    net: np.ndarray  # per-bar net returns, the only series the statistics see
    gross: np.ndarray  # before costs
    turnover: np.ndarray  # per-bar, in units of capital
    position: np.ndarray  # p[t], the weight actually held during bar t
    spec: MarketSpec

    @property
    def total_cost(self) -> float:
        return float(np.sum(self.gross - self.net))

    @property
    def gross_return(self) -> float:
        return float(np.sum(self.gross))

    @property
    def net_return(self) -> float:
        return float(np.sum(self.net))

    @property
    def n_trades(self) -> int:
        """Number of bars on which the position actually moved."""
        return int(np.count_nonzero(self.turnover > 1e-12))

    @property
    def avg_turnover(self) -> float:
        """Mean turnover per bar. Drives how hard costs bite."""
        return float(np.mean(self.turnover))

    @property
    def edge_per_turnover(self) -> float:
        """Gross return earned per unit of capital churned.

        This is the number that decides whether a strategy is real. Compare it to
        `spec.cost_per_turnover`. If your edge per unit of turnover is 4 bps and it
        costs you 6 bps to trade, no amount of parameter tuning will save you --
        you are paying for the privilege of being right.
        """
        churn = float(np.sum(self.turnover))
        if churn <= 1e-12:
            return 0.0
        return self.gross_return / churn

    @property
    def equity(self) -> np.ndarray:
        """Compounded equity curve, starting at 1.0."""
        return np.cumprod(1.0 + self.net)


def _lag(a: np.ndarray, k: int = 1) -> np.ndarray:
    """Shift forward by k bars, filling the front with zeros (flat before you start)."""
    if k <= 0:
        return a
    out = np.zeros_like(a)
    out[k:] = a[:-k]
    return out


def simulate(
    bars: Bars,
    weights: np.ndarray,
    spec: MarketSpec,
    *,
    extra_lag: int = 0,
) -> SimResult:
    """Run target weights through the cost model and return the realised net series.

    `extra_lag` delays every decision by additional bars. It is the instrument of the
    lookahead test: a strategy with a genuine multi-bar edge degrades gently as lag
    grows, while one that is peeking at the current bar falls off a cliff at lag 1.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (len(bars),):
        raise ValueError(f"weights has shape {w.shape}, expected ({len(bars)},)")

    # NaN is the conventional "still warming up" signal -- treat as flat rather than
    # poisoning the whole series. Infinities are a bug, not a convention.
    if np.any(np.isinf(w)):
        raise ValueError("weights contain inf")
    w = np.nan_to_num(w, nan=0.0)

    r = bars.returns
    p = _lag(w, 1 + extra_lag)  # position held during each bar

    gross = p * r

    turnover = np.abs(p - _lag(p, 1))
    cost = turnover * spec.cost_per_turnover

    if spec.impact_coef > 0:
        # Rough super-linear penalty for large rebalances. A faithful impact model needs
        # ADV and order size; this exists so that strategies which survive only by
        # assuming infinite depth are visibly punished.
        cost = cost + spec.impact_coef * np.power(turnover, 1.5)

    if spec.carry != 0.0:
        if spec.carry_on == "short":
            exposure = np.maximum(0.0, -p)  # equity borrow: only shorts pay
        else:
            exposure = np.abs(p)  # perp funding: approximate both sides paying
        cost = cost + spec.carry * exposure

    return SimResult(
        net=gross - cost,
        gross=gross,
        turnover=turnover,
        position=p,
        spec=spec,
    )
