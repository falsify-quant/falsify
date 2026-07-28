"""Core data types: bars, market cost specs, and the strategy contract.

The central idea in `falsify` is that a strategy is a *pure function from bars to
target weights*. Everything else -- costs, timing, P&L -- is the engine's job, not
the strategy's. That separation is what makes the whole prosecution possible: if the
engine owns the fill simulation, it can re-run your strategy under costs you did not
choose, on data you did not see, with lags you did not apply.

A strategy that computes its own P&L cannot be cross-examined. That is why most
retail backtests cannot be checked.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

__all__ = [
    "Bars",
    "MarketSpec",
    "Strategy",
    "CRYPTO_PERP_TAKER",
    "CRYPTO_SPOT_TAKER",
    "EQUITY_LIQUID",
    "EQUITY_SMALLCAP",
    "PRESETS",
]


# --------------------------------------------------------------------------------------
# Bars
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Bars:
    """OHLCV series. All arrays are 1-D, equal length, ordered oldest to newest.

    `ts` is epoch seconds (UTC). It is optional but strongly recommended -- the regime
    tests use it to label periods, and without it they fall back to positional slicing.
    """

    close: np.ndarray
    open: np.ndarray | None = None
    high: np.ndarray | None = None
    low: np.ndarray | None = None
    volume: np.ndarray | None = None
    ts: np.ndarray | None = None
    symbol: str = "?"

    def __post_init__(self) -> None:
        close = np.asarray(self.close, dtype=np.float64)
        object.__setattr__(self, "close", close)
        n = len(close)
        if n < 2:
            # Structural minimum only: one return needs two closes. The "enough data to
            # say anything honest" threshold lives in sweep(), because that is where
            # analysis happens -- a four-bar series is a legitimate object to hold and
            # slice, it is only dishonest to compute a Sharpe ratio from.
            raise ValueError(f"need at least 2 closes to form a return, got {n}")
        if not np.all(np.isfinite(close)):
            raise ValueError("close contains NaN or inf")
        if np.any(close <= 0):
            raise ValueError("close contains non-positive prices")

        for name in ("open", "high", "low", "volume", "ts"):
            arr = getattr(self, name)
            if arr is not None:
                arr = np.asarray(arr, dtype=np.float64)
                if len(arr) != n:
                    raise ValueError(f"{name} has length {len(arr)}, expected {n}")
                object.__setattr__(self, name, arr)

        if self.ts is not None and np.any(np.diff(self.ts) <= 0):
            raise ValueError("ts must be strictly increasing (sort your data)")

    def __len__(self) -> int:
        return len(self.close)

    @property
    def returns(self) -> np.ndarray:
        """Simple bar-over-bar returns. `returns[0]` is 0 by convention."""
        r = np.zeros_like(self.close)
        r[1:] = np.diff(self.close) / self.close[:-1]
        return r

    @property
    def inferred_bars_per_year(self) -> float | None:
        """Bars per year implied by the timestamps, or None if there are none.

        Counted over the whole span rather than from the gap between bars, so weekends
        and market holidays are handled without special-casing: 252 for daily equities,
        8,760 for hourly crypto, both from the same arithmetic.
        """
        if self.ts is None or len(self.ts) < 2:
            return None
        span = float(self.ts[-1] - self.ts[0])
        if span <= 0:
            return None
        return len(self.ts) / (span / (365.25 * 86400.0))

    def slice(self, start: int, stop: int) -> "Bars":
        cut = lambda a: None if a is None else a[start:stop]  # noqa: E731
        return Bars(
            close=self.close[start:stop],
            open=cut(self.open),
            high=cut(self.high),
            low=cut(self.low),
            volume=cut(self.volume),
            ts=cut(self.ts),
            symbol=self.symbol,
        )

    def with_close(self, close: np.ndarray) -> "Bars":
        """Replace the close series, dropping OHLV (which would no longer be consistent).

        Used by the permutation test, which manufactures synthetic price paths.
        """
        return Bars(close=np.asarray(close, dtype=np.float64), ts=self.ts, symbol=self.symbol)


# --------------------------------------------------------------------------------------
# Market / cost spec
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketSpec:
    """What it actually costs you to trade, and how often you get to.

    All rates are fractions of notional, not basis points, not percent. 0.0006 is 6 bps.

    `fee` and `half_spread` are charged per unit of turnover, where turnover is
    |w_t - w_{t-1}| -- so a flip from fully long to fully short is 2.0 turnover and pays
    twice. This is the correct accounting and it is the one retail backtests skip.
    """

    name: str
    asset_class: str  # "crypto" | "equity"
    bars_per_year: float

    fee: float = 0.0  # exchange/broker fee per side
    half_spread: float = 0.0  # half the quoted spread; you cross it to get filled
    impact_coef: float = 0.0  # sqrt-law impact, scaled by participation rate

    carry: float = 0.0  # per-bar financing on held positions (funding / borrow)
    carry_on: str = "abs"  # "abs" = both sides pay (perp funding, approx)
    #                        "short" = shorts only pay (equity borrow)

    @property
    def cost_per_turnover(self) -> float:
        """The all-in linear cost of moving 100% of capital once. The number that kills."""
        return self.fee + self.half_spread

    @property
    def bar_seconds(self) -> float:
        """Duration of one bar, implied by the calendar."""
        return 365.25 * 86400.0 / self.bars_per_year

    def scaled(self, factor: float) -> "MarketSpec":
        """Same market, costs multiplied. Drives the breakeven sweep."""
        return replace(
            self,
            name=f"{self.name} x{factor:g}",
            fee=self.fee * factor,
            half_spread=self.half_spread * factor,
            impact_coef=self.impact_coef * factor,
        )

    def at_bars_per_year(self, bars_per_year: float) -> "MarketSpec":
        """Retime the spec for a different bar size.

        Per-trade costs (fee, spread) are charged per unit of turnover and do not depend
        on how long a bar lasts. Carry does: funding and borrow accrue with wall-clock
        time, so a 6-hour bar owes six times what an hourly bar owes. Getting this
        backwards understates financing by the ratio of the bar sizes, which on a
        perpetual future is most of the cost of a slow strategy.
        """
        if bars_per_year <= 0:
            raise ValueError("bars_per_year must be positive")
        ratio = self.bars_per_year / bars_per_year  # old bars per new bar
        return replace(self, bars_per_year=bars_per_year, carry=self.carry * ratio)


# Realistic retail cost presets. These are deliberately not optimistic -- if your
# strategy only survives at rates better than these, it does not survive.

CRYPTO_PERP_TAKER = MarketSpec(
    name="crypto perp, taker, major CEX",
    asset_class="crypto",
    bars_per_year=365 * 24,  # hourly
    fee=0.00045,  # 4.5 bps taker
    half_spread=0.00005,  # ~0.5 bp on BTC/ETH; far worse on alts
    carry=0.0001 / 8,  # ~1 bp per 8h funding, spread per hour
    carry_on="abs",
)

CRYPTO_SPOT_TAKER = MarketSpec(
    name="crypto spot, taker, major CEX",
    asset_class="crypto",
    bars_per_year=365 * 24,
    fee=0.0006,  # 6 bps taker, standard retail tier
    half_spread=0.00005,
    carry=0.0,
)

EQUITY_LIQUID = MarketSpec(
    name="US equity, liquid large cap, IBKR",
    asset_class="equity",
    bars_per_year=252,  # daily
    fee=0.000035,  # ~0.35 bp; IBKR per-share on a ~$100 name
    half_spread=0.00005,  # ~0.5 bp on a penny-wide large cap
    carry=0.05 / 252,  # ~5%/yr borrow, shorts only
    carry_on="short",
)

EQUITY_SMALLCAP = MarketSpec(
    name="US equity, small cap",
    asset_class="equity",
    bars_per_year=252,
    fee=0.000035,
    half_spread=0.0015,  # 15 bps half-spread is normal down here and it ends most strategies
    impact_coef=0.1,
    carry=0.15 / 252,  # hard-to-borrow
    carry_on="short",
)

PRESETS: dict[str, MarketSpec] = {
    "crypto-perp": CRYPTO_PERP_TAKER,
    "crypto-spot": CRYPTO_SPOT_TAKER,
    "equity": EQUITY_LIQUID,
    "equity-smallcap": EQUITY_SMALLCAP,
}


# --------------------------------------------------------------------------------------
# Strategy contract
# --------------------------------------------------------------------------------------


class Strategy(Protocol):
    """Bars in, target weights out.

    Returns an array `w` of length len(bars), where `w[t]` is the fraction of capital you
    want to hold *going into bar t+1*, decided using information available at the close
    of bar t. 1.0 is fully long, -1.0 fully short, 0.0 flat. Leverage above 1 is allowed;
    the engine will not stop you, but it will charge you for it.

    The engine applies the one-bar execution lag itself. Do not lag inside your strategy
    or you will be lagged twice.

    Critically: `w[t]` must not depend on `bars.close[t+1:]`. The lookahead test exists
    because that rule gets broken constantly, usually by accident, usually via a
    centered rolling window or a fillna that back-propagates.
    """

    def __call__(self, bars: Bars, **params: float) -> np.ndarray: ...
