"""The asset universe, chosen by rule rather than by taste.

Picking instruments is the easiest place to accidentally decide the answer, so the choices
here are constrained in advance and the constraints are written down.

**Breadth over quality.** The equity list leads with broad ETFs spanning equity, bonds,
gold and oil, because an index is the market rather than a survivor of it. Single names
follow, chosen for long histories across different sectors.

**Losers on purpose.** GE, F, INTC and XOM are in the list precisely because their last
two decades were bad. A universe of today's winners answers "does trend following work on
things that went up", which is not the question.

**The survivorship problem is real and only partly fixable.** Every name here is one that
still trades. Companies that went to zero are absent, and delisted crypto is absent twice
over -- exchanges remove the pairs and the history with them. This biases long-biased
rules upward. ETFs mitigate it; nothing available without a paid survivorship-free
database eliminates it. The study says so in its own conclusions rather than burying it.

**No date-range choices.** Every series takes as much history as the free source will
give, ending today. Choosing a window is choosing a regime.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Asset", "EQUITIES", "CRYPTO", "FUTURES", "ALL", "by_class"]


@dataclass(frozen=True)
class Asset:
    symbol: str
    asset_class: str  # equity | crypto | futures
    kind: str  # index | sector | commodity | bond | single-name | major | alt | metal
    #            | grain | financial
    note: str = ""

    @property
    def cadence(self) -> str:
        return "hourly" if self.asset_class == "crypto" else "daily"


EQUITIES: list[Asset] = [
    # Broad exposure. The honest core of the study -- an index cannot survive its way
    # into the sample, because it is not a company.
    Asset("SPY", "equity", "index", "US large cap"),
    Asset("QQQ", "equity", "index", "US tech-weighted"),
    Asset("IWM", "equity", "index", "US small cap"),
    Asset("DIA", "equity", "index", "US mega cap"),
    Asset("EFA", "equity", "index", "developed markets ex-US"),
    Asset("EEM", "equity", "index", "emerging markets"),
    # Other asset classes, so the conclusions are not about equity beta wearing a costume.
    Asset("TLT", "equity", "bond", "20+ year Treasuries"),
    Asset("GLD", "equity", "commodity", "gold"),
    Asset("USO", "equity", "commodity", "crude oil; heavy contango drag"),
    # Single names across sectors, including four that spent the period going nowhere.
    Asset("AAPL", "equity", "single-name", "consumer tech"),
    Asset("MSFT", "equity", "single-name", "enterprise software"),
    Asset("JNJ", "equity", "single-name", "healthcare"),
    Asset("PG", "equity", "single-name", "consumer staples"),
    Asset("JPM", "equity", "single-name", "financials"),
    Asset("WMT", "equity", "single-name", "retail"),
    Asset("XOM", "equity", "single-name", "energy; a lost decade in the middle"),
    Asset("KO", "equity", "single-name", "staples; low volatility"),
    Asset("INTC", "equity", "single-name", "semis; a long decline"),
    Asset("GE", "equity", "single-name", "industrials; the canonical fallen giant"),
    Asset("F", "equity", "single-name", "autos; two decades sideways"),
]

CRYPTO: list[Asset] = [
    Asset("BTC-USD", "crypto", "major", ""),
    Asset("ETH-USD", "crypto", "major", ""),
    Asset("SOL-USD", "crypto", "alt", ""),
    Asset("ADA-USD", "crypto", "alt", ""),
    Asset("XRP-USD", "crypto", "alt", ""),
    Asset("DOGE-USD", "crypto", "alt", ""),
    Asset("LTC-USD", "crypto", "alt", "one of the few with a pre-2017 history"),
    Asset("LINK-USD", "crypto", "alt", ""),
    Asset("AVAX-USD", "crypto", "alt", ""),
    Asset("DOT-USD", "crypto", "alt", ""),
]

# Futures, added because six rules in the canon are futures systems that had no matching
# venue here -- Wilder and Lane developed on commodities, the Turtles traded futures, and
# LeBeau's book has it in the title.
#
# **The free data is only usable for some of them, and the cut is measured, not judged.**
# Yahoo's `=F` series are front-month splices carrying no roll return, so they track
# something spot-like that nobody can hold. Measured against a fund that does hold the
# asset, over the longest common window:
#
#     gold      GC=F vs GLD    +0.3%/yr      silver    SI=F vs SLV    +0.4%/yr
#     soybean   ZS=F vs SOYB   -0.9%/yr      copper    HG=F vs CPER   +1.2%/yr
#     corn      ZC=F vs CORN   +4.2%/yr      crude     CL=F vs USO    +7.9%/yr
#     nat gas   NG=F vs UNG   +23.3%/yr
#
# Anything above ~1.5%/yr is excluded: on `CL=F` the continuous series returns +20% over
# twenty years while USO returns -77%, so a long-biased rule would bank a return nobody
# could have earned. The control is `ES=F` vs `SPY` at -1.9%/yr, which is the dividend
# yield with the sign the method predicts -- that is what says the number is carry rather
# than noise.
#
# The financial contracts carry no ETF comparison, and are included on the different
# ground that their basis is a financing rate rather than a storage cost: small,
# mechanical, and without the storage dynamics that produce a 23%/yr wedge in gas.
#
# What survives is a real limitation and is stated in the study: energy and the grains are
# where several of these systems were actually developed, and they are exactly where the
# free data cannot support the test.
FUTURES: list[Asset] = [
    Asset("GC=F", "futures", "metal", "gold; roll drag 0.3%/yr measured against GLD"),
    Asset("SI=F", "futures", "metal", "silver; 0.4%/yr against SLV"),
    Asset("HG=F", "futures", "metal", "copper; 1.2%/yr against CPER"),
    Asset("ZS=F", "futures", "grain", "soybeans; -0.9%/yr against SOYB, mild backwardation"),
    Asset("ES=F", "futures", "financial", "S&P 500; the roll-drag control"),
    Asset("ZN=F", "futures", "financial", "10-year Treasury note"),
    Asset("6E=F", "futures", "financial", "euro FX"),
]

ALL: list[Asset] = EQUITIES + CRYPTO + FUTURES


def by_class(asset_class: str) -> list[Asset]:
    return [a for a in ALL if a.asset_class == asset_class]
