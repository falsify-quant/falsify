"""Live monitoring: has the running bot drifted from the thing that was tested?

What this deliberately does NOT do
----------------------------------
It does not tell you the edge has decayed. It cannot, and neither can anything else.
`falsify` already computes the Minimum Track Record Length, and for the strategies this
was built against the answer was 24 years on one and 467 on another. At those Sharpes a
drawdown is indistinguishable from bad luck for longer than you will be alive, so a
monitor that flashes red when the equity curve dips is generating noise and calling it
risk. Watching a Sharpe ratio month to month is astrology with a p-value.

What is detectable in days rather than decades
----------------------------------------------
    costs      Realised fees and slippage against what the backtest assumed. Needs a
               handful of fills, not a track record, and it is the single most common
               reason a live strategy underperforms its backtest.
    divergence Did the bot actually take the positions its own strategy called for?
               Replay the strategy over the live window and compare bar by bar. This
               finds bugs, missed cycles, stale data and silent exception swallowing --
               all of which are certainties over a long enough run, and none of which
               show up in a P&L chart until they have cost real money.
    heartbeat  Is it still running at all? A bot that has quietly stopped looks exactly
               like a bot with no signals.
    envelope   Is the equity curve outside what the backtest's own return distribution
               says is plausible? Bootstrapped, not a t-test, and honest about being a
               coarse instrument on short samples.

Three of those four are exact. Only the last is statistical, and it is deliberately
scoped to catch gross breakage rather than to adjudicate performance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .sim import simulate
from .spec import Bars, MarketSpec

__all__ = ["Fill", "Veto", "LiveRecord", "Alert", "MonitorVerdict", "monitor",
           "strategy_eras", "config_age"]

SEVERITY = {"ok": 0, "watch": 1, "alarm": 2}


@dataclass(frozen=True)
class Fill:
    ts: float
    symbol: str
    side: str  # "BUY" | "SELL"
    qty: float  # base units
    price: float  # fill price
    fee: float  # absolute, quote currency
    ref_price: float | None = None  # price the signal saw, if known -> slippage
    reason: str = ""  # what the bot logged; used to date configuration changes

    @property
    def notional(self) -> float:
        return abs(self.qty * self.price)

    @property
    def fee_rate(self) -> float:
        return self.fee / self.notional if self.notional > 0 else 0.0

    @property
    def slippage_rate(self) -> float | None:
        """Signed cost of the fill against the price the decision was made at.

        Positive means it went against you: paid above reference on a buy, received
        below it on a sell.
        """
        if not self.ref_price or self.ref_price <= 0:
            return None
        d = (self.price - self.ref_price) / self.ref_price
        return d if self.side.upper() == "BUY" else -d


@dataclass
class Veto:
    """A signal the bot deliberately did not act on: risk cap, cooldown, sector limit."""

    ts: float
    symbol: str
    reason: str


@dataclass
class LiveRecord:
    """What actually happened, normalised. Build with an adapter or by hand."""

    fills: list[Fill] = field(default_factory=list)
    equity_ts: np.ndarray = field(default_factory=lambda: np.array([]))
    equity: np.ndarray = field(default_factory=lambda: np.array([]))
    vetoes: list[Veto] = field(default_factory=list)
    name: str = "live"

    def __post_init__(self) -> None:
        self.fills = sorted(self.fills, key=lambda f: f.ts)
        self.equity_ts = np.asarray(self.equity_ts, dtype=np.float64)
        self.equity = np.asarray(self.equity, dtype=np.float64)
        if len(self.equity_ts) != len(self.equity):
            raise ValueError("equity_ts and equity must be the same length")
        if len(self.equity_ts) > 1:
            order = np.argsort(self.equity_ts)
            self.equity_ts, self.equity = self.equity_ts[order], self.equity[order]

    def for_symbol(self, symbol: str) -> list[Fill]:
        return [f for f in self.fills if f.symbol == symbol]

    def vetoes_for(self, symbol: str, t0: float, t1: float) -> list[Veto]:
        return [v for v in self.vetoes if v.symbol == symbol and t0 <= v.ts <= t1]

    def trading_returns(self, flow_threshold: float = 0.20
                        ) -> tuple[np.ndarray, list[tuple[float, float, float]]]:
        """Per-step returns with external cash flows removed.

        A live equity curve is not a return series. Deposits and withdrawals move it
        without anyone having traded, and a funding event dwarfs any edge: on the
        account this was built against, a $300 deposit reads as +203% in one step and
        turns a +5% month into +897%. Comparing that to a backtest is meaningless in
        the most flattering possible direction.

        Steps larger than `flow_threshold` are treated as flows and chain-linked around
        rather than counted. That is a heuristic, so the flows found are returned for
        inspection rather than silently dropped -- a genuinely violent market move on a
        concentrated book can trip it, and you should be the one to decide.
        """
        eq = self.equity
        if len(eq) < 2:
            return np.array([]), []

        raw = eq[1:] / eq[:-1] - 1.0
        is_flow = np.abs(raw) > flow_threshold
        flows = [(float(self.equity_ts[i + 1]), float(eq[i]), float(eq[i + 1]))
                 for i in np.flatnonzero(is_flow)]
        return raw[~is_flow], flows

    @property
    def span_days(self) -> float:
        if len(self.equity_ts) > 1:
            return float(self.equity_ts[-1] - self.equity_ts[0]) / 86400.0
        if len(self.fills) > 1:
            return (self.fills[-1].ts - self.fills[0].ts) / 86400.0
        return 0.0

    def held_series(self, symbol: str, ts: np.ndarray) -> np.ndarray:
        """Reconstruct 'was a position open' on a given timestamp grid, from fills.

        Deliberately binary rather than a weight. The bot's position *size* comes from
        a separate sizing and risk layer; what is under test here is whether the signal
        fired, so comparing sizes would mix a signal bug with a sizing decision.
        """
        qty = 0.0
        out = np.zeros(len(ts), dtype=np.float64)
        fills = self.for_symbol(symbol)
        i = 0
        for k, t in enumerate(ts):
            while i < len(fills) and fills[i].ts <= t:
                f = fills[i]
                qty += f.qty if f.side.upper() == "BUY" else -f.qty
                i += 1
            out[k] = 1.0 if qty > 1e-9 else 0.0
        return out


@dataclass
class Alert:
    name: str
    severity: str  # "ok" | "watch" | "alarm"
    headline: str
    detail: dict = field(default_factory=dict)
    advice: str = ""


@dataclass
class MonitorVerdict:
    status: str
    alerts: list[Alert]
    record: LiveRecord
    notes: list[str] = field(default_factory=list)

    @property
    def ordered(self) -> list[Alert]:
        return sorted(self.alerts, key=lambda a: -SEVERITY[a.severity])


# --------------------------------------------------------------------------------------
# 0. Configuration age -- run this first or everything below it lies
# --------------------------------------------------------------------------------------

_SYMBOLISH = re.compile(r"^[A-Z0-9]{2,10}[-/][A-Z0-9]{2,6}\b[:,]?\s*")
_NUMERIC = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def _signature(reason: str, words: int = 3) -> str:
    """Collapse a log line to the shape of the behaviour that produced it.

    Strips a leading symbol and all numbers, so "trend up: px 95.14 > sma50 79.38" and
    "trend up: px 3.665 > sma50 2.99" land in the same bucket while an LLM's prose does
    not accidentally join them.
    """
    s = _SYMBOLISH.sub("", (reason or "").strip())
    s = _NUMERIC.sub("#", s).lower()
    return " ".join(s.replace(":", " ").split()[:words]) or "(none)"


def strategy_eras(record: "LiveRecord") -> list[tuple[str, float, float, int]]:
    """(signature, first_seen, last_seen, count) for every distinct behaviour in the fills.

    A strategy change does not announce itself. It shows up as a new shape of log line
    appearing and an old one stopping, which is exactly what this surfaces.
    """
    buckets: dict[str, list[float]] = {}
    for f in record.fills:
        buckets.setdefault(_signature(f.reason), []).append(f.ts)
    return sorted(
        ((sig, min(ts), max(ts), len(ts)) for sig, ts in buckets.items()),
        key=lambda e: e[1],
    )


def config_age(record: "LiveRecord", since: float | None, now: float | None = None) -> Alert:
    """How much of the live record was produced by the configuration running today.

    This has to come first, because every other check compares live behaviour against a
    backtest of the CURRENT settings. Run it across a configuration change and the
    divergence test will scream about a strategy that was replaced weeks ago -- which is
    exactly what happened the first time this was pointed at a real bot: it compared
    today's mechanical trend engine against a month in which an LLM was making the
    decisions, and reported it as a failed exit.
    """
    if not record.fills:
        return Alert("config", "watch", "No fills — nothing to date a configuration from.",
                     {"n_fills": 0})

    now = now if now is not None else record.fills[-1].ts
    eras = strategy_eras(record)
    newest = max(eras, key=lambda e: e[1])
    detail = {
        "eras": [(s, f, l, n) for s, f, l, n in eras],
        "newest_signature": newest[0],
        "newest_first_seen": newest[1],
        "since": since,
    }

    if since is None:
        return Alert(
            "config", "watch",
            f"No configuration date given. The fills contain {len(eras)} distinct "
            f"behaviours, the newest first seen {(now - newest[1])/86400:.1f} days ago — "
            "if that was a strategy change, everything below is comparing across it.",
            detail,
            "Pass `since=` the timestamp of the last config change so the comparison "
            "covers one strategy rather than a blend of several.",
        )

    in_window = [f for f in record.fills if f.ts >= since]
    days = (now - since) / 86400.0
    detail.update({"days_live": days, "fills_in_window": len(in_window),
                   "fills_total": len(record.fills)})

    if len(in_window) < 5:
        return Alert(
            "config", "watch",
            f"Current configuration has been live {days:.1f} days and produced "
            f"{len(in_window)} fill(s) — out of {len(record.fills)} in the record. "
            "Too little to compare against anything.",
            detail,
            "The checks below cover only this window. Everything earlier belongs to a "
            "strategy that is no longer running and cannot be used as evidence for it.",
        )
    return Alert(
        "config", "ok",
        f"Current configuration live {days:.1f} days, {len(in_window)} of "
        f"{len(record.fills)} fills. Comparisons below are scoped to it.",
        detail,
    )


# --------------------------------------------------------------------------------------
# 1. Cost audit -- few fills needed, highest yield
# --------------------------------------------------------------------------------------


def audit_costs(record: LiveRecord, spec: MarketSpec, *, tolerance: float = 0.25) -> Alert:
    """Compare realised fees and slippage against the modelled cost spec.

    Works from the first handful of fills. A backtest built on maker fees while the
    live path crosses the spread is the most common way a good strategy turns into a
    losing one, and it shows up here long before it shows up in the equity curve.
    """
    fills = record.fills
    if not fills:
        return Alert("costs", "watch", "No fills yet — nothing to audit.",
                     {"n_fills": 0})

    notional = sum(f.notional for f in fills)
    fees = sum(f.fee for f in fills)
    realised_fee = fees / notional if notional > 0 else 0.0

    slips = [f.slippage_rate for f in fills if f.slippage_rate is not None]
    realised_slip = float(np.mean(slips)) if slips else None

    modelled = spec.fee
    realised_total = realised_fee + (realised_slip if realised_slip is not None else 0.0)
    modelled_total = spec.cost_per_turnover

    by_side = {}
    for side in ("BUY", "SELL"):
        s = [f for f in fills if f.side.upper() == side]
        if s:
            n = sum(f.notional for f in s)
            by_side[side] = sum(f.fee for f in s) / n if n > 0 else 0.0

    detail = {
        "n_fills": len(fills),
        "total_notional": notional,
        "total_fees": fees,
        "realised_fee_bps": realised_fee * 1e4,
        "modelled_fee_bps": modelled * 1e4,
        "realised_slippage_bps": realised_slip * 1e4 if realised_slip is not None else None,
        "modelled_half_spread_bps": spec.half_spread * 1e4,
        "realised_total_bps": realised_total * 1e4,
        "modelled_total_bps": modelled_total * 1e4,
        "buy_fee_bps": by_side.get("BUY", 0.0) * 1e4,
        "sell_fee_bps": by_side.get("SELL", 0.0) * 1e4,
    }

    if modelled_total <= 0:
        return Alert("costs", "watch", "Cost spec has no cost to compare against.", detail)

    ratio = realised_total / modelled_total
    diff_bps = (realised_total - modelled_total) * 1e4

    if ratio > 1 + tolerance:
        return Alert(
            "costs", "alarm",
            f"Costing {diff_bps:+.0f} bps per unit turnover MORE than the backtest assumed "
            f"({realised_total*1e4:.0f} vs {modelled_total*1e4:.0f} bps, {ratio:.2f}x).",
            detail,
            "Re-run the backtest at the realised rate before trusting any of its numbers. "
            "At this gap the strategy that was tested is not the one that is running.",
        )
    if ratio < 1 - tolerance:
        return Alert(
            "costs", "ok",
            f"Costing {abs(diff_bps):.0f} bps per unit turnover LESS than modelled "
            f"({realised_total*1e4:.0f} vs {modelled_total*1e4:.0f} bps). The backtest is "
            "pessimistic, so its results are a floor rather than a forecast.",
            detail,
            "Worth re-running the backtest at the realised rate — a strategy that failed "
            "on cost margin may clear it once the real fills are used.",
        )
    return Alert(
        "costs", "ok",
        f"Fills match the model: {realised_total*1e4:.0f} bps realised against "
        f"{modelled_total*1e4:.0f} bps assumed, over {len(fills)} fills.",
        detail,
    )


# --------------------------------------------------------------------------------------
# 2. Replay divergence -- the one no P&L chart will show you
# --------------------------------------------------------------------------------------


def replay_divergence(
    strategy,
    bars: Bars,
    record: LiveRecord,
    symbol: str,
    params: dict,
    *,
    tolerate_bars: int = 2,
) -> Alert:
    """Replay the strategy over the live window and compare to what the bot did.

    The comparison is exact and needs no statistics: on each bar the strategy either
    wanted a position or it did not, and the bot either had one or it did not.

    `tolerate_bars` forgives short transitions -- a fill lands somewhere inside the bar
    that decided it, and a poll interval is not instantaneous. Sustained disagreement is
    the signal; a one-bar lag at each flip is just execution.

    The two directions of disagreement mean completely different things and must not be
    added together:

        MISSED  strategy wanted a position, bot was flat. Usually the risk layer doing
                its job -- position caps, daily spend caps, cooldowns, a sector limit,
                or the pair being outside that week's universe. Expected, not a fault.
        EXTRA   bot held a position the strategy did not want. This is the dangerous
                one: a failed exit, a stop that never fired, a sell that errored and was
                never retried. It is the direction that loses money quietly.

        An earlier version of this function summed them and fired ALARM at 42%
        agreement on a bot whose risk caps had vetoed 799 entries. It was measuring the
        portfolio layer and calling it a bug.
    """
    if bars.ts is None:
        return Alert("divergence", "watch",
                     "Bars have no timestamps — cannot align the replay to live fills.", {})

    fills = record.for_symbol(symbol)
    if not fills:
        return Alert("divergence", "ok",
                     f"{symbol}: no live fills in the window — nothing to diverge from.",
                     {"n_fills": 0})

    start = min(f.ts for f in fills)
    if len(record.equity_ts):
        start = min(start, float(record.equity_ts[0]))
    end = float(bars.ts[-1])

    want_full = np.nan_to_num(np.asarray(strategy(bars, **params), dtype=np.float64))
    mask = (bars.ts >= start) & (bars.ts <= end)
    if mask.sum() < 5:
        return Alert("divergence", "watch",
                     f"{symbol}: live window covers only {int(mask.sum())} bars.",
                     {"bars": int(mask.sum())})

    ts = bars.ts[mask]
    want = (want_full[mask] > 0).astype(np.float64)
    have = record.held_series(symbol, ts)

    def episodes_of(flag: np.ndarray) -> list[tuple[int, int]]:
        out, i = [], 0
        while i < len(flag):
            if flag[i]:
                j = i
                while j + 1 < len(flag) and flag[j + 1]:
                    j += 1
                if j - i + 1 > tolerate_bars:
                    out.append((i, j))
                i = j + 1
            else:
                i += 1
        return out

    missed_eps = episodes_of((want > 0) & (have == 0))
    extra_eps = episodes_of((want == 0) & (have > 0))

    missed_bars = sum(b - a + 1 for a, b in missed_eps)
    extra_bars = sum(b - a + 1 for a, b in extra_eps)

    # How much of the "wanted but flat" time the bot explicitly refused on purpose.
    explained = 0
    for a, b in missed_eps:
        if record.vetoes_for(symbol, float(ts[a]), float(ts[b])):
            explained += b - a + 1
    explained_frac = explained / missed_bars if missed_bars else 1.0

    detail = {
        "symbol": symbol,
        "bars_compared": int(len(want)),
        "missed_bars": int(missed_bars),
        "missed_episodes": len(missed_eps),
        "missed_explained_by_veto": explained_frac,
        "extra_bars": int(extra_bars),
        "extra_episodes": len(extra_eps),
        "agreement": 1.0 - (missed_bars + extra_bars) / len(want),
    }
    if extra_eps:
        a, b = max(extra_eps, key=lambda e: e[1] - e[0])
        detail["worst_extra_from_ts"] = float(ts[a])
        detail["worst_extra_to_ts"] = float(ts[b])

    if extra_bars > tolerate_bars:
        return Alert(
            "divergence", "alarm",
            f"{symbol}: the bot held a position the strategy did not want on "
            f"{extra_bars} of {len(want)} bars, across {len(extra_eps)} episode(s). "
            "An exit that should have fired did not.",
            detail,
            "Check the decision log at the longest episode for a failed or vetoed SELL. "
            "This is the direction that loses money without showing up as an error.",
        )

    if missed_bars == 0:
        return Alert("divergence", "ok",
                     f"{symbol}: the bot held exactly what the strategy called for across "
                     f"all {len(want)} bars.", detail)

    if explained_frac >= 0.8:
        return Alert(
            "divergence", "ok",
            f"{symbol}: no unwanted positions. The bot skipped {missed_bars} bars the "
            f"strategy wanted, and {explained_frac:.0%} of that is explained by its own "
            "risk vetoes — the portfolio layer working, not a fault.",
            detail,
        )

    return Alert(
        "divergence", "watch",
        f"{symbol}: no unwanted positions, but {missed_bars} of {len(want)} bars where the "
        f"strategy wanted a position and the bot was flat, only {explained_frac:.0%} of it "
        "matched to a logged veto.",
        detail,
        "Unexplained skips are usually a missed cycle, a stale candle, or a swallowed "
        "exception — none of which appear in P&L. Cross-check the journal for those windows.",
    )


# --------------------------------------------------------------------------------------
# 3. Heartbeat
# --------------------------------------------------------------------------------------


def heartbeat(record: LiveRecord, *, expected_interval_s: float, now: float | None = None,
              gap_multiple: float = 4.0) -> Alert:
    """Detect a bot that has quietly stopped, which looks identical to one with no signals."""
    ts = record.equity_ts
    if len(ts) < 3:
        return Alert("heartbeat", "watch", "Not enough equity snapshots to judge liveness.",
                     {"n": int(len(ts))})

    now = now if now is not None else float(ts[-1])
    gaps = np.diff(ts)
    median_gap = float(np.median(gaps))
    worst_gap = float(np.max(gaps))
    since_last = now - float(ts[-1])
    threshold = max(expected_interval_s, median_gap) * gap_multiple

    detail = {
        "snapshots": int(len(ts)),
        "median_gap_s": median_gap,
        "worst_gap_s": worst_gap,
        "since_last_s": since_last,
        "expected_interval_s": expected_interval_s,
        "long_gaps": int(np.count_nonzero(gaps > threshold)),
    }

    if since_last > threshold:
        return Alert("heartbeat", "alarm",
                     f"No equity snapshot for {since_last/3600:.1f}h — expected roughly "
                     f"every {median_gap/60:.0f}m. The bot may not be running.",
                     detail, "Check the service and the journal before anything else here "
                             "is worth reading.")
    if detail["long_gaps"]:
        return Alert("heartbeat", "watch",
                     f"{detail['long_gaps']} gap(s) longer than "
                     f"{threshold/3600:.1f}h in the record; worst was {worst_gap/3600:.1f}h.",
                     detail, "Downtime means missed signals the backtest assumed you took.")
    return Alert("heartbeat", "ok",
                 f"Alive: {len(ts):,} snapshots, median gap {median_gap/60:.0f}m, "
                 f"no outages.", detail)


# --------------------------------------------------------------------------------------
# 4. Equity envelope -- coarse, and says so
# --------------------------------------------------------------------------------------


def equity_envelope(record: LiveRecord, backtest_returns: np.ndarray,
                    *, n_boot: int = 5000, seed: int = 0) -> Alert:
    """Where does the live result sit inside the backtest's own return distribution?

    Bootstrapped over windows the same length as the live run, so the comparison is
    like for like. This is a breakage detector, not a performance review: on a short
    sample the envelope is so wide that almost nothing falls outside it, which is the
    honest answer rather than a shortcoming to be tuned away.
    """
    eq = record.equity
    if len(eq) < 10 or eq[0] <= 0:
        return Alert("envelope", "watch", "Not enough equity history to place the run.",
                     {"n": int(len(eq))})

    # Deposits and withdrawals are not returns. Strip them before comparing to anything.
    steps, flows = record.trading_returns()
    if len(steps) < 5:
        return Alert("envelope", "watch",
                     "Too little equity history left after removing external flows.",
                     {"n": int(len(steps)), "flows": len(flows)})

    live_return = float(np.prod(1.0 + steps) - 1.0)
    curve = np.cumprod(1.0 + steps)
    peak = np.maximum.accumulate(curve)
    live_dd = float(np.max(1.0 - curve / peak))

    r = np.asarray(backtest_returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 50:
        return Alert("envelope", "watch", "Backtest series too short to bootstrap.",
                     {"n_backtest": int(len(r))})

    # Match the live window in *bars*, using the backtest's own bar size.
    n = max(5, min(len(r) - 1, int(len(eq))))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r) - n, size=n_boot)
    paths = np.array([np.prod(1.0 + r[i:i + n]) - 1.0 for i in idx])

    pct = float(np.mean(paths <= live_return))
    detail = {
        "live_return": live_return,
        "live_max_drawdown": live_dd,
        "external_flows": len(flows),
        "gross_equity_change": float(eq[-1] / eq[0] - 1.0),
        "percentile": pct,
        "boot_p05": float(np.percentile(paths, 5)),
        "boot_median": float(np.percentile(paths, 50)),
        "boot_p95": float(np.percentile(paths, 95)),
        "window_bars": n,
        "n_boot": n_boot,
    }

    if pct < 0.01:
        return Alert(
            "envelope", "watch",
            f"Live return {live_return:+.1%} sits at the {pct:.1%} percentile of what the "
            f"backtest produced over windows this long. Below its 1st percentile.",
            detail,
            "Worth investigating, but a short window cannot separate a broken strategy "
            "from an ordinary bad patch. Check the exact tests above first — they can.",
        )
    flow_note = (f" ({len(flows)} deposit/withdrawal step(s) removed; raw equity moved "
                 f"{detail['gross_equity_change']:+.0%})" if flows else "")
    return Alert(
        "envelope", "ok",
        f"Trading return {live_return:+.1%} is at the {pct:.0%} percentile of the "
        f"backtest's own distribution over {n}-bar windows (5th–95th: "
        f"{detail['boot_p05']:+.1%} to {detail['boot_p95']:+.1%}). Unremarkable"
        f"{flow_note}.",
        detail,
    )


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def monitor(
    record: LiveRecord,
    spec: MarketSpec,
    *,
    strategy=None,
    bars_by_symbol: dict[str, Bars] | None = None,
    params: dict | None = None,
    backtest_returns: np.ndarray | None = None,
    expected_interval_s: float = 3600.0,
    since: float | None = None,
    now: float | None = None,
) -> MonitorVerdict:
    """Run every check the available data supports, and say which ones it did not.

    `since` is the timestamp of the last configuration change. Everything except the
    heartbeat is scoped to it, because comparing live behaviour to a backtest of the
    current settings is only meaningful over the period those settings were running.
    """
    full = record
    alerts: list[Alert] = [config_age(full, since, now=now)]
    notes: list[str] = []

    if since is not None:
        record = LiveRecord(
            fills=[f for f in full.fills if f.ts >= since],
            equity_ts=full.equity_ts[full.equity_ts >= since] if len(full.equity_ts) else full.equity_ts,
            equity=full.equity[full.equity_ts >= since] if len(full.equity_ts) else full.equity,
            vetoes=[v for v in full.vetoes if v.ts >= since],
            name=full.name,
        )

    alerts.append(audit_costs(record, spec))

    # Liveness is about the process, not the strategy, so it uses the whole record.
    if len(full.equity_ts):
        alerts.append(heartbeat(full, expected_interval_s=expected_interval_s, now=now))
    else:
        notes.append("No equity snapshots — liveness not checked.")

    if strategy is not None and bars_by_symbol and params is not None:
        for symbol, bars in bars_by_symbol.items():
            alerts.append(replay_divergence(strategy, bars, record, symbol, params))
    else:
        notes.append("No strategy replay — divergence not checked, which is the check "
                     "that finds bugs.")

    if backtest_returns is not None:
        alerts.append(equity_envelope(record, backtest_returns))
    else:
        notes.append("No backtest series supplied — equity envelope not computed.")

    notes.append(
        "Edge decay is deliberately not tested. At the Sharpe ratios these strategies "
        "post, distinguishing decay from noise takes decades of live data — any monitor "
        "claiming to detect it on months is reporting noise."
    )

    worst = max((SEVERITY[a.severity] for a in alerts), default=0)
    status = {0: "OK", 1: "WATCH", 2: "ALARM"}[worst]
    return MonitorVerdict(status=status, alerts=alerts, record=record, notes=notes)
