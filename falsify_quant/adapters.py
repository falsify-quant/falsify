"""Read a live record out of a running bot's sqlite database.

Written against the schema shared by Joseph's crypto and IBKR bots:

    trades            (id, pair, side, price, base_qty, quote_amount, fee, ts, reason)
    equity_snapshots  (ts, value_usd)
    decisions         (id, ts, pair, signal, action, reason)

The useful trick here is the `reason` column. The bots record the price the *signal*
saw alongside the fill:

    "trend up: px 3.665 > sma50 2.99018 (rising) > sma20 3.23475"
    "stop-maker:trend-break (px 65.04 < sma50 65.3046)"

Pulling that number out turns an unmeasurable into a measurable: the gap between the
price the decision was made at and the price the fill happened at is realised slippage,
which is otherwise invisible and is a large part of why live underperforms backtest.
It is a log line written for humans, being used as telemetry.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .monitor import Fill, LiveRecord, Veto

__all__ = ["load_bot_db", "SYMBOL_ALIASES"]

_PX = re.compile(r"\bpx\s+([0-9]*\.?[0-9]+)")

# The crypto bot trades -USDC on Coinbase Advanced Trade; history comes from Exchange,
# which quotes against USD and has the -USDC books delisted.
SYMBOL_ALIASES = {"-USDC": "-USD"}


def _to_epoch(v) -> float | None:
    """Accept unix seconds (int or float) or an ISO-8601 string."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.timestamp()


def _canonical(symbol: str, alias: bool) -> str:
    if not alias:
        return symbol
    for old, new in SYMBOL_ALIASES.items():
        if symbol.endswith(old):
            return symbol[: -len(old)] + new
    return symbol


def _tables(db: sqlite3.Connection) -> set[str]:
    return {r[0] for r in db.execute("select name from sqlite_master where type='table'")}


def load_bot_db(
    path: str | Path,
    *,
    alias_symbols: bool = True,
    since: float | None = None,
    name: str | None = None,
) -> LiveRecord:
    """Open a bot's trader.db read-only and normalise it into a LiveRecord.

    Read-only by URI, so pointing this at a database a live daemon is writing to cannot
    disturb it. `since` filters to a window; leave it out for the whole history.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    db = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    present = _tables(db)

    fills: list[Fill] = []
    if "trades" in present:
        cols = {r[1] for r in db.execute("pragma table_info(trades)")}
        for r in db.execute("select * from trades order by ts"):
            ts = _to_epoch(r["ts"])
            if ts is None or (since is not None and ts < since):
                continue
            qty = float(r["base_qty"] or 0.0)
            price = float(r["price"] or 0.0)
            if qty <= 0 or price <= 0:
                continue

            ref = None
            if "reason" in cols and r["reason"]:
                m = _PX.search(str(r["reason"]))
                if m:
                    ref = float(m.group(1))

            fills.append(Fill(
                ts=ts,
                symbol=_canonical(str(r["pair"]), alias_symbols),
                side=str(r["side"]).upper(),
                qty=qty,
                price=price,
                fee=abs(float(r["fee"] or 0.0)),
                ref_price=ref,
                reason=str(r["reason"] or "") if "reason" in cols else "",
            ))

    eq_ts: list[float] = []
    eq_val: list[float] = []
    if "equity_snapshots" in present:
        cols = {r[1] for r in db.execute("pragma table_info(equity_snapshots)")}
        value_col = "value_usd" if "value_usd" in cols else next(
            (c for c in cols if c != "ts"), None)
        if value_col:
            for r in db.execute(f"select ts, {value_col} as v from equity_snapshots order by ts"):
                ts = _to_epoch(r["ts"])
                if ts is None or (since is not None and ts < since):
                    continue
                try:
                    v = float(r["v"])
                except (TypeError, ValueError):
                    continue
                if v > 0:
                    eq_ts.append(ts)
                    eq_val.append(v)

    # Vetoes explain the difference between "the signal fired" and "a position opened".
    # Without them, a bot whose risk caps are working looks identical to one that is
    # silently failing to place orders.
    vetoes: list[Veto] = []
    if "decisions" in present:
        cols = {r[1] for r in db.execute("pragma table_info(decisions)")}
        if {"ts", "pair", "action"} <= cols:
            has_reason = "reason" in cols
            q = ("select ts, pair, " + ("reason" if has_reason else "'' as reason") +
                 " from decisions where action = 'VETO' order by ts")
            for r in db.execute(q):
                ts = _to_epoch(r["ts"])
                if ts is None or (since is not None and ts < since):
                    continue
                vetoes.append(Veto(ts=ts,
                                   symbol=_canonical(str(r["pair"]), alias_symbols),
                                   reason=str(r["reason"] or "")))

    db.close()
    return LiveRecord(
        fills=fills,
        equity_ts=np.array(eq_ts),
        equity=np.array(eq_val),
        vetoes=vetoes,
        name=name or p.parent.name,
    )
