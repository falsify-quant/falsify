"""Key-free OHLCV loaders, so you can point this at something real in one line.

Two public sources, no accounts, no API keys, stdlib only:

    crypto    Coinbase Exchange public candles   (BTC-USD, ETH-USD, SOL-USD, ...)
    equity    Yahoo Finance chart API            (AAPL, SPY, NVDA, ...)

Neither is a substitute for your own execution data. If you have real fills from your
own bot, use those -- `Bars` takes any numpy array, and your fills know things about
slippage that no free daily bar ever will.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np

from .spec import Bars

__all__ = ["load_crypto", "load_equity", "load", "GRANULARITY"]

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) falsify/0.1"}

# Coinbase granularity values, in seconds. Those are the only ones the API accepts.
GRANULARITY = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "6h": 21600,
    "1d": 86400,
}


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def load_crypto(
    symbol: str = "BTC-USD",
    *,
    interval: str = "1h",
    bars: int = 8000,
    pause: float = 0.22,
) -> Bars:
    """Fetch candles from Coinbase Exchange's public API.

    Coinbase caps each request at 300 candles, so this pages backwards from now. At the
    default hourly interval, 8,000 bars is a bit under a year and takes ~27 requests.
    """
    if interval not in GRANULARITY:
        raise ValueError(f"interval must be one of {sorted(GRANULARITY)}, got {interval!r}")
    gran = GRANULARITY[interval]
    step = 300 * gran

    end = datetime.now(timezone.utc).replace(microsecond=0)
    rows: dict[int, tuple[float, float, float, float, float]] = {}

    while len(rows) < bars:
        start = end - timedelta(seconds=step)
        url = (
            f"https://api.exchange.coinbase.com/products/{symbol}/candles"
            f"?granularity={gran}&start={start.isoformat()}&end={end.isoformat()}"
        )
        try:
            import json

            batch = json.loads(_get(url))
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"Coinbase rejected the request for {symbol} ({e.code}). "
                "Check the product id — it looks like 'BTC-USD', not 'BTCUSDT'."
            ) from e

        if not batch:
            break  # ran off the start of the product's history

        for t, lo, hi, op, cl, vol in batch:
            rows[int(t)] = (float(op), float(hi), float(lo), float(cl), float(vol))

        end = start
        time.sleep(pause)  # Coinbase public rate limit is ~10 req/s; stay well under

    if len(rows) < 50:
        raise RuntimeError(f"only got {len(rows)} candles for {symbol}; nothing to test")

    ts = np.array(sorted(rows)[-bars:], dtype=np.float64)
    ohlcv = np.array([rows[int(t)] for t in ts], dtype=np.float64)

    return Bars(
        open=ohlcv[:, 0], high=ohlcv[:, 1], low=ohlcv[:, 2],
        close=ohlcv[:, 3], volume=ohlcv[:, 4],
        ts=ts, symbol=f"{symbol}@{interval}",
    )


def load_equity(symbol: str = "SPY", *, bars: int = 5000) -> Bars:
    """Fetch daily bars from Yahoo Finance.

    Uses the *adjusted* close, and rescales OHL by the same adjustment factor so the bar
    stays internally consistent. This matters more than it sounds: backtesting a
    dividend payer on unadjusted closes books a fake loss on every ex-dividend date, and
    on something like SPY over twenty years that is most of the return.

    Requests an explicit date window rather than `range=max`. Yahoo answers `range=max`
    with silently *downsampled* bars -- 403 rows spanning 33 years of SPY, i.e. monthly
    data wearing a daily label. Backtesting a daily strategy on that produces numbers
    that are wrong in every direction at once and look completely normal.
    """
    import json

    period2 = int(time.time())
    period1 = period2 - int((bars / 252.0 + 2) * 365.25 * 86400)  # trading days -> calendar
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={period1}&period2={period2}&interval=1d"
    )
    try:
        payload = json.loads(_get(url))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Yahoo rejected {symbol!r} ({e.code}). Check the ticker.") from e

    result = (payload.get("chart") or {}).get("result")
    if not result:
        err = ((payload.get("chart") or {}).get("error") or {}).get("description", "no data")
        raise RuntimeError(f"Yahoo returned nothing for {symbol!r}: {err}")

    res = result[0]
    quote = res["indicators"]["quote"][0]
    adj = (res["indicators"].get("adjclose") or [{}])[0].get("adjclose")
    ts_all = res["timestamp"]
    close_all = quote["close"]
    if adj is None:
        adj = close_all

    ts, o, h, low, c, v = [], [], [], [], [], []
    for i in range(len(ts_all)):
        px, ac = close_all[i], adj[i]
        if px is None or ac is None or px <= 0:
            continue  # holidays and halted sessions come back as nulls
        k = ac / px  # adjustment factor for this bar
        ts.append(float(ts_all[i]))
        c.append(float(ac))
        o.append(float((quote["open"][i] or px) * k))
        h.append(float((quote["high"][i] or px) * k))
        low.append(float((quote["low"][i] or px) * k))
        v.append(float(quote["volume"][i] or 0.0))

    if len(c) < 50:
        raise RuntimeError(f"only {len(c)} usable rows for {symbol!r}; nothing to test")

    sl = slice(-bars, None)
    return Bars(
        open=np.array(o[sl]), high=np.array(h[sl]), low=np.array(low[sl]),
        close=np.array(c[sl]), volume=np.array(v[sl]),
        ts=np.array(ts[sl]), symbol=symbol.upper(),
    )


def load(symbol: str, *, asset_class: str, interval: str = "1h", bars: int = 5000) -> Bars:
    """Dispatch to the right loader for the asset class."""
    if asset_class == "crypto":
        return load_crypto(symbol, interval=interval, bars=bars)
    return load_equity(symbol, bars=bars)
