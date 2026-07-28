"""On-disk bar cache, so the study depends on a snapshot rather than on two web APIs.

Three reasons this is not just `load()` with a memo:

**Reproducibility.** Re-running the study a month later against live endpoints produces
different numbers for every cell, and there is no way afterwards to tell a real change
from a month of extra data. The cache freezes the input.

**Fingerprints.** Each series gets a SHA-256 over its closes and timestamps. It goes into
the results database next to every verdict, so a third party can confirm they are looking
at the same history rather than assuming it. A study whose data cannot be identified is
an anecdote with error bars.

**Rate limits.** Coinbase pages 300 candles at a time; eight thousand hourly bars is
twenty-seven requests, and the study touches each series once per strategy. Without a
cache the run is mostly `time.sleep`.

Everything stored is a plain numeric array in an `.npz`, so loading it executes nothing --
`np.load` refuses object arrays by default and there is no reason to relax that here.

The cache directory is deliberately outside version control: it is tens of megabytes of
redistributable-but-not-mine market data. The fingerprints are what get committed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from falsify.data import load_crypto, load_equity
from falsify.spec import Bars

__all__ = ["CACHE_DIR", "fingerprint", "get", "describe", "warm"]

CACHE_DIR = Path(__file__).resolve().parent / "data"

_FIELDS = ("open", "high", "low", "close", "volume", "ts")


def fingerprint(bars: Bars) -> str:
    """A stable 16-hex-character identity for a price series.

    Over closes and timestamps only. Volume is excluded on purpose: vendors revise it
    after the fact far more often than they revise prices, and a fingerprint that changes
    when nothing you tested on changed is a fingerprint people learn to ignore.
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(bars.close, dtype=np.float64).tobytes())
    if bars.ts is not None:
        h.update(np.ascontiguousarray(bars.ts, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class Entry:
    symbol: str
    asset_class: str
    interval: str
    path: Path

    @property
    def exists(self) -> bool:
        return self.path.exists()


def _entry(symbol: str, asset_class: str, interval: str) -> Entry:
    safe = symbol.replace("/", "-").replace("\\", "-")
    return Entry(symbol, asset_class, interval,
                 CACHE_DIR / f"{asset_class}_{safe}_{interval}.npz")


def _to_bars(z, symbol: str) -> Bars:
    kw = {f: z[f] for f in _FIELDS if f in z.files and z[f].size}
    return Bars(symbol=str(z["symbol"]) if "symbol" in z.files else symbol, **kw)


def get(
    symbol: str,
    asset_class: str,
    *,
    interval: str = "1d",
    bars: int = 6000,
    refresh: bool = False,
) -> Bars:
    """Return a cached series, downloading it once if it is not there yet."""
    e = _entry(symbol, asset_class, interval)
    if e.exists and not refresh:
        with np.load(e.path) as z:
            return _to_bars(z, symbol)

    if asset_class == "crypto":
        got = load_crypto(symbol, interval=interval, bars=bars)
    else:
        got = load_equity(symbol, bars=bars)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {f: np.asarray(getattr(got, f)) for f in _FIELDS if getattr(got, f) is not None}
    payload["symbol"] = np.array(got.symbol)
    np.savez_compressed(e.path, **payload)

    meta = {
        "symbol": symbol,
        "asset_class": asset_class,
        "interval": interval,
        "fingerprint": fingerprint(got),
        "bars": len(got),
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "first_ts": float(got.ts[0]) if got.ts is not None else None,
        "last_ts": float(got.ts[-1]) if got.ts is not None else None,
    }
    e.path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return got


def describe(bars: Bars) -> dict:
    """The provenance block stored alongside every verdict."""
    out: dict = {"fingerprint": fingerprint(bars), "bars": len(bars)}
    if bars.ts is not None:
        out["first"] = datetime.fromtimestamp(float(bars.ts[0]), timezone.utc).date().isoformat()
        out["last"] = datetime.fromtimestamp(float(bars.ts[-1]), timezone.utc).date().isoformat()
    return out


def warm(assets, *, interval_for, bars_for, refresh: bool = False, log=print) -> dict[str, str]:
    """Download everything up front so the study itself never touches the network.

    Failures are reported and skipped rather than fatal. Free endpoints delist things,
    rename things and rate-limit without warning, and losing a nine-hour study run to one
    missing ticker is not a reasonable failure mode.
    """
    prints: dict[str, str] = {}
    for a in assets:
        try:
            b = get(a.symbol, a.asset_class, interval=interval_for(a),
                    bars=bars_for(a), refresh=refresh)
            d = describe(b)
            prints[a.symbol] = d["fingerprint"]
            log(f"  {a.symbol:<10} {d['bars']:>6} bars  "
                f"{d.get('first', '?')} -> {d.get('last', '?')}  {d['fingerprint']}")
        except Exception as exc:  # noqa: BLE001 -- one bad ticker must not end the study
            log(f"  {a.symbol:<10} SKIPPED: {type(exc).__name__}: {exc}")
    return prints
