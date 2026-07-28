"""Run the prosecution across every strategy and every asset, and record all of it.

    python -m corpus.run                       # the whole study, resumable
    python -m corpus.run --only golden-cross   # one strategy
    python -m corpus.run --cadence daily       # skip the hourly crypto sweep

Design notes, all of which exist because the output is meant to be *quoted*:

**Resumable, keyed on the cell.** A study that has to complete in one process is a study
that never completes. Every cell is committed as it finishes, and re-running skips what
is already there. `--fresh` starts over.

**Deterministic per cell, not per run.** The seed for a cell is derived from the study
seed and the cell's own identity, so a partial run, a resumed run and a run of one cell
in isolation all produce the same number. Seeding once per process would make results
depend on the order cells happened to execute in.

**Provenance next to every verdict.** Library version, git commit, the interpreter and
numeric library versions, the data fingerprint, the grid, and the exact shipped
parameters, all stored per cell. The claim this study wants to make is only worth making
if somebody hostile can check it.

**The shipped variant is what gets scored, and the grid winner is recorded beside it.**
The gap between the two is the study's cleanest single number: how much better the best
of a modest search looks than the version that actually got published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sqlite3
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import falsify_quant
from falsify_quant.harness import sweep
from falsify_quant.spec import CRYPTO_SPOT_TAKER, EQUITY_LIQUID, MarketSpec

from . import cache
from .assets import ALL, Asset
from strategies.canon import CANON, Candidate

DB_PATH = Path(__file__).resolve().parent / "study.db"

# History requested per cadence. Equity is capped by what Yahoo will return; crypto by
# how far back Coinbase listed the pair.
BARS = {"daily-equity": 6000, "daily-crypto": 3000, "hourly-crypto": 40000}

# Crypto is run at three settings, not two. `daily-matched` is the daily series clipped
# to exactly the window the hourly series covers.
#
# Without it the cadence comparison is worthless. Hourly history on a free endpoint runs
# out after a few years, so "daily scored 78 and hourly scored 1" would be comparing
# 2018-2026 against 2022-2026 -- two different markets, one of which contains the 2021
# bull run. The matched cell holds the period fixed and varies only how often the rule is
# allowed to act, which is the thing actually under examination.
CADENCE_SETS = {
    "equity": ("daily",),
    "crypto": ("daily", "hourly", "daily-matched"),
}


# --------------------------------------------------------------------------------------
# Cell definition
# --------------------------------------------------------------------------------------


def base_cadence(cadence: str) -> str:
    """The bar size a cadence actually loads. `daily-matched` is daily, clipped later."""
    return "hourly" if cadence == "hourly" else "daily"


def spec_for(asset: Asset, cadence: str) -> MarketSpec:
    if asset.asset_class == "equity":
        return EQUITY_LIQUID
    if base_cadence(cadence) == "daily":
        return CRYPTO_SPOT_TAKER.at_bars_per_year(365.0)
    return CRYPTO_SPOT_TAKER


def interval_for(asset: Asset, cadence: str) -> str:
    return "1h" if base_cadence(cadence) == "hourly" else "1d"


def bars_for(asset: Asset, cadence: str) -> int:
    return BARS[f"{base_cadence(cadence)}-{asset.asset_class}"]


def cadences_for(asset: Asset) -> tuple[str, ...]:
    """Equity is daily only -- there is no free intraday history worth studying."""
    return CADENCE_SETS[asset.asset_class]


def clip_to_window(bars, t0: float, t1: float):
    """The same series over a shorter span, for the matched-window comparison."""
    if bars.ts is None:
        raise ValueError("cannot match a window without timestamps")
    keep = (bars.ts >= t0) & (bars.ts <= t1)
    if keep.sum() < 50:
        raise ValueError(
            f"only {int(keep.sum())} bars survive the matched window; the hourly history "
            "is too short to compare against"
        )
    fields = {f: getattr(bars, f)[keep] for f in
              ("open", "high", "low", "close", "volume", "ts")
              if getattr(bars, f) is not None}
    return type(bars)(symbol=bars.symbol, **fields)


def cell_seed(study_seed: int, strategy: str, symbol: str, cadence: str) -> int:
    key = f"{study_seed}|{strategy}|{symbol}|{cadence}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


# --------------------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    started_utc      TEXT,
    finished_utc     TEXT,
    falsify_version  TEXT,
    git_sha          TEXT,
    n_permutations   INTEGER,
    study_seed       INTEGER,
    environment      TEXT
);

CREATE TABLE IF NOT EXISTS cells (
    run_id        TEXT, strategy TEXT, symbol TEXT, cadence TEXT,
    family        TEXT, asset_class TEXT, kind TEXT,
    status        TEXT,
    score         REAL, label TEXT, broken INTEGER,
    n_trials      INTEGER, bars INTEGER, years REAL, bars_per_year REAL,
    sharpe_annual REAL, best_sharpe_annual REAL, search_premium REAL,
    fingerprint   TEXT, first_date TEXT, last_date TEXT,
    shipped_json  TEXT, grid_json TEXT, best_params_json TEXT,
    seed          INTEGER, elapsed_s REAL, error TEXT,
    PRIMARY KEY (run_id, strategy, symbol, cadence)
);

CREATE TABLE IF NOT EXISTS findings (
    run_id   TEXT, strategy TEXT, symbol TEXT, cadence TEXT,
    name     TEXT, score REAL, fatal INTEGER, headline TEXT, detail_json TEXT,
    PRIMARY KEY (run_id, strategy, symbol, cadence, name)
);

CREATE INDEX IF NOT EXISTS cells_by_strategy ON cells (run_id, strategy);
CREATE INDEX IF NOT EXISTS findings_by_name  ON findings (run_id, name);
"""


def _clean(obj):
    """Recursively convert findings into something `json` can encode losslessly.

    Not a `JSONEncoder.default` hook, which is the obvious approach and does not work:
    `np.float64` is a *subclass of Python float*, so the encoder serialises it natively
    and `default` is never consulted. A NaN therefore comes out as a bare `NaN` token,
    which is valid for Python's own parser and invalid for every other JSON reader on
    earth -- including the one whoever audits this study will use. `np.float32`, not
    being a float subclass, would take the hook and behave differently again.

    Non-finite values become null. A missing number should read as missing, not as a
    token that makes the whole record unparseable.
    """
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (int, str)) or obj is None:
        return obj
    return str(obj)


def _dumps(obj) -> str:
    return json.dumps(_clean(obj), sort_keys=True, allow_nan=False)


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    # Long studies get split across processes -- equity in one, crypto in another, a
    # single strategy re-run in a third. Each commit is a handful of rows and holds the
    # write lock for milliseconds, so waiting is always the right answer and the default
    # five-second timeout is the wrong one: losing an hour of finished cells to a lock
    # collision is a far worse outcome than a paused commit.
    con.execute("PRAGMA busy_timeout = 60000")
    con.executescript(SCHEMA)
    return con


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=10,
        )
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}" if sha else "unknown"
    except Exception:  # noqa: BLE001 -- provenance is nice to have, not required to run
        return "unknown"


def _environment() -> str:
    import scipy

    return _dumps({
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    })


def done_cells(con: sqlite3.Connection, run_id: str) -> set[tuple[str, str, str]]:
    rows = con.execute(
        "SELECT strategy, symbol, cadence FROM cells WHERE run_id=? AND status='ok'",
        (run_id,),
    ).fetchall()
    return {tuple(r) for r in rows}


# --------------------------------------------------------------------------------------
# One cell
# --------------------------------------------------------------------------------------


def run_cell(
    cand: Candidate,
    asset: Asset,
    cadence: str,
    *,
    n_permutations: int,
    seed: int,
) -> dict:
    """Sweep, prosecute the shipped variant, and return everything worth recording."""
    started = time.time()
    bars = cache.get(
        asset.symbol, asset.asset_class,
        interval=interval_for(asset, cadence), bars=bars_for(asset, cadence),
    )
    if cadence == "daily-matched":
        hourly = cache.get(asset.symbol, asset.asset_class, interval="1h",
                           bars=BARS["hourly-crypto"])
        bars = clip_to_window(bars, float(hourly.ts[0]), float(hourly.ts[-1]))
    spec = spec_for(asset, cadence)

    actual = bars.inferred_bars_per_year
    if actual is not None:
        ratio = actual / spec.bars_per_year
        if not 0.6 < ratio < 1.7:
            raise ValueError(
                f"{asset.symbol} at {cadence}: data is ~{actual:,.0f} bars/year but the "
                f"spec says {spec.bars_per_year:,.0f}. Every annualised figure would be "
                f"off by {max(ratio, 1 / ratio):.1f}x."
            )

    sw = sweep(cand.fn, bars, spec, cand.grid, valid=cand.valid)
    index = sw.index_of(cand.shipped)
    verdict = falsify_quant.run_on_sweep(
        sw, index, n_permutations=n_permutations, seed=seed
    )

    bpy = spec.bars_per_year
    best = sw.best_index
    shipped_sr = falsify_quant.annualise(float(sw.sharpes[index]), bpy)
    best_sr = falsify_quant.annualise(float(sw.sharpes[best]), bpy)
    prov = cache.describe(bars)

    return {
        "verdict": verdict,
        "row": {
            "family": cand.family,
            "asset_class": asset.asset_class,
            "kind": asset.kind,
            "status": "ok",
            "score": float(verdict.score),
            "label": verdict.label,
            "broken": int(verdict.broken),
            "n_trials": int((~sw.failed).sum()),
            "bars": len(bars),
            "years": len(bars) / bpy,
            "bars_per_year": bpy,
            "sharpe_annual": shipped_sr,
            "best_sharpe_annual": best_sr,
            # How much the search flatters the result. Positive means the grid found
            # something that looks better than what the source published -- on the same
            # data, with no new information, which is the definition of the problem.
            "search_premium": best_sr - shipped_sr,
            "fingerprint": prov["fingerprint"],
            "first_date": prov.get("first"),
            "last_date": prov.get("last"),
            "shipped_json": _dumps(cand.shipped),
            "grid_json": _dumps({k: list(v) for k, v in cand.grid.items()}),
            "best_params_json": _dumps(sw.best_params),
            "seed": seed,
            "elapsed_s": time.time() - started,
            "error": None,
        },
    }


def store(con: sqlite3.Connection, run_id: str, cand: Candidate, asset: Asset,
          cadence: str, row: dict, verdict=None) -> None:
    key = (run_id, cand.name, asset.symbol, cadence)
    cols = ["run_id", "strategy", "symbol", "cadence", *row.keys()]
    con.execute(
        f"INSERT OR REPLACE INTO cells ({','.join(cols)}) "
        f"VALUES ({','.join('?' * len(cols))})",
        [*key, *row.values()],
    )
    con.execute(
        "DELETE FROM findings WHERE run_id=? AND strategy=? AND symbol=? AND cadence=?",
        key,
    )
    if verdict is not None:
        con.executemany(
            "INSERT OR REPLACE INTO findings "
            "(run_id, strategy, symbol, cadence, name, score, fatal, headline, detail_json) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            [(*key, f.name, float(f.score), int(f.fatal), f.headline, _dumps(f.detail))
             for f in verdict.findings],
        )
    con.commit()


# --------------------------------------------------------------------------------------
# The study
# --------------------------------------------------------------------------------------


def plan(strategies: list[Candidate], assets: list[Asset],
         cadences: tuple[str, ...]) -> list[tuple[Candidate, Asset, str]]:
    out = []
    for a in assets:
        for cd in cadences_for(a):
            if cd not in cadences:
                continue
            for c in strategies:
                # A candidate declares the *bar sizes* it is meaningful at, so the
                # matched-window cadence is checked as the daily series it actually is.
                # Comparing the cadence label directly drops every matched cell on the
                # floor without a word, which is exactly what it did.
                if base_cadence(cd) not in c.cadences:
                    continue
                out.append((c, a, cd))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the falsify corpus study.")
    ap.add_argument("--out", type=Path, default=DB_PATH)
    ap.add_argument("--run-id", default="main")
    ap.add_argument("--permutations", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--only", action="append", default=None,
                    help="strategy name; repeatable")
    ap.add_argument("--symbol", action="append", default=None,
                    help="asset symbol; repeatable")
    ap.add_argument("--cadence", action="append", default=None,
                    choices=["daily", "hourly", "daily-matched"])
    ap.add_argument("--asset-class", choices=["equity", "crypto"], default=None)
    ap.add_argument("--fresh", action="store_true", help="ignore completed cells")
    ap.add_argument("--refresh-data", action="store_true", help="re-download every series")
    ap.add_argument("--warm-only", action="store_true", help="fill the cache and stop")
    args = ap.parse_args(argv)

    strategies = [c for c in CANON if not args.only or c.name in args.only]
    assets = [a for a in ALL
              if (not args.symbol or a.symbol in args.symbol)
              and (not args.asset_class or a.asset_class == args.asset_class)]
    cadences = tuple(args.cadence) if args.cadence else ("daily", "hourly", "daily-matched")
    if not strategies or not assets:
        print("nothing selected", file=sys.stderr)
        return 2

    # `daily-matched` reads both series, so warm by the bar size actually downloaded.
    print(f"warming the bar cache ({len(assets)} assets)")
    wanted = {base_cadence(c) for c in cadences}
    if "daily-matched" in cadences:
        wanted |= {"daily", "hourly"}
    for cd in sorted(wanted):
        pool = [a for a in assets
                if any(base_cadence(x) == cd for x in cadences_for(a))]
        if not pool:
            continue
        print(f" {cd}:")
        cache.warm(
            pool,
            interval_for=lambda a, cd=cd: interval_for(a, cd),
            bars_for=lambda a, cd=cd: bars_for(a, cd),
            refresh=args.refresh_data,
        )
    if args.warm_only:
        return 0

    con = connect(args.out)
    con.execute(
        "INSERT OR REPLACE INTO runs (run_id, started_utc, falsify_version, git_sha, "
        "n_permutations, study_seed, environment) VALUES (?,?,?,?,?,?,?)",
        (args.run_id, datetime.now(timezone.utc).isoformat(timespec="seconds"),
         falsify_quant.__version__, _git_sha(), args.permutations, args.seed, _environment()),
    )
    con.commit()

    todo = plan(strategies, assets, cadences)
    already = set() if args.fresh else done_cells(con, args.run_id)
    todo = [t for t in todo if (t[0].name, t[1].symbol, t[2]) not in already]

    print(f"\n{len(todo)} cells to run"
          f"{f' ({len(already)} already done)' if already else ''}"
          f", {args.permutations} permutations each\n")

    t0 = time.time()
    failures = 0
    for i, (cand, asset, cadence) in enumerate(todo, 1):
        seed = cell_seed(args.seed, cand.name, asset.symbol, cadence)
        tag = f"{cand.name}/{asset.symbol}/{cadence}"
        try:
            got = run_cell(cand, asset, cadence,
                           n_permutations=args.permutations, seed=seed)
            store(con, args.run_id, cand, asset, cadence, got["row"], got["verdict"])
            r = got["row"]
            print(f"[{i:>4}/{len(todo)}] {tag:<44} {r['score']:5.1f}  "
                  f"{r['label']:<15} SR {r['sharpe_annual']:+5.2f} "
                  f"(best {r['best_sharpe_annual']:+5.2f})  {r['elapsed_s']:5.1f}s")
        except Exception as exc:  # noqa: BLE001 -- one bad cell must not end the study
            failures += 1
            row = {k: None for k in (
                "family", "asset_class", "kind", "status", "score", "label", "broken",
                "n_trials", "bars", "years", "bars_per_year", "sharpe_annual",
                "best_sharpe_annual", "search_premium", "fingerprint", "first_date",
                "last_date", "shipped_json", "grid_json", "best_params_json", "seed",
                "elapsed_s", "error")}
            row.update(family=cand.family, asset_class=asset.asset_class, kind=asset.kind,
                       status="error", seed=seed,
                       error=f"{type(exc).__name__}: {exc}")
            store(con, args.run_id, cand, asset, cadence, row)
            print(f"[{i:>4}/{len(todo)}] {tag:<44} ERROR  {type(exc).__name__}: {exc}")
            if not isinstance(exc, (ValueError, KeyError, RuntimeError)):
                traceback.print_exc(limit=3)

    con.execute("UPDATE runs SET finished_utc=? WHERE run_id=?",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), args.run_id))
    con.commit()

    mins = (time.time() - t0) / 60.0
    print(f"\ndone: {len(todo) - failures} cells in {mins:.1f} min"
          f"{f', {failures} errors' if failures else ''}")
    print(f"results in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
