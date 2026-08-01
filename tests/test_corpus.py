"""The study machinery, tested for the properties the study's credibility rests on.

None of this is clever code. It is tested anyway, because the corpus study is meant to be
*quoted*, and the ways it could quietly be wrong are all mundane: a seed that depends on
execution order, a fingerprint that does not change when the data does, a matched window
that is not matched, a percentile that is off by one element.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from corpus import cache
from corpus.aggregate import _q, build, load, write_csv
from corpus.assets import ALL, CRYPTO, EQUITIES, by_class
from corpus.run import (
    SCHEMA,
    _dumps,
    base_cadence,
    bars_for,
    cadences_for,
    cell_seed,
    clip_to_window,
    connect,
    done_cells,
    interval_for,
    plan,
    spec_for,
)
from falsify_quant.spec import Bars
from strategies.canon import CANON, by_name


def _bars(n=500, seed=0):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, n))
    ts = 1_600_000_000.0 + np.arange(n) * 86400.0
    return Bars(close=close, high=close * 1.01, low=close * 0.99,
                volume=np.ones(n), ts=ts, symbol="X")


# --------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------


def test_cell_seed_depends_only_on_the_cell():
    """A resumed run, a partial run and a single cell run alone must all agree.

    Seeding once per process instead would make every result depend on the order cells
    happened to execute in, which is the one thing a reproducibility claim cannot survive.
    """
    a = cell_seed(1, "golden-cross", "SPY", "daily")
    assert a == cell_seed(1, "golden-cross", "SPY", "daily")
    assert a != cell_seed(2, "golden-cross", "SPY", "daily")
    assert a != cell_seed(1, "dual-ma", "SPY", "daily")
    assert a != cell_seed(1, "golden-cross", "QQQ", "daily")
    assert a != cell_seed(1, "golden-cross", "SPY", "hourly")


def test_cell_seeds_do_not_collide_across_the_real_study():
    seeds = {cell_seed(20260727, c.name, a.symbol, cd)
             for c in CANON for a in ALL for cd in cadences_for(a)}
    total = sum(len(cadences_for(a)) for a in ALL) * len(CANON)
    assert len(seeds) == total, "two cells share a seed"


def test_cell_seed_is_a_valid_numpy_seed():
    s = cell_seed(20260727, "golden-cross", "BTC-USD", "hourly")
    assert 0 <= s < 2**32
    np.random.default_rng(s)


# --------------------------------------------------------------------------------------
# Fingerprints
# --------------------------------------------------------------------------------------


def test_fingerprint_is_stable_and_sensitive():
    b = _bars()
    assert cache.fingerprint(b) == cache.fingerprint(_bars())

    moved = _bars()
    moved.close[123] += 1e-6
    assert cache.fingerprint(moved) != cache.fingerprint(b)

    shifted = Bars(close=b.close, ts=b.ts + 1.0, symbol=b.symbol)
    assert cache.fingerprint(shifted) != cache.fingerprint(b)


def test_fingerprint_ignores_volume():
    """Vendors revise volume constantly. A print that churns is a print people ignore."""
    b = _bars()
    other = Bars(close=b.close, high=b.high, low=b.low, ts=b.ts,
                 volume=b.volume * 3.0, symbol=b.symbol)
    assert cache.fingerprint(other) == cache.fingerprint(b)


def test_describe_reports_the_window():
    d = cache.describe(_bars(100))
    assert d["bars"] == 100
    assert d["first"] < d["last"]
    assert len(d["fingerprint"]) == 16


def _entry_with_age(tmp_path, hours: float | None):
    """A cache entry whose sidecar claims it was fetched *hours* ago (None = no sidecar)."""
    e = cache.Entry("X", "crypto", "1h", tmp_path / "crypto_X_1h.npz")
    np.savez(e.path, close=np.ones(10), ts=np.arange(10, dtype=float), symbol=np.array("X"))
    if hours is not None:
        when = datetime.now(timezone.utc) - timedelta(hours=hours)
        e.path.with_suffix(".json").write_text(
            json.dumps({"fetched_utc": when.isoformat(timespec="seconds")}), encoding="utf-8")
    return e


@pytest.mark.parametrize("hours,interval,stale", [
    (0.1, "1h", False),   # fetched minutes ago, hourly bars -> still current
    (2.0, "1h", True),    # a bar has closed since -> a refetch would add data
    (2.0, "1d", False),   # two hours is nothing to a daily series
    (30.0, "1d", True),   # more than a day -> stale
])
def test_cache_entry_expires_after_one_bar_interval(tmp_path, hours, interval, stale):
    e = _entry_with_age(tmp_path, hours)
    assert cache._is_stale(e, interval) is stale


def test_cache_entry_without_provenance_is_treated_as_stale(tmp_path):
    """Entries written before the sidecar existed must heal themselves rather than be
    served forever. This module recorded `fetched_utc` and then never read it back, so
    a cached series was returned no matter how old it was -- which is how two machines
    came to hold different data under the same filename and score the same strategy
    30 points apart (2026-08-01). Provenance you do not act on is not provenance."""
    assert cache._is_stale(_entry_with_age(tmp_path, None), "1d") is True


def test_cache_entry_with_unparseable_provenance_is_treated_as_stale(tmp_path):
    e = _entry_with_age(tmp_path, 0.1)
    e.path.with_suffix(".json").write_text("{not json", encoding="utf-8")
    assert cache._is_stale(e, "1h") is True


# --------------------------------------------------------------------------------------
# The matched window
# --------------------------------------------------------------------------------------


def test_clip_to_window_keeps_alignment():
    b = _bars(500)
    t0, t1 = float(b.ts[100]), float(b.ts[399])
    c = clip_to_window(b, t0, t1)

    assert len(c) == 300
    np.testing.assert_array_equal(c.close, b.close[100:400])
    np.testing.assert_array_equal(c.ts, b.ts[100:400])
    np.testing.assert_array_equal(c.high, b.high[100:400])
    assert c.symbol == b.symbol


def test_clip_to_window_refuses_a_useless_window():
    b = _bars(500)
    with pytest.raises(ValueError, match="too short"):
        clip_to_window(b, float(b.ts[0]), float(b.ts[10]))


def test_matched_cadence_loads_daily_bars():
    """`daily-matched` is a daily series clipped, not an hourly one relabelled."""
    assert base_cadence("daily-matched") == "daily"
    btc = CRYPTO[0]
    assert interval_for(btc, "daily-matched") == "1d"
    assert bars_for(btc, "daily-matched") == bars_for(btc, "daily")
    assert spec_for(btc, "daily-matched").bars_per_year == pytest.approx(365.0)
    assert spec_for(btc, "hourly").bars_per_year == pytest.approx(365 * 24)


def test_crypto_costs_do_not_change_with_the_bar_size():
    """The comparison is only fair if a turnover costs the same at both cadences."""
    btc = CRYPTO[0]
    daily, hourly = spec_for(btc, "daily-matched"), spec_for(btc, "hourly")
    assert daily.cost_per_turnover == pytest.approx(hourly.cost_per_turnover)


# --------------------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------------------


def test_seasonal_rules_are_not_run_on_hourly_bars():
    cells = plan(CANON, by_class("crypto"), ("daily", "hourly", "daily-matched"))
    hourly = {c.name for c, _, cd in cells if cd == "hourly"}
    assert "turn-of-month" not in hourly
    assert "turn-of-month" in {c.name for c, _, cd in cells if cd == "daily"}


def test_the_matched_window_is_actually_planned():
    """The cadence comparison is the study's sharpest claim and it is opt-in by plan().

    A candidate declares the bar *sizes* it is meaningful at, so `daily-matched` has to be
    matched against `daily`. Comparing the label directly drops every matched cell without
    an error, and the study still finishes -- just with nothing to compare.
    """
    cells = plan(CANON, by_class("crypto"), ("daily", "hourly", "daily-matched"))
    matched = {c.name for c, _, cd in cells if cd == "daily-matched"}
    daily = {c.name for c, _, cd in cells if cd == "daily"}
    assert matched == daily, "the matched window does not cover the same rules as daily"
    assert len(matched) == len(CANON)


def test_matched_and_hourly_pair_up_for_every_symbol():
    cells = plan(CANON, by_class("crypto"), ("hourly", "daily-matched"))
    pairs = {(c.name, a.symbol) for c, a, cd in cells if cd == "hourly"}
    matched = {(c.name, a.symbol) for c, a, cd in cells if cd == "daily-matched"}
    assert pairs <= matched, "an hourly cell has no matched-window counterpart"


def test_equities_are_daily_only():
    cells = plan(CANON, EQUITIES, ("daily", "hourly", "daily-matched"))
    assert {cd for _, _, cd in cells} == {"daily"}


def test_plan_respects_the_cadence_filter():
    """Derived from ALL rather than from a list of classes.

    Written the other way it counted equities and crypto by hand, so adding futures broke
    a test about cadence filtering for reasons that had nothing to do with cadence.
    """
    cells = plan(CANON, ALL, ("daily",))
    assert {cd for _, _, cd in cells} == {"daily"}
    eligible = [a for a in ALL if "daily" in cadences_for(a)]
    assert len(cells) == len(eligible) * len(CANON)
    assert len(eligible) == len(ALL), "every class in this universe runs daily"


def test_every_asset_has_a_spec_and_an_interval():
    for a in ALL:
        for cd in cadences_for(a):
            assert spec_for(a, cd).bars_per_year > 0
            assert interval_for(a, cd) in ("1d", "1h")
            assert bars_for(a, cd) >= 1000


# --------------------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------------------


def test_numpy_json_survives_what_findings_actually_contain():
    payload = {
        "n": np.int64(7),
        "x": np.float32(1.5),
        "flag": np.bool_(True),
        "series": np.arange(3),
        "nan": np.float64("nan"),
        "inf": np.float64("inf"),
    }
    got = json.loads(_dumps(payload))
    assert got == {"n": 7, "x": 1.5, "flag": True, "series": [0, 1, 2],
                   "nan": None, "inf": None}


def test_non_finite_floats_become_null_not_invalid_json():
    """`json` writes bare NaN by default, which nothing else can parse back."""
    text = _dumps({"v": np.float64("nan")})
    assert "NaN" not in text
    json.loads(text)


def test_done_cells_only_counts_successes(tmp_path):
    con = connect(tmp_path / "s.db")
    cols = "run_id,strategy,symbol,cadence,status"
    con.execute(f"INSERT INTO cells ({cols}) VALUES ('r','a','SPY','daily','ok')")
    con.execute(f"INSERT INTO cells ({cols}) VALUES ('r','b','SPY','daily','error')")
    con.execute(f"INSERT INTO cells ({cols}) VALUES ('other','c','SPY','daily','ok')")
    con.commit()

    assert done_cells(con, "r") == {("a", "SPY", "daily")}


def test_schema_is_idempotent(tmp_path):
    """Resuming reopens the database. Running the schema again must be a no-op."""
    p = tmp_path / "s.db"
    connect(p).close()
    con = connect(p)
    con.executescript(SCHEMA)
    assert con.execute("SELECT count(*) FROM cells").fetchone()[0] == 0


def test_a_cell_is_unique_on_strategy_symbol_cadence(tmp_path):
    """Re-running a cell must replace its row, not accumulate rows."""
    con = connect(tmp_path / "s.db")
    for score in (10.0, 90.0):
        con.execute(
            "INSERT OR REPLACE INTO cells (run_id,strategy,symbol,cadence,status,score) "
            "VALUES ('r','a','SPY','daily','ok',?)", (score,))
    con.commit()
    rows = con.execute("SELECT score FROM cells").fetchall()
    assert rows == [(90.0,)]


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------


def _fake_study(path, n_per=6):
    """A study database with a known shape, for testing the report generator."""
    con = connect(path)
    con.execute(
        "INSERT INTO runs (run_id, started_utc, falsify_version, git_sha, "
        "n_permutations, study_seed, environment) VALUES "
        "('main','2026-07-27T00:00:00','0.1.0','abc1234',100,42,?)",
        (_dumps({"python": "3.12.0", "numpy": "2.0.0", "scipy": "1.14.0"}),))

    labels = ["SURVIVED", "PLAUSIBLE", "UNPROVEN", "LIKELY OVERFIT", "NO EDGE FOUND",
              "NO EDGE FOUND"]
    rows, finds = [], []
    # Six strategies, not four: the cadence section needs at least five matched pairs
    # before it will say anything, and one crypto symbol contributes one pair each.
    for c in CANON[:6]:
        for i, sym in enumerate(["SPY", "QQQ", "BTC-USD"]):
            ac = "crypto" if sym.endswith("-USD") else "equity"
            for cd in (("daily", "hourly", "daily-matched") if ac == "crypto"
                       else ("daily",)):
                rows.append({
                    "run_id": "main", "strategy": c.name, "symbol": sym, "cadence": cd,
                    "family": c.family, "asset_class": ac, "kind": "index",
                    "status": "ok",
                    "score": float(90 - 15 * (len(rows) % n_per)),
                    "label": labels[len(rows) % n_per], "broken": 0,
                    "n_trials": 24, "bars": 5000, "years": 19.8, "bars_per_year": 252.0,
                    "sharpe_annual": 0.5 + 0.1 * i,
                    "best_sharpe_annual": 1.1 + 0.1 * i,
                    "search_premium": 0.6,
                    "fingerprint": f"fp{i}", "first_date": "2002-01-01",
                    "last_date": "2026-07-27",
                    "shipped_json": _dumps(c.shipped),
                    "grid_json": _dumps({k: list(v) for k, v in c.grid.items()}),
                    "best_params_json": _dumps(c.shipped),
                    "seed": 1234, "elapsed_s": 1.5, "error": None,
                })
                for t, s in [("causality", 1.0), ("costs", 0.3), ("deflation", 0.8),
                             ("pbo", 0.4), ("permutation", 0.05), ("regime", 0.7)]:
                    finds.append(("main", c.name, sym, cd, t, s, 0, f"{t} headline", "{}"))

    # Insert by name. Positional inserts silently rot the moment a column is added.
    cols = list(rows[0])
    con.executemany(
        f"INSERT INTO cells ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
        [[r[k] for k in cols] for r in rows],
    )
    con.executemany("INSERT INTO findings VALUES (?,?,?,?,?,?,?,?,?)", finds)
    con.commit()
    con.close()
    _fake_study.columns = cols
    return len(rows)


def test_fake_study_covers_every_cells_column(tmp_path):
    """The fixture must stay in step with the schema, or these tests test nothing.

    Add a column to `cells` and forget the fixture, and every aggregation test below
    keeps passing while silently exercising a `None`.
    """
    import re

    from corpus.run import SCHEMA as S

    _fake_study(tmp_path / "study.db")
    block = re.search(r"CREATE TABLE IF NOT EXISTS cells \((.*?)PRIMARY KEY", S, re.S)
    # Split on commas, not lines -- the schema declares several columns per line.
    declared = {f.strip().split()[0] for f in block.group(1).split(",") if f.strip()}
    assert len(declared) > 20, "the schema parse found suspiciously few columns"
    assert declared == set(_fake_study.columns), "the fixture and the schema have drifted"


def test_report_builds_and_contains_the_load_bearing_sections(tmp_path):
    n = _fake_study(tmp_path / "study.db")
    run, cells, findings = load(tmp_path / "study.db", "main")
    assert len(cells) == n

    md = build(run, cells, findings)
    for heading in ["What the study found", "Verdicts", "Which test does the killing",
                    "By family", "The short leg", "What the search is worth",
                    "Choosing the window", "Trading the same rule faster", "By strategy",
                    "What this does not establish", "Reproducing it"]:
        assert heading in md, f"missing section: {heading}"

    assert "abc1234" in md and "0.1.0" in md  # provenance survives into the report
    assert "NaN" not in md


def test_report_names_no_vendors(tmp_path):
    """The aggregation publishes strategies and citations, never a commercial verdict."""
    _fake_study(tmp_path / "study.db")
    run, cells, findings = load(tmp_path / "study.db", "main")
    md = build(run, cells, findings).lower()
    for word in ["signal service", "subscribers", "refund", "scam", "fraud"]:
        assert word not in md


def test_cadence_section_needs_matched_pairs(tmp_path):
    """Without the matched window there is nothing honest to say, so it says nothing."""
    from corpus.aggregate import cadence_effect

    _fake_study(tmp_path / "study.db")
    _, cells, _ = load(tmp_path / "study.db", "main")
    assert cadence_effect(cells), "matched pairs present but the section is empty"

    unmatched = [c for c in cells if c["cadence"] != "daily-matched"]
    assert cadence_effect(unmatched) == []


def test_csv_round_trips(tmp_path):
    _fake_study(tmp_path / "study.db")
    _, cells, _ = load(tmp_path / "study.db", "main")
    out = tmp_path / "results.csv"
    write_csv(cells, out)

    text = out.read_text(encoding="utf-8").splitlines()
    assert len(text) == len(cells) + 1
    assert text[0].startswith("strategy,family,symbol")
    assert "fingerprint" in text[0]


def test_load_rejects_a_missing_run(tmp_path):
    _fake_study(tmp_path / "study.db")
    with pytest.raises(SystemExit, match="no run"):
        load(tmp_path / "study.db", "nope")


@pytest.mark.parametrize("p,want", [(0.0, 1.0), (0.5, 3.0), (1.0, 5.0), (0.25, 2.0)])
def test_percentile_interpolates(p, want):
    assert _q([5.0, 1.0, 3.0, 2.0, 4.0], p) == pytest.approx(want)


def test_percentile_of_empty_is_nan():
    assert _q([], 0.5) != _q([], 0.5)


# --------------------------------------------------------------------------------------
# The universe
# --------------------------------------------------------------------------------------


def test_universe_has_no_duplicates_and_covers_both_classes():
    syms = [a.symbol for a in ALL]
    assert len(syms) == len(set(syms))
    assert len(by_class("equity")) >= 15 and len(by_class("crypto")) >= 8


def test_universe_includes_instruments_that_did_badly():
    """A list of today's winners answers a different question than the one being asked."""
    assert {"GE", "F", "INTC", "USO"} <= {a.symbol for a in EQUITIES}


def test_universe_is_not_only_single_names():
    kinds = {a.kind for a in EQUITIES}
    assert {"index", "bond", "commodity"} <= kinds
