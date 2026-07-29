"""Turn a study database into aggregate findings.

    python -m corpus.aggregate                      # writes FINDINGS.md and results.csv
    python -m corpus.aggregate --db other.db

**Aggregates only.** Nothing here names a commercial product, a vendor, a course or a
signal service and attaches a verdict to it. That restraint is partly legal -- publishing
"we tested X's system and it does not work" invites a letter regardless of the arithmetic
-- and partly because the aggregate claim is the more interesting one. "Of the eighteen
best-known published rules, the median scored N and only M cleared their own trading
costs" says something about the field. A league table just starts an argument about one
entry.

The per-cell CSV *is* published, because the strategies are public knowledge with public
citations and the whole point is that the numbers can be checked. The line is between
prosecuting the canon, which is scholarship, and prosecuting a named seller, which is not
this study's job.

Percentiles are reported alongside medians throughout. A study that reports only a median
is asking to be quoted as if every strategy scored it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "study.db"

BAND_ORDER = ["SURVIVED", "PLAUSIBLE", "UNPROVEN", "LIKELY OVERFIT", "NO EDGE FOUND",
              "BROKEN"]

TEST_TITLES = {
    "causality": "Causality (lookahead)",
    "costs": "Cost breakeven",
    "deflation": "Deflated Sharpe",
    "pbo": "Backtest overfitting (PBO)",
    "permutation": "Monte-Carlo permutation",
    "regime": "Regime concentration",
}


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def load(db: Path, run_id: str) -> tuple[dict, list[dict], dict]:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row

    run = con.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if run is None:
        raise SystemExit(f"no run {run_id!r} in {db}. Have: "
                         f"{[r[0] for r in con.execute('SELECT run_id FROM runs')]}")

    cells = [dict(r) for r in con.execute(
        "SELECT * FROM cells WHERE run_id=? AND status='ok'", (run_id,))]
    if not cells:
        raise SystemExit(f"run {run_id!r} has no completed cells")

    findings: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in con.execute("SELECT * FROM findings WHERE run_id=?", (run_id,)):
        findings[(r["strategy"], r["symbol"], r["cadence"])][r["name"]] = dict(r)

    con.close()
    return dict(run), cells, findings


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.0f}%" if whole else "-"


def _q(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _fmt(x: float, nd: int = 1) -> str:
    return "-" if x != x else f"{x:.{nd}f}"


# --------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------


def _home_turf_cells(cells: list[dict]) -> tuple[list[dict], int]:
    """Daily cells scored where their source tested, and how many rules have no such venue.

    Shared so the headline and the `Home turf` section cannot drift apart -- the whole
    point of that section is that the headline number should not be read without it.
    """
    try:
        from strategies.canon import CANON
    except Exception:  # pragma: no cover
        return [], 0
    domains = {c.name: c.domain for c in CANON}
    matched = [c for c in cells
               if c["cadence"] == "daily"
               and domains.get(c["strategy"]) == "equity-index"
               and _venue(c) == "equity index ETF"]
    homeless = sum(1 for c in CANON if c.domain in ("futures", "cross-asset"))
    return matched, homeless


def headline(cells: list[dict], findings: dict) -> list[str]:
    scores = [c["score"] for c in cells]
    labels = defaultdict(int)
    for c in cells:
        labels[c["label"]] += 1

    n = len(cells)
    survived = labels["SURVIVED"] + labels["PLAUSIBLE"]
    broken = sum(1 for c in cells if c["broken"])

    failed_costs = sum(1 for k, f in findings.items()
                       if "costs" in f and f["costs"]["score"] < 0.5)
    scored_costs = sum(1 for f in findings.values() if "costs" in f)

    out = [
        "## What the study found",
        "",
        f"**{n:,} verdicts.** Each is one published rule, at the parameters its source "
        f"named, on one instrument, over as much history as a free data feed will give.",
        "",
        f"- **Median score {_fmt(statistics.median(scores))} / 100.** "
        f"Quartiles {_fmt(_q(scores, 0.25))} and {_fmt(_q(scores, 0.75))}; "
        f"the best cell scored {_fmt(max(scores))} and the worst {_fmt(min(scores))}.",
        f"- **{_pct(survived, n)} reached PLAUSIBLE or better** "
        f"({survived:,} of {n:,}). {_pct(labels['NO EDGE FOUND'], n)} came back "
        f"NO EDGE FOUND -- indistinguishable from having tested noise.",
        f"- **{_pct(failed_costs, scored_costs)} could not clear their own trading costs** "
        f"at retail rates, before any question of overfitting arises.",
    ]
    if broken:
        out.append(f"- **{broken} cell(s) failed the causality gate.** Every strategy here "
                   f"was written from a published description with the lookahead test "
                   f"already in the loop, so this number says more about how hard the bug "
                   f"is to avoid than about the sources.")
    else:
        out.append("- **No cell failed the causality gate.** Expected, and worth stating "
                   "plainly: these are clean-room implementations written against a "
                   "truncation test. The lookahead rate in *published implementations* is "
                   "a different study, and this one does not measure it.")
    matched, _ = _home_turf_cells(cells)
    if matched:
        out += [
            "",
            f"**Read that median with `Home turf` below, not on its own.** It pools every "
            f"rule over every instrument, including rules run in markets their sources "
            f"never claimed. Scored where their own authors tested them, the equity-index "
            f"rules median "
            f"**{_fmt(statistics.median([c['score'] for c in matched]))}**. The futures "
            f"systems, given futures, do not recover the same way -- but the contracts "
            f"they were actually developed on are the ones whose free data could not be "
            f"trusted, so that comparison is narrower than it looks. Both are in "
            f"`Home turf`. The pooled figure is the right answer to \"what happens if you "
            f"take the canon and point it at whatever you can download\", which is what "
            f"most people do -- and the wrong answer to \"does this rule work\".",
        ]
    out.append("")
    return out


def by_label(cells: list[dict]) -> list[str]:
    counts = defaultdict(int)
    for c in cells:
        counts[c["label"]] += 1
    n = len(cells)
    rows = ["## Verdicts", "", "| Verdict | Cells | Share |", "|---|---:|---:|"]
    for lbl in BAND_ORDER:
        if counts[lbl]:
            rows.append(f"| {lbl} | {counts[lbl]:,} | {_pct(counts[lbl], n)} |")
    rows.append("")
    return rows


def by_family(cells: list[dict]) -> list[str]:
    groups = defaultdict(list)
    for c in cells:
        groups[c["family"]].append(c)

    rows = ["## By family", "",
            "| Family | Cells | Median | 25th | 75th | Best | PLAUSIBLE+ |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for fam, g in sorted(groups.items(), key=lambda kv: -statistics.median(
            [c["score"] for c in kv[1]])):
        s = [c["score"] for c in g]
        good = sum(1 for c in g if c["label"] in ("SURVIVED", "PLAUSIBLE"))
        rows.append(f"| {fam} | {len(g):,} | {_fmt(statistics.median(s))} | "
                    f"{_fmt(_q(s, 0.25))} | {_fmt(_q(s, 0.75))} | {_fmt(max(s))} | "
                    f"{_pct(good, len(g))} |")
    rows.append("")
    return rows


def by_strategy(cells: list[dict]) -> list[str]:
    groups = defaultdict(list)
    for c in cells:
        groups[c["strategy"]].append(c)

    rows = ["## By strategy", "",
            "Every rule at the parameters its own source published, across every "
            "instrument and cadence it was run on.",
            "",
            "| Strategy | Family | Cells | Median | Best cell | PLAUSIBLE+ | Median SR |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for name, g in sorted(groups.items(), key=lambda kv: -statistics.median(
            [c["score"] for c in kv[1]])):
        s = [c["score"] for c in g]
        sr = [c["sharpe_annual"] for c in g if c["sharpe_annual"] is not None]
        good = sum(1 for c in g if c["label"] in ("SURVIVED", "PLAUSIBLE"))
        rows.append(f"| `{name}` | {g[0]['family']} | {len(g):,} | "
                    f"{_fmt(statistics.median(s))} | {_fmt(max(s))} | "
                    f"{_pct(good, len(g))} | "
                    f"{_fmt(statistics.median(sr), 2) if sr else '-'} |")
    rows.append("")
    return rows


def by_test(cells: list[dict], findings: dict) -> list[str]:
    per_test: dict[str, list[float]] = defaultdict(list)
    for f in findings.values():
        for name, row in f.items():
            per_test[name].append(row["score"])

    rows = ["## Which test does the killing", "",
            "Each check scores 0 to 1. The verdict is their weighted geometric mean, so "
            "the column that matters is how often a check lands near zero -- one fatal "
            "leg drags the whole score down regardless of the others.",
            "",
            "| Check | Median | Failed (< 0.5) | Near-fatal (< 0.1) |",
            "|---|---:|---:|---:|"]
    for name in ["causality", "costs", "deflation", "pbo", "permutation", "regime"]:
        s = per_test.get(name)
        if not s:
            continue
        bad = sum(1 for x in s if x < 0.5)
        awful = sum(1 for x in s if x < 0.1)
        rows.append(f"| {TEST_TITLES.get(name, name)} | {_fmt(statistics.median(s), 2)} | "
                    f"{_pct(bad, len(s))} | {_pct(awful, len(s))} |")
    rows.append("")
    return rows


def search_premium(cells: list[dict]) -> list[str]:
    gaps = [c["search_premium"] for c in cells if c["search_premium"] is not None]
    if not gaps:
        return []
    big = sum(1 for g in gaps if g > 0.5)
    return [
        "## What the search is worth",
        "",
        "Every cell sweeps a modest grid around the published parameters -- a few dozen "
        "combinations, far fewer than the literature has tried. `search premium` is the "
        "annualised Sharpe of the best combination minus the Sharpe of the one the source "
        "actually published, on identical data. It is the size of the free lunch available "
        "to anyone willing to report their best run.",
        "",
        f"- Median premium **{_fmt(statistics.median(gaps), 2)} Sharpe**; "
        f"75th percentile {_fmt(_q(gaps, 0.75), 2)}; largest {_fmt(max(gaps), 2)}.",
        f"- **{_pct(big, len(gaps))} of cells** had a best-in-grid at least 0.5 Sharpe "
        f"above the published version.",
        "",
        "That is the entire gap between a strategy that looks publishable and one that "
        "does not, and it is available on pure noise. It is also a *lower* bound: the "
        "grids here are small, and nobody stops at one grid.",
        "",
    ]


def cadence_effect(cells: list[dict]) -> list[str]:
    """Same rule, same asset, same window, different bar size."""
    by_key: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for c in cells:
        if c["asset_class"] != "crypto":
            continue
        by_key[(c["strategy"], c["symbol"])][c["cadence"]] = c

    pairs = [(v["daily-matched"], v["hourly"]) for v in by_key.values()
             if "daily-matched" in v and "hourly" in v]
    if len(pairs) < 5:
        return []

    deltas = [h["score"] - d["score"] for d, h in pairs]
    d_scores = [d["score"] for d, _ in pairs]
    h_scores = [h["score"] for _, h in pairs]
    worse = sum(1 for x in deltas if x < 0)
    d_turn = [d["sharpe_annual"] for d, _ in pairs]
    h_turn = [h["sharpe_annual"] for _, h in pairs]

    return [
        "## Trading the same rule faster",
        "",
        "Crypto is run twice on **the same calendar window**: once on daily bars and once "
        "on hourly. Same rule, same parameters, same instrument, same dates, same cost per "
        "unit of turnover. The only difference is how often the rule is allowed to act.",
        "",
        "The matched window is not a nicety. Hourly history from a free endpoint runs out "
        "after a few years, so comparing it against a full daily history compares two "
        "different markets and blames the bar size. See the section above for how much "
        "damage that does -- it was enough to reverse this study's first conclusion.",
        "",
        f"- Median score **{_fmt(statistics.median(d_scores))} daily** vs "
        f"**{_fmt(statistics.median(h_scores))} hourly** across {len(pairs)} matched pairs.",
        f"- Median annualised Sharpe **{_fmt(statistics.median(d_turn), 2)} daily** vs "
        f"**{_fmt(statistics.median(h_turn), 2)} hourly**.",
        f"- **{_pct(worse, len(pairs))} of rules scored worse hourly.** "
        f"Median change {_fmt(statistics.median(deltas), 1)} points.",
        "",
        "Bar size is not a free parameter. A rule pays its costs per decision, and moving "
        "to a bar twenty-four times shorter multiplies the decisions without multiplying "
        "the signal.",
        "",
    ]


def direction_effect(cells: list[dict]) -> list[str]:
    """Long-only rules against symmetric ones, on the same instruments.

    Read carefully. This is an observational split, not a controlled one: the two groups
    are different rules, not one rule traded two ways. It is reported because the gap is
    large and consistent, and because the mechanism is not mysterious -- a symmetric rule
    spends a good share of its life short an asset class with a positive expected return,
    and pays borrow for the privilege.
    """
    from strategies.canon import by_name

    groups: dict[bool, list[dict]] = {True: [], False: []}
    for c in cells:
        try:
            groups[by_name(c["strategy"]).long_only].append(c)
        except KeyError:
            continue
    if not groups[True] or not groups[False]:
        return []

    def line(label: str, g: list[dict]) -> str:
        s = [c["score"] for c in g]
        sr = [c["sharpe_annual"] for c in g if c["sharpe_annual"] is not None]
        good = sum(1 for c in g if c["label"] in ("SURVIVED", "PLAUSIBLE"))
        names = sorted({c["strategy"] for c in g})
        return (f"| {label} | {len(names)} | {len(g):,} | "
                f"{_fmt(statistics.median(s))} | {_fmt(_q(s, 0.75))} | "
                f"{_pct(good, len(g))} | "
                f"{_fmt(statistics.median(sr), 2) if sr else '-'} |")

    return [
        "## The short leg",
        "",
        "| Rules | Distinct | Cells | Median | 75th | PLAUSIBLE+ | Median SR |",
        "|---|---:|---:|---:|---:|---:|---:|",
        line("Long only", groups[True]),
        line("Long and short", groups[False]),
        "",
        "The same trend idea is in this study twice: `golden-cross` goes long above the "
        "crossover and flat below it, `dual-ma` goes short instead of flat. They are not "
        "the same strategy with a switch flipped -- different periods, different sources "
        "-- so this is an observation rather than an experiment. But the direction of the "
        "gap is consistent and the mechanism is not subtle: a symmetric rule spends much "
        "of its life short an asset class with a positive expected return, and pays "
        "financing to do it.",
        "",
        "Worth stating because the symmetric version is what gets taught. The long-only "
        "reading is what the press reports and what the tactical-allocation literature "
        "actually tested.",
        "",
    ]


def window_effect(cells: list[dict]) -> list[str]:
    """Same rule, same asset, same bar size, different stretch of history.

    This comparison exists because the cadence comparison needed it. The first run showed
    the golden cross scoring 82 on eight years of daily bitcoin and 6 on hourly, which
    reads as a devastating verdict on trading faster -- until the same rule on the *same
    window* at daily scored 7.5. Almost all of the collapse was the window.

    Length and period move together here and cannot be separated: the shorter sample is
    also the more recent one, and a shorter sample is penalised on its own merits because
    deflation is less forgiving when there is less evidence. The comparison is reported as
    "choice of window" rather than as either one.
    """
    by_key: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for c in cells:
        if c["asset_class"] == "crypto":
            by_key[(c["strategy"], c["symbol"])][c["cadence"]] = c

    pairs = [(v["daily"], v["daily-matched"]) for v in by_key.values()
             if "daily" in v and "daily-matched" in v]
    if len(pairs) < 5:
        return []

    full = [f["score"] for f, _ in pairs]
    short = [s["score"] for _, s in pairs]
    flipped = sum(1 for f, s in pairs
                  if f["label"] in ("SURVIVED", "PLAUSIBLE")
                  and s["label"] not in ("SURVIVED", "PLAUSIBLE"))
    good_full = sum(1 for f, _ in pairs if f["label"] in ("SURVIVED", "PLAUSIBLE"))

    # The median shortening is small and badly misrepresents the effect: most of the
    # universe listed recently and barely loses anything, while the few long histories
    # lose several years each. Report the split, and use the cells that barely moved as
    # a control -- if those disagreed, the comparison would be measuring a bug.
    lost = [f["years"] - s["years"] for f, s in pairs]
    barely = [(f, s) for f, s in pairs if f["years"] - s["years"] < 0.25]
    heavy = [(f, s) for f, s in pairs if f["years"] - s["years"] >= 1.0]
    agree = sum(1 for f, s in barely if abs(f["score"] - s["score"]) < 5)

    out = [
        "## Choosing the window",
        "",
        "The same rules, the same instruments, the same daily bars — scored over all "
        "available history, and then over only the stretch the hourly feed also covers.",
        "",
        f"- Median score **{_fmt(statistics.median(full))} on the full history** vs "
        f"**{_fmt(statistics.median(short))} on the recent window**, across "
        f"{len(pairs)} pairs.",
        f"- **{flipped} of {good_full}** cases that reached PLAUSIBLE or better on the "
        f"full history failed to on the shorter one.",
        "",
        f"The median history lost is only {_fmt(statistics.median(lost), 1)} years, and "
        f"that number is misleading. Most of this universe listed recently and loses "
        f"almost nothing; the long-lived pairs lose {_fmt(_q(lost, 0.9), 1)} years at the "
        f"90th percentile — and what they lose is the 2021 bull market, which is where a "
        f"crypto trend rule earned everything it earned.",
        "",
    ]
    if barely:
        out += [
            f"That split is also the control. In the {len(barely)} pairs whose two windows "
            f"differ by under three months, **{agree} agree within five points** — so the "
            f"comparison is measuring the history that was removed, not an error in "
            f"removing it.",
            "",
        ]
    if heavy:
        hf = statistics.median([f["score"] for f, _ in heavy])
        hs = statistics.median([s["score"] for _, s in heavy])
        out += [
            f"In the {len(heavy)} pairs that lose a year or more, the median goes "
            f"**{_fmt(hf)} → {_fmt(hs)}**.",
            "",
        ]
    out += [
        "Two effects are tangled here and cannot be separated with this design: the "
        "shorter window is also the *more recent* one, and a shorter sample is penalised "
        "on its own merits, because deflation is less forgiving when there is less "
        "evidence to deflate. Both are real, and both are things a backtester chooses.",
        "",
        "This section exists because it changed a conclusion. The cadence comparison "
        "below initially looked like a rout — a rule scoring 82 daily and 6 hourly — "
        "until the same rule over the same window at daily scored 7.5. Nearly all of that "
        "gap was the window, and attributing it to the bar size would have been wrong. "
        "The date range is a researcher degree of freedom like any other, and it is the "
        "one nobody reports.",
        "",
    ]
    return out


def asset_class_split(cells: list[dict]) -> list[str]:
    groups = defaultdict(list)
    for c in cells:
        groups[(c["asset_class"], c["cadence"])].append(c)

    rows = ["## By market", "",
            "| Market | Cadence | Cells | Median | PLAUSIBLE+ | Median years |",
            "|---|---|---:|---:|---:|---:|"]
    for (ac, cd), g in sorted(groups.items()):
        s = [c["score"] for c in g]
        good = sum(1 for c in g if c["label"] in ("SURVIVED", "PLAUSIBLE"))
        yrs = [c["years"] for c in g if c["years"] is not None]
        rows.append(f"| {ac} | {cd} | {len(g):,} | {_fmt(statistics.median(s))} | "
                    f"{_pct(good, len(g))} | {_fmt(statistics.median(yrs), 1)} |")
    rows.append("")
    return rows


# Equity index ETFs in this universe. Broad-market baskets, as distinct from the single
# names and the bond/commodity funds, because "index" is the home turf several of these
# rules were written for and the distinction is load-bearing below.
INDEX_ETFS = frozenset({"SPY", "QQQ", "DIA", "IWM", "EEM", "EFA"})
OTHER_ETFS = frozenset({"TLT", "GLD", "USO"})


def _venue(cell: dict) -> str:
    if cell["asset_class"] == "crypto":
        return "crypto"
    if cell["symbol"] in INDEX_ETFS:
        return "equity index ETF"
    if cell["symbol"] in OTHER_ETFS:
        return "bond/commodity ETF"
    return "single stock"


def home_turf(cells: list[dict]) -> list[str]:
    """Score each rule where its own source said it worked, and where it did not.

    Added after publication, because a reader pointed out that running an index-reversion
    rule on an altcoin is not a test of the rule. They were right, and the correction is
    large enough that the headline median cannot be read without it.

    This is a subgroup analysis proposed after seeing the results, which is the exact
    move this library exists to be suspicious of. Two things keep it honest, and neither
    makes it a controlled experiment: the domain labels come from reading each citation
    rather than from the scores, and they are printed below so the assignment can be
    argued with.
    """
    try:
        from strategies.canon import CANON
    except Exception:  # pragma: no cover - the study cannot run without the canon anyway
        return []

    domains = {c.name: c.domain for c in CANON}
    daily = [c for c in cells if c["cadence"] == "daily" and c["strategy"] in domains]
    if not daily:
        return []

    by_domain = defaultdict(list)
    for c in CANON:
        by_domain[c.domain].append(c.name)

    matched = [c for c in daily
               if domains[c["strategy"]] == "equity-index" and _venue(c) == "equity index ETF"]
    away = [c for c in daily
            if domains[c["strategy"]] == "equity-index" and _venue(c) != "equity index ETF"]
    if not matched or not away:
        return []

    all_med = statistics.median([c["score"] for c in cells])
    m_med = statistics.median([c["score"] for c in matched])
    a_med = statistics.median([c["score"] for c in away])
    m_good = sum(1 for c in matched if c["label"] in ("SURVIVED", "PLAUSIBLE"))

    out = [
        "## Home turf",
        "",
        "A rule tested somewhere its author never claimed it worked is not being tested. "
        "Every citation was read and labelled with the market **the source itself used**, "
        "before any of these scores were looked at:",
        "",
        "| Domain in the source | Rules |",
        "|---|---|",
    ]
    for dom in ("equity-index", "futures", "cross-asset", "unstated"):
        if by_domain.get(dom):
            out.append(f"| `{dom}` | {', '.join(f'`{n}`' for n in sorted(by_domain[dom]))} |")
    out += [
        "",
        f"For the six rules whose sources tested equity indices, this universe contains "
        f"the matching venue. On daily bars, scored on index ETFs against everywhere "
        f"else:",
        "",
        f"- **On home turf: median {_fmt(m_med)}**, {_pct(m_good, len(matched))} reaching "
        f"PLAUSIBLE or better, across {len(matched)} cells.",
        f"- Everywhere else: median {_fmt(a_med)}, across {len(away)} cells.",
        f"- The study-wide median is {_fmt(all_med)}.",
        "",
        "**That is the single largest effect in this study, and it qualifies the headline "
        "number rather than sitting beside it.** A good part of the overall median is "
        "rules being scored in markets they never claimed.",
        "",
    ]

    # The finding that actually matters, and the one nobody asked about.
    futures = sorted(by_domain.get("futures", []))
    if futures:
        away = [c for c in daily
                if domains[c["strategy"]] == "futures" and c["asset_class"] != "futures"]
        f_med = statistics.median([c["score"] for c in away]) if away else float("nan")
        out += [
            f"**A third of the canon is not from this universe at all.** "
            f"{', '.join(f'`{n}`' for n in futures)} come from commodity and financial "
            f"futures — Wilder and Lane developed on commodities, the Turtles traded "
            f"futures, and LeBeau's book has it in the title. Their median of "
            f"{_fmt(f_med)} across {len(away)} equity and crypto cells is therefore not a "
            f"verdict on them; it measures what happens when a futures system is pointed "
            f"at equities and crypto, which is what most retail platforms invite you to "
            f"do. Whether that is the *only* reason they fail is answered directly below, "
            f"because futures were added to find out.",
            "",
        ]

    # The direct test of the gap named above, once futures were added.
    fut_cells = [c for c in cells if c["asset_class"] == "futures"]
    if fut_cells:
        fut_rules = [c for c in fut_cells if domains.get(c["strategy"]) == "futures"]
        other_rules = [c for c in fut_cells if domains.get(c["strategy"]) != "futures"]
        away = [c for c in daily
                if domains.get(c["strategy"]) == "futures" and c["asset_class"] != "futures"]
        if fut_rules and other_rules:
            out += [
                "### Then the futures rules were given futures",
                "",
                f"Seven contracts were added to close the gap above. **It did not go the "
                f"way the equity-index result did.** The six futures systems score a "
                f"median of **{_fmt(statistics.median([c['score'] for c in fut_rules]))}** "
                f"on futures, against "
                f"{_fmt(statistics.median([c['score'] for c in away]))} on the equities "
                f"and crypto they were never meant for. Home turf bought them nothing. "
                f"They are the bottom six rules in the table below, and every other rule "
                f"in the canon beats them on their own ground "
                f"({_fmt(statistics.median([c['score'] for c in other_rules]))} median).",
                "",
                "**This is not the finding it looks like, and the reason is the data.** "
                "Only contracts whose free continuous series could be validated are here: "
                "metals, soybeans, and the financials. Crude, natural gas and corn are "
                "excluded because their series carry no roll return and overstate what is "
                "achievable by up to 23%/yr. Those excluded markets are a large part of "
                "where these systems were actually developed and traded -- the Turtles "
                "were in energy and grains, not in copper. So the honest statement is "
                "narrow: **the futures systems did not recover on the futures that can be "
                "trusted here, and those are not the futures they were built for.**",
                "",
                "One further reason to read this narrowly: most of what does well on "
                "futures is on `ES=F`, an equity index contract, and it is the "
                "equity-index rules that do it. That is the section above reappearing "
                "rather than anything about futures.",
                "",
            ]

    # The counterweight: matching the domain does not rescue everything.
    unstated = [c for c in daily if domains[c["strategy"]] == "unstated"]
    if unstated:
        u_med = statistics.median([c["score"] for c in unstated])
        out += [
            f"The rules whose sources name no market at all — "
            f"{', '.join(f'`{n}`' for n in sorted(by_domain['unstated']))} — sit at a "
            f"median of {_fmt(u_med)}. There is nowhere to move them to. A rule that "
            f"never said where it worked cannot be defended on the grounds that it was "
            f"being tested in the wrong place.",
            "",
        ]
    return out


def caveats(run: dict, cells: list[dict]) -> list[str]:
    fps = sorted({(c["symbol"], c["cadence"], c["fingerprint"]) for c in cells})
    env = json.loads(run["environment"]) if run["environment"] else {}
    return [
        "## What this does not establish",
        "",
        "**The true number of trials is far larger than the grid.** Each cell deflates by "
        "a few dozen combinations. The real search behind `RSI(14) at 30/70` is fifty "
        "years of practitioners trying everything and publishing what worked. Every score "
        "here is therefore an **upper bound**.",
        "",
        "**The futures series are front-month splices, and three markets were dropped "
        "because of it.** Free continuous futures data carries no roll return, so it "
        "tracks something spot-like that nobody can hold. Measured against a fund that "
        "does hold the asset, the wedge is 0.3%/yr for gold and 23.3%/yr for natural "
        "gas. Anything above ~1.5%/yr was excluded, which removed crude, natural gas and "
        "corn -- a large part of where the futures systems in this canon were actually "
        "developed. What remains is metals, soybeans and the financials, and even there "
        "the series is a splice rather than a back-adjusted contract. Doing this properly "
        "needs data that costs money.",
        "",
        "**Survivorship runs through the whole asset list.** Every instrument still "
        "trades. Companies that went to zero are absent; delisted crypto is absent twice "
        "over, because exchanges remove the pair and the history with it. This flatters "
        "long-biased rules. Index ETFs mitigate it and nothing free eliminates it.",
        "",
        "**These are clean-room implementations, not the sources' code.** They were "
        "written from published descriptions against a truncation test, so the causality "
        "results say nothing about the lookahead rate in real implementations -- which is "
        "the more interesting question and a different study.",
        "",
        "**A low score is not proof a rule does not work.** It is a statement that this "
        "sample cannot distinguish it from noise once the search that produced it is "
        "priced in. That is a weaker claim, and it is the only one the arithmetic supports.",
        "",
        "**Costs are modelled, not realised.** Retail rates on liquid instruments, with no "
        "market impact, no partial fills and no borrow that ever goes special. Real "
        "execution is worse, so this errs toward flattering.",
        "",
        "## Reproducing it",
        "",
        "```bash",
        "python -m corpus.run --out corpus/study.db",
        "python -m corpus.aggregate",
        "```",
        "",
        f"- falsify `{run['falsify_version']}`, commit `{run['git_sha']}`",
        f"- Python {env.get('python', '?')}, numpy {env.get('numpy', '?')}, "
        f"scipy {env.get('scipy', '?')}",
        f"- {run['n_permutations']} permutations per cell, study seed `{run['study_seed']}`",
        f"- Per-cell seeds derive from the study seed and the cell's identity, so a "
        f"single cell run alone reproduces its number exactly.",
        "",
        f"Data fingerprints (SHA-256 over closes and timestamps, first 16 hex) for all "
        f"{len(fps)} series are in `results.csv`. Free feeds revise history; if your "
        f"fingerprints differ, your data differs, and the numbers should be expected to.",
        "",
    ]


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------


def build(run: dict, cells: list[dict], findings: dict) -> str:
    when = datetime.now(timezone.utc).date().isoformat()
    head = [
        "# The published canon, prosecuted",
        "",
        f"*Generated {when} from `corpus/study.db`, run `{run['run_id']}`.*",
        "",
        "Eighteen of the best-known published trading rules, each at the parameters its "
        "own source named, run against every instrument in a fixed universe and attacked "
        "with the same six tests. No optimisation, no parameter selection, no choosing "
        "the window afterwards.",
        "",
        "The question is not whether these rules can be made to look good -- anything can "
        "-- but whether the versions people actually trade survive being told how large "
        "the search behind them was.",
        "",
    ]
    parts = [
        head, headline(cells, findings), by_label(cells), by_test(cells, findings),
        by_family(cells), direction_effect(cells), search_premium(cells),
        window_effect(cells), cadence_effect(cells), asset_class_split(cells),
        home_turf(cells), by_strategy(cells), caveats(run, cells),
    ]
    return "\n".join(line for part in parts for line in part).rstrip() + "\n"


CSV_COLUMNS = [
    "strategy", "family", "symbol", "asset_class", "kind", "cadence",
    "score", "label", "broken", "sharpe_annual", "best_sharpe_annual", "search_premium",
    "n_trials", "bars", "years", "first_date", "last_date", "fingerprint",
    "shipped_json", "best_params_json", "seed",
]


def write_csv(cells: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for c in sorted(cells, key=lambda c: (c["strategy"], c["symbol"], c["cadence"])):
            w.writerow(c)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate a falsify corpus study.")
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--run-id", default="main")
    ap.add_argument("--out", type=Path, default=HERE / "FINDINGS.md")
    ap.add_argument("--csv", type=Path, default=HERE / "results.csv")
    args = ap.parse_args(argv)

    run, cells, findings = load(args.db, args.run_id)
    args.out.write_text(build(run, cells, findings), encoding="utf-8")
    write_csv(cells, args.csv)

    print(f"{len(cells):,} cells -> {args.out}")
    print(f"{len(cells):,} rows  -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
