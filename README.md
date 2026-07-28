# falsify

[![tests](https://github.com/josephlangstroth-debug/falsify/actions/workflows/ci.yml/badge.svg)](https://github.com/josephlangstroth-debug/falsify/actions/workflows/ci.yml)
[![python](https://img.shields.io/pypi/pyversions/falsify-quant)](https://pypi.org/project/falsify-quant/)
[![license](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)

**Every backtesting tool helps you search for parameters that look good. This one assumes you already did that, and tries to prove what you found is an artifact of the search.**

There is no optimizer here. There is no knob that improves a score.

```
94/100  SURVIVED        Nothing here killed it.
 8/100  NO EDGE FOUND   The backtest is consistent with there being nothing here at all.
 0/100  BROKEN          LOOKAHEAD. Decisions change when future bars are withheld.
```

---

## Why this doesn't already exist

The math has been public for a decade — Bailey and López de Prado published the Deflated Sharpe Ratio and the Probability of Backtest Overfitting in 2014. It lives in research libraries that assume you already know what a deflated Sharpe is.

The commercial tools are built by companies that sell indicators and strategy generators. They are structurally incapable of shipping a product whose main output is "your strategy is nothing."

## The one design decision that matters

**`falsify` runs your parameter sweep itself.** It cannot accept an uploaded backtest report, and that isn't a limitation — it's the whole point.

The Deflated Sharpe Ratio needs two numbers no report contains: **N**, how many variants you actually tried, and **V**, the variance of their Sharpe ratios. Ask a human for N and they say "a few." The real answer is the size of the grid times the number of times they re-ran it after not liking the answer.

A tool that scores a submitted equity curve cannot deflate anything, because the search that produced it happened somewhere the tool cannot see.

---

## Install

```bash
pip install falsify-quant
```

numpy and scipy. Nothing else — no pandas, no build step, no account.

From source:

```bash
git clone https://github.com/josephlangstroth-debug/falsify && cd falsify
pip install -e ".[dev]"
```

## Use

Your strategy is a pure function from bars to target weights:

```python
# strategies/trend.py
import numpy as np
from falsify.examples import rolling_mean

GRID = {"fast": [5, 8, 12, 20, 30, 50], "slow": [60, 90, 120, 160, 200, 250]}

def valid(p):
    return p["fast"] < p["slow"]

def strategy(bars, fast=20, slow=100):
    f, s = rolling_mean(bars.close, int(fast)), rolling_mean(bars.close, int(slow))
    w = np.where(f > s, 1.0, -1.0)
    w[np.isnan(f) | np.isnan(s)] = np.nan   # warmup -> flat
    return w
```

`falsify` applies the execution lag, charges the costs, and **checks your causality claim rather than taking your word for it.**

```bash
python -m falsify strategies/trend.py --symbol BTC-USD --market crypto-perp --interval 1h
python -m falsify strategies/trend.py --symbol SPY     --market equity
python -m falsify strategies/trend.py --csv my_fills.csv --market crypto-spot
```

Data loaders need no API keys — Coinbase public candles for crypto, Yahoo for equities. If you have real fills from your own bot, use those instead: your fills know things about slippage no free daily bar ever will.

### Reference implementations

`falsify.indicators` is a causal indicator set — EMA, Wilder smoothing, RSI, ATR, Bollinger, Keltner, Donchian, stochastic, %R — under one contract: `out[i]` depends on `x[:i+1]`, and warmup is NaN rather than backfilled. Every one of them is unit-tested by truncation, because a leak in the indicator layer is invisible in the signal layer.

`strategies/canon.py` implements eighteen published rules against those primitives, each with a citation and **the parameters its own source named** — 50/200, RSI(14) at 30/70, Bollinger(20, 2), Turtle 20/10. Useful as a baseline, or to check your own implementation against something that has been made to prove it does not read ahead.

```python
from strategies.canon import by_name
c = by_name("donchian")
verdict = falsify.run(c.fn, bars=bars, spec=spec, grid=c.grid,
                      valid=c.valid, params=c.shipped)   # score what the book says
```

---

## The prosecution

Seven independent attempts to kill the strategy, roughly in the order they land.

### 1. Causality — the only test that finds bugs rather than weaknesses

Re-runs your strategy on truncated history and checks its past decisions are unchanged. If `w[t]` computed from `bars[0:k]` differs from `w[t]` computed from the full series for any `t < k`, the strategy is reading the future.

This is exact, not statistical. There is no p-value to argue with. It catches centered rolling windows, `pandas.rolling(center=True)`, `savgol_filter`, backward fills, full-sample normalization compared against a fixed threshold, and any indicator computed over the whole array before slicing.

**A lookahead finding is a gate, not a term.** A strategy that reads the future doesn't get a score, because there's nothing to score.

### 2. Costs — usually the whole story

Compares the edge earned per unit of turnover against what that turnover costs. Because net return is linear in the cost rate, the breakeven cost is exact rather than searched: gross P&L divided by total turnover.

> Dead on costs. The edge is 4.1 bps per unit of turnover and trading costs 5.0 bps. You are paying 1.2x your edge to collect it.

Most retail strategies die here and the arithmetic takes one line.

### 3. Deflation — charging the Sharpe for the size of the search

> Sharpe 0.83 is worse than the search alone explains. Trying 36 variants is expected to yield 2.48 on pure noise. Confidence this is real: 13%.

Also reports Minimum Track Record Length — how long you'd have to trade live to prove the edge. It is routinely longer than a human lifetime.

### 4. Out-of-sample rank — does the in-sample winner keep winning?

Combinatorially Symmetric Cross-Validation over all 12,870 ways to split the history into equal halves.

**One improvement on the textbook method.** PBO asks whether the *ranking* generalizes, which is not the same question as whether the strategy works. If every variant in the grid carries a real edge, the ranking among them is near-random and PBO climbs toward 0.5 — while the thing you actually care about, "does the variant I picked still make money out of sample," is a resounding yes. That's a plateau, and a plateau is the good case. `falsify` measures the absolute out-of-sample level alongside the rank and lets either form of robustness carry the finding.

### 5. Search on noise — run the same search on data with nothing in it

Manufactures histories with your market's drift and volatility but no exploitable structure, then runs the identical parameter sweep on each.

> Indistinguishable from noise. Running this exact search on 100 structureless histories produced a result this good 50% of the time.

The test people find hardest to argue with, because it makes no assumption about the strategy at all — it measures the search procedure itself.

Two nulls, and the choice matters. `--null iid` destroys all serial structure, isolating the search. `--null block` preserves volatility clustering and short-horizon autocorrelation — strictly harder. A strategy whose entire edge *is* lag-1 autocorrelation passes the first and fails the second, and both facts are worth knowing.

### 6. Universe selection — the search you ran before you wrote the grid

Every test above takes the asset as given. That's a large blind spot, because choosing *what to trade* is itself a search, usually bigger than the parameter sweep and almost never documented.

> You try a 200-day trend filter on eighteen crypto pairs. It looks great on two. You ship those two. The backtest for those two is honest, the grid was small, the deflated Sharpe clears — and it's still an artifact, because the trial count that mattered was eighteen times larger than the one you deflated by.

```python
falsify.run_universe(strategy, bars_by_symbol, spec, grid,
                     chosen=["BTC-USD", "ETH-USD"], params={"window": 200})
```

Sweeps the whole candidate universe at the **same** shipped parameters, then asks whether the chosen set is explainable by picking the best k of N. That question has an exact answer — enumerate all C(N,k) subsets and count how many do at least as well. No asymptotics, no distributional assumption, just the complete null. Reports breadth (what fraction of the universe works), the median asset's return, and a Sharpe deflated by **grid × assets** rather than grid alone.

**Assets are aligned to a common window first, and this is not optional.** Without it the ranking measures *when each asset happened to list*, not which one the strategy works on — and it flatters whatever has the longest history. Crypto is especially treacherous here: listing dates cluster around bull markets, and delistings punch holes mid-series. On a real 17-asset pool this correction moved breadth from 71% to 47% and flipped the verdict from BROAD to SELECTION BIAS.

Two honest limits. The test measures whether the chosen assets were unusually good — not *why* they were chosen; a defensible non-return prior (liquidity, market cap) is a real answer. And it assumes the universe was searched. If you genuinely picked on day one and never looked at the rest, the deflation doesn't apply — though it's worth asking whether you'd have tried the second asset if the first had failed.

### 7. Regime spread

Whether the P&L is spread across time or is one lucky quarter, plus the handful of bars carrying the entire result.

> Profitable in 4/6 periods; the best 1% of bars carry 499% of all P&L.

---

## Live monitoring

Once a strategy is running, the question changes from "is this real?" to "is the thing running still the thing I tested?"

```python
from falsify.adapters import load_bot_db
from falsify.monitor import monitor

record = load_bot_db("trader.db")          # read-only; safe on a live database
monitor(record, spec, strategy=..., bars_by_symbol=..., params=...,
        since=LAST_CONFIG_CHANGE)
```

**It deliberately does not test for edge decay.** It cannot, and neither can anything else. falsify computes Minimum Track Record Length, and on the strategies this was built against the answers were 24 years and 467 years. At those Sharpes a drawdown is indistinguishable from bad luck for longer than you will be alive. A monitor that flashes red when equity dips is generating noise and calling it risk.

What *is* detectable in days:

| check | what it catches |
|---|---|
| **config age** | how much of the record was produced by the settings running *today* |
| **costs** | realised fees and slippage vs. what the backtest assumed — needs a handful of fills, not a track record |
| **divergence** | replay the strategy and compare bar by bar: did the bot hold what its own strategy called for? |
| **heartbeat** | a bot that has quietly stopped looks identical to one with no signals |
| **envelope** | is equity outside what the backtest's own bootstrapped distribution allows? |

Three of the five are exact. Only the envelope is statistical, and it's scoped to catch gross breakage rather than adjudicate performance.

**Three design decisions that came from pointing it at a real bot and being wrong:**

**Config age runs first, and everything else is scoped to it.** The first run compared today's mechanical trend engine against a month in which an LLM was making the decisions, and reported it as a failed exit. A strategy change doesn't announce itself — it shows up as a new shape of log line appearing and an old one stopping, which `strategy_eras()` surfaces.

**Divergence has two directions and they must never be summed.** *Missed* (strategy wanted a position, bot was flat) is usually the risk layer working — position caps, cooldowns, sector limits. *Extra* (bot held what the strategy didn't want) is the dangerous one: a failed exit, a stop that never fired. Summing them fired ALARM at 42% agreement on a bot whose caps had correctly vetoed 799 entries.

**A live equity curve is not a return series.** Deposits and withdrawals move it without anyone trading, and a funding event dwarfs any edge — a $300 deposit read as +203% in one step and turned a +5% month into +897%. Flows are detected and chain-linked around, then *reported* rather than silently dropped, because a violent move on a concentrated book can trip the same heuristic.

## Scoring

A **weighted geometric mean**, not an arithmetic one. That's a deliberate epistemic choice: a strategy is only as good as its weakest leg, and averaging lets one fatal flaw hide behind five comfortable passes. Under a geometric mean, any component near zero drags the whole score to zero — which is correct, because an edge that doesn't clear its trading costs is worth nothing no matter how stable its parameter surface is.

A test that could not be computed scores zero. In a tool whose entire job is skepticism, "uncomputable" must never read as "passed."

---

## Attested verdicts

A verdict is more useful as a *credential* than as a tool. Nobody pays to be told their
own strategy is nothing; people do care what a stranger's strategy is worth before wiring
money at it.

```bash
falsify mystrategy.py --symbol AAPL --attest
falsify --verify their-verdict.json     # exit 1 if tampered, 2 if unreadable
```

Verification does not merely re-hash. It **recomputes the headline score from the
individual findings** using the published weights, and checks that every scored test is
still present. That closes the two cheap forgeries:

| Forgery | Caught by |
|---|---|
| Raise the score | content hash |
| Raise the score *and* re-hash | score arithmetic — it now contradicts its own findings |
| Raise one finding and re-hash | score arithmetic |
| **Delete the test that failed**, recompute over the rest | coverage — the arithmetic is correct over what remains, so only the missing test gives it away |
| Delete the causality gate | its own check — causality is a gate, not a weighted term, so coverage would not see it |

The strategy source is **fingerprinted, not embedded** — you can attest without publishing
your code, and prove the match later if you ever hand it over.

**What it cannot do, stated plainly:** an attestation carries whatever date its author
typed, and nothing inside the document can contradict that. There is a test asserting
backdating is undetectable. To make the date mean something, publish the hash somewhere
with a clock you don't control and record where:

```bash
falsify mystrategy.py --symbol AAPL --attest --anchor git=https://github.com/you/r/commit/abc
```

Publish the hash **first**, then let time pass, then show the results. A hash published
afterwards proves only that you can use a hash function. Without an anchor, `verify`
reports the date as a caveat rather than passing it silently.

---

## Guards against silent corruption

Two mistakes that survive code review and quietly invalidate everything:

**Bar spacing vs. cost calendar.** Every annualized figure scales by `bars_per_year`. Load daily bars while the spec says hourly and your Sharpe is inflated sixfold with nothing looking wrong. Checked, not trusted.

**Downsampled data wearing a daily label.** Yahoo answers `range=max` with 403 rows spanning 33 years of SPY — monthly bars presented as daily. The loader requests explicit date windows instead.

---

## Two test suites, answering different questions

```bash
pytest               # 437 unit tests: is the arithmetic right?
python selftest.py   # 9 known-answer cases: does it measure anything?
```

Integration tests alone are not enough. A wrong constant in the deflation formula would
leave every known-answer case passing — noise still scores low, real signal still scores
high — while making every reported number subtly wrong. So the unit tests check the math
against things true independently of the implementation:

- **`expected_max_sharpe` against Monte Carlo.** The most load-bearing number in the library: draw N standard normals 40,000 times, take the max, compare to the closed form. Verified for N = 5 … 1000.
- **PSR must penalise negative skew.** Two series, matched mean and standard deviation, mirrored skew. A sign error on the skew term would make falsify *reward* the strategy shape that collects premiums and blows up — the most expensive mistake it could make.
- **MinTRL round-trips through PSR.** MinTRL says how many observations reach 95% confidence; feed that many back into PSR and it must return 0.95. Two separately written formulas closing a loop to 1e-6.
- **PBO under a true null lands at 0.5**, not 0. And a plateau — every variant carrying the same real edge — must score *well*, not as failure.
- **Cost accounting hand-computed.** `close = 100, 110, 121, 121` with `w = 1, 1, 0, 0` → `net = 0, 0.099, 0.10, -0.001`, asserted exactly. Every cost verdict rests on this.
- **The execution lag.** A strategy fed `sign(r[t])` must earn nothing; one fed `sign(r[t+1])` must earn `|r|` every bar.

**Writing these found a real bug in the flagship feature.** The lookahead detector was blind to a strategy reading exactly *one* bar ahead — it skipped the final bar of each truncated window, which is the only index where one-bar lookahead shows up. Off-by-one is the most common form of the bug it exists to catch. Fixed; both real bots re-verified and their verdicts unchanged.

It also renamed the library's `test_*` functions to `check_*`, because pytest was collecting them as test cases in any project importing falsify.

## Self-test

```bash
python selftest.py
```

Nine known-answer cases. There are exactly two ways a tool like this can be worthless:

| | |
|---|---|
| **A.** Best of 36 variants fit to pure GBM noise | must score **near zero** |
| **B.** Centered moving average (reads n/2 bars ahead) | must be caught as **BROKEN** |
| **B2.** Full-sample z-score vs. a fixed threshold | must be caught as **BROKEN** |
| **C.** Genuine slow trends, traded at the right horizon | must score **high** |
| **D.** Best 2 of 12 pure-noise assets, shipped | must be caught (**15/100**, basket DSR 0.01) |
| **E.** Two *arbitrary* assets from a universe that all trends | must **not** be called selection bias (**100/100**) |
| **F.** Bot that follows its strategy exactly | must **not** alarm |
| **G.** Bot whose exit silently stopped firing | must **ALARM** on the *extra* direction |
| **H.** Equity curve containing a deposit | must report **+2%**, not the raw **+308%** |

Case C is the one that actually constrains the design. It is easy to build a tool that says no to everything — that sounds rigorous and is exactly as informative as a stopped clock.

Two things worth knowing, both discovered by these tests failing:

**Not every leak is a leak.** Normalizing by full-sample mean and standard deviation is invisible to a `fast > slow` comparison, because an affine transform applied to both sides cancels. It only bites when compared against a constant. Whether preprocessing leaks depends on what you do with it downstream — which is why the causality test checks behavior rather than inspecting code.

**"No signal" and "wrong instrument for the signal" look identical on an equity curve.** An earlier version of case C used AR(1) returns and failed. AR(1) momentum lives at lag 1 — autocorrelation decays as φ^k, so at φ=0.35 it's 0.35 at lag 1 and 0.004 by lag 5. A 20/90 crossover structurally cannot see it, no matter how large φ gets. The signal was real; the strategy was the wrong instrument. That distinction is now preserved as `momentum_market` vs `trending_market`.

---

## The corpus study

Eighteen published rules × thirty instruments × three cadences, each scored **at the
parameters its own source named**, not at the best of a grid.

```bash
python -m corpus.run          # resumable, deterministic per cell
python -m corpus.aggregate    # writes corpus/FINDINGS.md and corpus/results.csv
```

Method and caveats in [`corpus/README.md`](corpus/README.md); results in
[`corpus/FINDINGS.md`](corpus/FINDINGS.md). Aggregate statistics and per-strategy medians
only — the study prosecutes the canon, which is scholarship, and does not point itself at
named commercial products, which is a different activity.

Every cell records the falsify version, the commit, the interpreter and library versions,
the grid, the shipped parameters, and a SHA-256 fingerprint of the price series. Per-cell
seeds derive from the cell's own identity rather than from a run counter, so a resumed
run, a partial run and a single cell run in isolation all produce the same number.

---

## A high score is not a prediction

It means these particular tests failed to kill the strategy. That is the only thing it means.

falsify cannot tell you a strategy will make money. Nothing can. It can tell you that a
specific set of ways to be fooled have been checked and did not fire, which is a smaller
claim and the only honest one available.

Two limits worth stating plainly:

**It deflates by the trials you hand it.** The grid is counted exactly. The choices you
made *before* writing the grid — which market, which side, which timeframe, which of the
eight ideas you tried first — are not, unless you use `run_universe`. The true trial count
is always larger than the one in the report.

**Edge decay is not detectable here, or anywhere.** The Minimum Track Record Length in
every report is the honest number: at the Sharpe ratios retail strategies post, separating
decay from an ordinary bad patch routinely takes decades of live data.

---

## License

AGPL-3.0-or-later. Free to use, modify and self-host, including commercially.

If you run a modified falsify as a **network service**, the AGPL requires you to offer
that service's users the corresponding source. Using it inside your own trading operation
— however commercial — triggers nothing; only offering it to others as a service does.

## Prior art

Deflated Sharpe Ratio, Probability of Backtest Overfitting and Combinatorially Symmetric
Cross-Validation are due to David Bailey, Jonathan Borwein, Marcos López de Prado and
Qiji Jim Zhu. The mathematics is theirs and has been public since 2014. What is here is
the packaging, the adversarial framing, the causality check, the universe-selection test
and the live monitor.
