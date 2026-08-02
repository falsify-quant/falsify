# falsify

[![tests](https://github.com/falsify-quant/falsify/actions/workflows/ci.yml/badge.svg)](https://github.com/falsify-quant/falsify/actions/workflows/ci.yml)
[![pypi](https://img.shields.io/pypi/v/falsify-quant)](https://pypi.org/project/falsify-quant/)
[![python](https://img.shields.io/pypi/pyversions/falsify-quant)](https://pypi.org/project/falsify-quant/)
[![license](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)

**You found a trading strategy that looks great in a backtest. This tells you whether it's real.**

Not by agreeing with you. By spending seven separate attempts trying to prove it isn't.

```bash
pip install falsify-quant
falsify-quant --demo
```

That scores three strategies you didn't write, on prices that don't exist, in a few
seconds — no network, no data feed, nothing to write first. A real edge survives, the
same rule on noise dies, and a version of it that peeks one bar into the future is
caught outright:

```
 95.4/100  SURVIVED         A genuine edge
  1.5/100  NO EDGE FOUND    The same rule on noise
  0.0/100  BROKEN           A rule that peeks one bar ahead
```

Then point it at your own:

```bash
falsify-quant mystrategy.py --symbol SPY --market equity
```

---

## Where this came from

I had a crypto bot trading real money, and an equities bot alongside it. Months of
walk-forward work, a backtest I believed, and a slow suspicion I could not make precise —
the thing was not losing, exactly, it just never did what the backtest said it would.

So I stopped tuning it and wrote something to attack it instead. The first honest verdicts
were these:

| what I had running | what it came back with |
|---|---|
| equities bot, five tickers | **0/100 — NO EDGE FOUND, on every one of them** |
| crypto trend engine | **53/100 — UNPROVEN** |
| 200-day overlay | **82/100 — SURVIVED** |

Not one of the five equity names cleared its own commission, and two had negative expected
return *before* costs. I had been paying to run them.

That is the whole reason this exists, and it is also the only credential it has. I did not
build a tool that agrees with me — I built one that took two live configurations away from
me, and I changed them. If it had passed everything of mine, you should not install it.

The same tool then ran against **890 backtests of eighteen published trading rules**, each
at the parameters its own source named. Median **3.7 out of 100**; **51% could not clear
their own trading costs** before overfitting arose at all. Read that alongside the *Home
turf* section though — scored in the markets their authors actually tested, the same rules
median **56.9**. Both numbers are in [`corpus/FINDINGS.md`](corpus/FINDINGS.md), and
reproducing it is two commands.

---

## The problem

You tested a trading idea against historical prices. You tried a few settings. One
combination returned 40% a year. The equity curve goes up and to the right.

Here is the uncomfortable part: **every tool you used to find that was designed to help
you find it.** Backtesters, optimisers, strategy builders — they search. If you try
enough combinations, one of them looks brilliant on any data at all, including data with
nothing in it. That is not a flaw in the strategy. It is arithmetic.

Nothing you own is designed to talk you out of it. That is what this is.

```
94/100  SURVIVED        Nothing here killed it.
 8/100  NO EDGE FOUND   The backtest is consistent with there being nothing here at all.
 0/100  BROKEN          LOOKAHEAD. Decisions change when future bars are withheld.
```

There is no optimiser here, and no setting that improves your score. That is deliberate.
A tool that could make your number go up would get used to make your number go up.

---

## What you actually do

Install it:

```bash
pip install falsify-quant
```

If you'd rather click than type:

```bash
falsify-quant-gui
```

That opens a page in your browser. Pick a market, point it at a strategy file, press Run.
No account, no upload, nothing leaves your machine — it's a small local server, and the
whole page is one file with no JavaScript dependencies.

**It counts how many times you've asked.** This matters more than it sounds. A graphical
interface makes re-running free: tweak the settings, press Run, watch the number move. Do
that six times and keep the best, and you have just performed the exact search this tool
exists to charge you for — except invisibly, because each run looks like a fresh question.

So the app remembers. Every run inside an *investigation* is added to the next verdict's
deflation, with a running total on screen:

```
3 searches this investigation, 108 combinations total.
The next verdict is deflated by all of them.
```

There's a **New investigation** button, because sometimes the next question genuinely is
unrelated. It tells you how many combinations you're discarding, and it asks first. That
difference — a button you press versus a default you never notice — is the whole design.

Write your strategy as a plain Python function. Prices go in; a number per bar comes out
saying how much you want to be holding — `1.0` fully long, `0.0` in cash, `-1.0` fully
short:

```python
# strategies/trend.py
import numpy as np
from falsify_quant.indicators import rolling_mean

# Every combination you want checked. falsify runs all of them.
GRID = {"fast": [5, 8, 12, 20, 30, 50], "slow": [60, 90, 120, 160, 200, 250]}

def valid(p):
    return p["fast"] < p["slow"]        # skip nonsense combinations

def strategy(bars, fast=20, slow=100):
    """Buy when the short average is above the long one, sell short when it isn't."""
    f = rolling_mean(bars.close, int(fast))
    s = rolling_mean(bars.close, int(slow))
    w = np.where(f > s, 1.0, -1.0)
    w[np.isnan(f) | np.isnan(s)] = np.nan   # not enough history yet -> stay out
    return w
```

Point it at a market:

```bash
falsify-quant strategies/trend.py --symbol SPY --market equity
```

You get a score out of 100, a sentence explaining it, and an HTML report. Prices are
fetched for you — no API keys, no account. Use `--csv yourfile.csv` for your own data.

### Why it insists on running the search itself

**falsify cannot score a backtest you already ran.** That sounds like a limitation. It is
the entire point.

To know whether a result is luck, you have to know how many things were tried. Ask anyone
how many combinations they tested and the answer is "a few." The real answer is the size
of the grid, times the number of times they changed the grid after not liking the answer.

A tool handed a finished equity curve has no way to know that number, so it cannot correct
for it. falsify runs your grid itself, so it counts honestly.

---

## The strategy file

This is the input format — what a strategy has to look like for falsify to run it. It is
not advice on finding an edge, and there is none of that anywhere in this project. There
is no optimiser here and no setting that improves a score, so the only thing this section
can help you do is get your existing idea *in*.

For the avoidance of doubt about whose side the tool is on, here is what it said about the
three strategies its author had running on real money: the equities bot scored **0/100 on
all five names it traded**, the crypto trend engine **53/100 UNPROVEN**, and only the third
— a 200-day overlay — came back **82/100 SURVIVED**. Two live configurations were changed
because of it.

That spread is the point. A validator whose author reports that it cleared all his own
work would not be worth installing.

Don't start from a blank file:

```bash
falsify-quant --new mystrategy.py
```

That writes a working moving-average strategy you can score immediately, with the
contract in a comment at the top. Edit it into your idea.

The contract is small enough to state in full:

```python
def strategy(bars, fast=20, slow=100):    # any parameters you like, all with defaults
    c = bars.close                        # numpy array, NOT a DataFrame column
    ...
    return w                              # one weight per bar, same length as bars.close

GRID = {"fast": [10, 20, 40], "slow": [50, 100, 200]}
```

- **`bars` is a `Bars` object, not a DataFrame.** `bars.close`, `bars.open`, `bars.high`,
  `bars.low`, `bars.volume` are plain numpy arrays. Pandas users: `pd.Series(bars.close)`.
- **Return one weight per bar.** `1.0` fully long, `0.0` flat, `-1.0` fully short;
  fractions are fine. Same length as `bars.close`.
- **Use `np.nan` for the warmup**, not a shorter array. The engine reads NaN as flat.
- **Use only bars up to and including `i`** when computing weight `i`. This is checked,
  not trusted — see [Causality](#1-causality--the-only-test-that-finds-bugs-rather-than-weaknesses).
- **Don't apply execution lag or costs yourself.** The engine does both. Shifting your own
  signal forward a bar gets you charged the lag twice.

Optionally define `valid(params) -> bool` to skip nonsensical combinations
(`fast >= slow`). Combinations rejected there are not counted as trials, so they cost
you nothing in deflation.

**Keep `GRID` small.** Every combination is a lottery ticket the deflation has to charge
you for. Twelve trials you can justify beats two hundred you cannot — and unlike every
other backtesting tool, here a bigger search makes your score *worse*, which is the
intended incentive.

### Porting a strategy you already have

Most people don't have a weight series. They have **entries and exits** — that's
vectorbt's `from_signals(entries, exits)`, backtrader's `buy()`/`sell()`, Pine's
`strategy.entry`/`strategy.close`, and how anyone describes an idea out loud. There's a
helper for exactly that translation:

```python
from falsify_quant import from_signals

def strategy(bars, fast=20, slow=100):
    f, s = sma(bars.close, fast), sma(bars.close, slow)
    return from_signals(entries=f > s, exits=f < s, warmup=slow)
```

The rule is **last event wins**, which answers the three questions the event model
leaves open, the way people actually mean them:

- an entry while already long is a no-op, **not a second position** — getting this wrong
  is how a hand-written port silently trades twice;
- an exit while already flat is a no-op;
- before the first event, you are flat.

On a bar with conflicting signals the priority is exits > short entries > long entries.
It takes `short_entries` / `short_exits` for two-sided strategies, and `size=` to scale.
The conversion only ever forward-fills, so it is causal by construction — and there is a
test asserting that, because it's the property falsify would otherwise fail you for.

**If you already have a position series** — a pandas column, a list, a numpy array —
there is nothing to translate:

```python
from falsify_quant import from_positions

def strategy(bars, lookback=20):
    sig = my_existing_backtest(bars.close, lookback)   # returns 0/1 per bar
    return from_positions(sig, warmup=lookback)
```

`bars.close` is a numpy array, so anything that reads one works — `pd.Series(bars.close)`
included.

**What you cannot port is a finished equity curve or a trade log.** Those don't say how
many parameter combinations you tried to get there, and without that number there is
nothing to deflate. That is the one constraint this tool cannot relax without becoming
the thing it was built to check.

### If it won't run

falsify names the actual mistake rather than making you guess. The four most common:

| what you wrote | what you get told |
|---|---|
| `bars["close"]` | `'Bars' object is not subscriptable` — use `bars.close`; it's not a DataFrame |
| a `GRID` key that isn't a parameter | the offending key, next to the parameters your `strategy` does accept |
| forgot the `return` | `strategy() returned None -- did you forget to return?` |
| returned a shorter array | `returned 1496 weights for 1500 bars`, and to pad the warmup with NaN |

If only *some* combinations fail, the run continues on the rest and prints a warning
saying how many were skipped and why. Those cells are excluded from the deflation, so
the trial count reflects what actually ran — but you are told, because a grid that
half-ran is not the grid you think you searched.

---

## Reading the score

| Score | Verdict | What it means |
|---:|---|---|
| 80–100 | **SURVIVED** | None of these tests killed it. Not proof it works — the absence of proof it doesn't. |
| 60–79 | **PLAUSIBLE** | Passed the serious tests, comfortably on some and narrowly on others. Worth paper trading. |
| 40–59 | **UNPROVEN** | Not obviously broken, not demonstrably real. Your sample can't tell the difference yet. |
| 20–39 | **LIKELY OVERFIT** | Most of what looks like edge is explained by the size of your search. |
| 0–19 | **NO EDGE FOUND** | Consistent with there being nothing here at all. |
| 0 | **BROKEN** | The strategy used information it could not have had. Fix the bug and re-run. |

**A high score is not a prediction.** It means seven specific ways of being fooled were
checked and none fired. falsify cannot tell you a strategy will make money. Nothing can.

---

## What it checks, in plain English

Seven attempts to kill your strategy, roughly in the order they land.

| | The mistake | How it's caught |
|---|---|---|
| **1** | **Your code peeks at the future.** Almost always by accident — one line that averages across today instead of up to today. | Re-runs your strategy with the later data deleted. If yesterday's decision changes, it was reading tomorrow. Exact, not statistical. |
| **2** | **The profit doesn't survive trading costs.** Fees and spread eat an edge that only ever existed on paper. | Works out exactly what cost rate reduces you to zero, and compares it to what you actually pay. |
| **3** | **You tried enough settings that one had to look good.** | Works out how good the *best of nothing* would have looked, given how many combinations you ran, and subtracts it. |
| **4** | **The winning settings only won on the data you picked them on.** | Splits the history 12,870 different ways and checks whether the in-sample winner keeps winning on the half it never saw. |
| **5** | **Your whole method finds "edges" in random data.** | Builds fake price histories with no pattern in them at all, runs your identical search on each, and counts how often it does this well by luck. |
| **6** | **You picked the asset after seeing which one worked.** Choosing *what* to trade is a search too, usually a bigger one than the settings. | Runs the same settings across your whole candidate list and asks whether picking those winners was luck. |
| **7** | **All the profit is one lucky quarter, or ten lucky bars.** | Splits the history into periods and checks the profit is spread across them rather than concentrated. |

Number 2 is where most retail strategies die, and the arithmetic takes one line. Number 1
is the only one that finds a *bug* rather than a weakness — and it finds them constantly.

---

## Words you'll see

Unavoidable jargon, defined once.

| Term | Meaning |
|---|---|
| **backtest** | Running a strategy against historical prices to see what it would have done. |
| **overfitting** | Tuning a strategy until it fits the past exactly, including the parts of the past that were random. |
| **lookahead** | Accidentally using information from the future. The most common serious bug in strategy code. |
| **Sharpe ratio** | Return divided by volatility. Roughly "how much profit per unit of nerves." Below 1 is normal for retail; above 2 usually means a bug. |
| **turnover** | How much you trade. Moving 100% of your capital from cash to a position and back is two units of turnover, and you pay costs on both. |
| **basis point (bp)** | One hundredth of a percent. Trading costs are quoted in these; 5bp is 0.05%. |
| **deflating** | Reducing a result to account for how many things were tried before it was found. |
| **out-of-sample** | Data the strategy was not tuned on. The only data whose opinion counts. |
| **p-value** | Roughly: the chance of a result this good happening by luck alone. Lower is better; 0.5 means a coin flip. |

---

## Ready-made strategies to compare against

`strategies/canon.py` implements **eighteen published trading rules** — the golden cross,
RSI, Bollinger bands, MACD, the Turtle breakout, and so on — each with a citation, and each
at **the settings its own source published**, not at whatever setting looks best.

They're useful two ways: as a baseline to beat, and as a reference implementation to check
your own code against. Every one is machine-verified not to read the future.

```python
from strategies.canon import by_name

c = by_name("donchian")                                  # Turtle 20/10 breakout
verdict = falsify.run(c.fn, bars=bars, spec=spec, grid=c.grid,
                      valid=c.valid, params=c.shipped)   # score what the book says
```

Underneath them, `falsify.indicators` is a causal indicator library — moving averages,
RSI, ATR, Bollinger, Keltner, Donchian, stochastic — all under one rule: the value at any
bar depends only on that bar and earlier ones. Each is unit-tested by deletion, because a
leak in the indicator layer is invisible in the strategy layer.

---

# How each check works

Everything above is the short version. This is the arithmetic.

## 1. Causality — the only test that finds bugs rather than weaknesses

Re-runs your strategy on truncated history and checks its past decisions are unchanged. If
`w[t]` computed from `bars[0:k]` differs from `w[t]` computed from the full series for any
`t < k`, the strategy is reading the future.

This is exact, not statistical. There is no p-value to argue with. It catches centred
rolling windows, `pandas.rolling(center=True)`, `savgol_filter`, backward fills,
full-sample normalisation compared against a fixed threshold, and any indicator computed
over the whole array before slicing.

**A lookahead finding is a gate, not a term.** A strategy that reads the future doesn't get
a score, because there's nothing to score.

## 2. Costs — usually the whole story

Compares the edge earned per unit of turnover against what that turnover costs. Because net
return is linear in the cost rate, the breakeven cost is exact rather than searched: gross
P&L divided by total turnover.

> Dead on costs. The edge is 4.1 bps per unit of turnover and trading costs 5.0 bps. You
> are paying 1.2x your edge to collect it.

## 3. Deflation — charging the Sharpe for the size of the search

> Sharpe 0.83 is worse than the search alone explains. Trying 36 variants is expected to
> yield 2.48 on pure noise. Confidence this is real: 13%.

Also reports **Minimum Track Record Length** — how long you would have to trade live to
prove the edge is real. It is routinely longer than a human lifetime, and that is not a
quirk of the formula. It is what a Sharpe of 0.4 actually means.

## 4. Out-of-sample rank — does the in-sample winner keep winning?

Combinatorially Symmetric Cross-Validation over all 12,870 ways to split the history into
equal halves.

**One improvement on the textbook method.** The standard measure asks whether the *ranking*
of your variants generalises, which is not the same question as whether the strategy works.
If every variant in the grid carries a real edge, the ranking among them is near-random and
the statistic climbs toward 0.5 — while the thing you actually care about, "does the
variant I picked still make money out of sample," is a resounding yes. That's a plateau,
and a plateau is the good case. falsify measures the absolute out-of-sample level alongside
the rank and lets either form of robustness carry the finding.

## 5. Search on noise — run the same search on data with nothing in it

Manufactures histories with your market's drift and volatility but no exploitable
structure, then runs the identical parameter sweep on each.

> Indistinguishable from noise. Running this exact search on 100 structureless histories
> produced a result this good 50% of the time.

The test people find hardest to argue with, because it makes no assumption about the
strategy at all — it measures the search procedure itself.

Two nulls, and the choice matters. `--null iid` destroys all serial structure, isolating
the search. `--null block` preserves volatility clustering and short-horizon
autocorrelation — strictly harder. A strategy whose entire edge *is* lag-1 autocorrelation
passes the first and fails the second, and both facts are worth knowing.

## 6. Universe selection — the search you ran before you wrote the grid

Every test above takes the asset as given. That's a large blind spot, because choosing
*what to trade* is itself a search, usually bigger than the parameter sweep and almost
never documented.

> You try a 200-day trend filter on eighteen crypto pairs. It looks great on two. You ship
> those two. The backtest for those two is honest, the grid was small, the deflated Sharpe
> clears — and it's still an artifact, because the trial count that mattered was eighteen
> times larger than the one you deflated by.

```python
falsify_quant.run_universe(strategy, bars_by_symbol, spec, grid,
                     chosen=["BTC-USD", "ETH-USD"], params={"window": 200})
```

Sweeps the whole candidate universe at the **same** shipped parameters, then asks whether
the chosen set is explainable by picking the best k of N. That question has an exact answer
— enumerate all C(N,k) subsets and count how many do at least as well. No asymptotics, no
distributional assumption, just the complete null. Reports breadth (what fraction of the
universe works), the median asset's return, and a Sharpe deflated by **grid × assets**
rather than grid alone.

**Assets are aligned to a common window first, and this is not optional.** Without it the
ranking measures *when each asset happened to list*, not which one the strategy works on —
and it flatters whatever has the longest history. Crypto is especially treacherous here:
listing dates cluster around bull markets, and delistings punch holes mid-series. On a real
17-asset pool this correction moved breadth from 71% to 47% and flipped the verdict from
BROAD to SELECTION BIAS.

Two honest limits. The test measures whether the chosen assets were unusually good — not
*why* they were chosen; a defensible non-return prior (liquidity, market cap) is a real
answer. And it assumes the universe was searched. If you genuinely picked on day one and
never looked at the rest, the deflation doesn't apply — though it's worth asking whether
you'd have tried the second asset if the first had failed.

## 7. Regime spread

Whether the P&L is spread across time or is one lucky quarter, plus whether a handful of
bars carry the entire result.

> Profitable in 5/6 periods; the best 1% of bars carry 31% of all P&L, 1.0x what this
> return distribution alone would give.

That last clause is the part that took two attempts to get right, and it is worth
explaining because the naive version is very tempting.

The obvious statistic is *what share of total profit comes from the best 1% of bars*, with
a fixed threshold — say, fail above 150%. It looks like a test of lumpiness. It isn't. Its
numerator is set by the volatility and its denominator by the edge, so for any ordinary
series it lands at roughly one over the signal-to-noise ratio. A weak-but-perfectly-even
edge scores identically to a genuinely spiky one, and against a fixed cutoff the weak one
fails *purely for being weak*. Deflation already charges for that. Charging twice, inside a
geometric mean, is not conservatism — it's double counting, and running the corpus study
showed it was failing nine cells in ten and dragging every published score down by a
near-constant factor.

So the observed share is now compared against what an ordinary series with the same mean
and the same scale would produce anyway. Landing at the expected share means your profit
arrived exactly as unremarkably as chance would deliver it, which is the honest null. The
scale comes from a median-based estimate rather than the standard deviation, because the
outliers being tested for inflate the standard deviation that is supposed to detect them —
testing for outliers with a statistic they contaminate is the standard version of this
mistake, and the first fix made it.

---

## Live monitoring

Once a strategy is running, the question changes from "is this real?" to "is the thing
running still the thing I tested?"

```python
from falsify_quant.adapters import load_bot_db
from falsify_quant.monitor import monitor

record = load_bot_db("trader.db")          # read-only; safe on a live database
monitor(record, spec, strategy=..., bars_by_symbol=..., params=...,
        since=LAST_CONFIG_CHANGE)
```

**It deliberately does not test for edge decay.** It cannot, and neither can anything else.
falsify computes Minimum Track Record Length, and on the strategies this was built against
the answers were 24 years and 467 years. At those Sharpes a drawdown is indistinguishable
from bad luck for longer than you will be alive. A monitor that flashes red when equity
dips is generating noise and calling it risk.

What *is* detectable in days:

| check | what it catches |
|---|---|
| **config age** | how much of the record was produced by the settings running *today* |
| **costs** | realised fees and slippage vs. what the backtest assumed — needs a handful of fills, not a track record |
| **divergence** | replay the strategy and compare bar by bar: did the bot hold what its own strategy called for? |
| **heartbeat** | a bot that has quietly stopped looks identical to one with no signals |
| **envelope** | is equity outside what the backtest's own bootstrapped distribution allows? |

Three of the five are exact. Only the envelope is statistical, and it's scoped to catch
gross breakage rather than adjudicate performance.

**Three design decisions that came from pointing it at a real bot and being wrong:**

**Config age runs first, and everything else is scoped to it.** The first run compared
today's mechanical trend engine against a month in which an LLM was making the decisions,
and reported it as a failed exit. A strategy change doesn't announce itself — it shows up
as a new shape of log line appearing and an old one stopping, which `strategy_eras()`
surfaces.

**Divergence has two directions and they must never be summed.** *Missed* (strategy wanted
a position, bot was flat) is usually the risk layer working — position caps, cooldowns,
sector limits. *Extra* (bot held what the strategy didn't want) is the dangerous one: a
failed exit, a stop that never fired. Summing them fired ALARM at 42% agreement on a bot
whose caps had correctly vetoed 799 entries.

**A live equity curve is not a return series.** Deposits and withdrawals move it without
anyone trading, and a funding event dwarfs any edge — a $300 deposit read as +203% in one
step and turned a +5% month into +897%. Flows are detected and chain-linked around, then
*reported* rather than silently dropped, because a violent move on a concentrated book can
trip the same heuristic.

### Leaving it running

```bash
falsify-quant-watch --example > watch.json    # edit it
falsify-quant-watch watch.json --check        # validate config and strategy, poll nothing
falsify-quant-watch watch.json                # daemon
falsify-quant-watch watch.json --once         # one cycle, for cron
```

The loop is worth less than what it refuses to say. Four rules:

**Only transitions notify.** A divergence that has been alarming for six hours is one piece
of news, not seventy-two.

**Recoveries are news too.** An alarm that stopped because it resolved and one that stopped
because the daemon died look identical from outside.

**Silence is a claim, so it's earned.** A heartbeat goes out on a fixed schedule whatever
the alert state — the absence of one is itself the signal. The first run always beats,
because "the watchdog is up" is the most useful thing it ever says.

**Flapping is throttled, not hidden.** A check that toggles every poll is a threshold
sitting on top of the data. The useful message is "this flapped nine times in an hour",
once, not nine alerts.

State is kept on disk and written atomically, so a restart doesn't re-announce everything
it already told you, and a kill mid-write doesn't leave a truncated file. A failure inside
the monitor becomes an event rather than an exception — a locked trading database is not a
reason to stop watching it.

---

## Attested verdicts

A verdict is more useful as a *credential* than as a tool. Nobody pays to be told their own
strategy is nothing; people do care what a stranger's strategy is worth before wiring money
at it.

```bash
falsify-quant mystrategy.py --symbol AAPL --attest
falsify-quant --verify their-verdict.json     # exit 1 if tampered, 2 if unreadable
```

Verification does not merely re-hash. It **recomputes the headline score from the individual
findings** using the published weights, and checks that every scored test is still present.
That closes the cheap forgeries:

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
falsify-quant mystrategy.py --symbol AAPL --attest --anchor git=https://github.com/you/r/commit/abc
```

Publish the hash **first**, then let time pass, then show the results. A hash published
afterwards proves only that you can use a hash function. Without an anchor, `verify`
reports the date as a caveat rather than passing it silently.

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
seeds derive from the cell's own identity rather than from a run counter, so a resumed run,
a partial run and a single cell run in isolation all produce the same number.

---

## Design notes

### Scoring

A **weighted geometric mean**, not an arithmetic one. That's a deliberate choice: a
strategy is only as good as its weakest leg, and averaging lets one fatal flaw hide behind
five comfortable passes. Under a geometric mean, any component near zero drags the whole
score to zero — which is correct, because an edge that doesn't clear its trading costs is
worth nothing no matter how stable its parameter surface is.

A test that could not be computed scores zero. In a tool whose entire job is scepticism,
"uncomputable" must never read as "passed."

### Guards against silent corruption

Two mistakes that survive code review and quietly invalidate everything:

**Bar spacing vs. cost calendar.** Every annualised figure scales by `bars_per_year`. Load
daily bars while the spec says hourly and your Sharpe is inflated sixfold with nothing
looking wrong. Checked, not trusted.

**Downsampled data wearing a daily label.** Yahoo answers `range=max` with 403 rows
spanning 33 years of SPY — monthly bars presented as daily. The loader requests explicit
date windows instead.

### Two test suites, answering different questions

```bash
pytest               # 490 unit tests: is the arithmetic right?
python selftest.py   # 9 known-answer cases: does it measure anything?
```

Integration tests alone are not enough. A wrong constant in the deflation formula would
leave every known-answer case passing — noise still scores low, real signal still scores
high — while making every reported number subtly wrong. So the unit tests check the maths
against things true independently of the implementation:

- **`expected_max_sharpe` against Monte Carlo.** The most load-bearing number in the
  library: draw N standard normals 40,000 times, take the max, compare to the closed form.
  Verified for N = 5 … 1000.
- **PSR must penalise negative skew.** Two series, matched mean and standard deviation,
  mirrored skew. A sign error on the skew term would make falsify *reward* the strategy
  shape that collects premiums and blows up — the most expensive mistake it could make.
- **MinTRL round-trips through PSR.** MinTRL says how many observations reach 95%
  confidence; feed that many back into PSR and it must return 0.95. Two separately written
  formulas closing a loop to 1e-6.
- **PBO under a true null lands at 0.5**, not 0. And a plateau — every variant carrying the
  same real edge — must score *well*, not as failure.
- **Cost accounting hand-computed.** `close = 100, 110, 121, 121` with `w = 1, 1, 0, 0` →
  `net = 0, 0.099, 0.10, -0.001`, asserted exactly. Every cost verdict rests on this.
- **The execution lag.** A strategy fed `sign(r[t])` must earn nothing; one fed
  `sign(r[t+1])` must earn `|r|` every bar.
- **Every indicator and every catalogue strategy, by deletion.** Delete the future, and the
  past must not move. A leak in the indicator layer would not show up as a failure — it
  would show up as a study with a surprisingly good headline.

**Writing these found a real bug in the flagship feature.** The lookahead detector was
blind to a strategy reading exactly *one* bar ahead — it skipped the final bar of each
truncated window, which is the only index where one-bar lookahead shows up. Off-by-one is
the most common form of the bug it exists to catch. Fixed; both real bots re-verified and
their verdicts unchanged.

### Self-test

```bash
python selftest.py
```

Nine known-answer cases. There are exactly two ways a tool like this can be worthless: it
fails to catch a fake, or it refuses to pass a genuine result.

| | |
|---|---|
| **A.** Best of 36 variants fit to pure noise | must score **near zero** |
| **B.** Centred moving average (reads n/2 bars ahead) | must be caught as **BROKEN** |
| **B2.** Full-sample z-score vs. a fixed threshold | must be caught as **BROKEN** |
| **C.** Genuine slow trends, traded at the right horizon | must score **high** |
| **D.** Best 2 of 12 pure-noise assets, shipped | must be caught (**15/100**) |
| **E.** Two *arbitrary* assets from a universe that all trends | must **not** be called selection bias (**100/100**) |
| **F.** Bot that follows its strategy exactly | must **not** alarm |
| **G.** Bot whose exit silently stopped firing | must **ALARM** on the *extra* direction |
| **H.** Equity curve containing a deposit | must report **+2%**, not the raw **+308%** |

Case C is the one that actually constrains the design. It is easy to build a tool that says
no to everything — that sounds rigorous and is exactly as informative as a stopped clock.

Two things worth knowing, both discovered by these tests failing:

**Not every leak is a leak.** Normalising by full-sample mean and standard deviation is
invisible to a `fast > slow` comparison, because an affine transform applied to both sides
cancels. It only bites when compared against a constant. Whether preprocessing leaks
depends on what you do with it downstream — which is why the causality test checks
behaviour rather than inspecting code.

**"No signal" and "wrong instrument for the signal" look identical on an equity curve.** An
earlier version of case C used AR(1) returns and failed. AR(1) momentum lives at lag 1 —
autocorrelation decays as φ^k, so at φ=0.35 it's 0.35 at lag 1 and 0.004 by lag 5. A 20/90
crossover structurally cannot see it, no matter how large φ gets. The signal was real; the
strategy was the wrong instrument. That distinction is now preserved as `momentum_market`
vs `trending_market`.

---

## What this cannot do

**A high score is not a prediction.** It means these particular tests failed to kill the
strategy. That is the only thing it means.

**It deflates by the trials you hand it.** The grid is counted exactly. The choices you
made *before* writing the grid — which market, which side, which timeframe, which of the
eight ideas you tried first — are not, unless you use `run_universe`. The true trial count
is always larger than the one in the report.

**Edge decay is not detectable here, or anywhere.** The Minimum Track Record Length in every
report is the honest number: at the Sharpe ratios retail strategies post, separating decay
from an ordinary bad patch routinely takes decades of live data.

---

## Why this doesn't already exist

The mathematics has been public for a decade — Bailey and López de Prado published the
Deflated Sharpe Ratio and the Probability of Backtest Overfitting in 2014. It lives in
research libraries that assume you already know what a deflated Sharpe is.

The commercial tools are built by companies that sell indicators and strategy generators.
They are structurally incapable of shipping a product whose main output is "your strategy
is nothing."

## Requirements

Python 3.10+, numpy and scipy. Nothing else — no pandas, no build step, no account.

```bash
git clone https://github.com/falsify-quant/falsify && cd falsify
pip install -e ".[dev]"
```

## License

AGPL-3.0-or-later. Free to use, modify and self-host, including commercially.

If you run a modified falsify as a **network service**, the AGPL requires you to offer that
service's users the corresponding source. Using it inside your own trading operation —
however commercial — triggers nothing; only offering it to others as a service does.

## Prior art

Deflated Sharpe Ratio, Probability of Backtest Overfitting and Combinatorially Symmetric
Cross-Validation are due to David Bailey, Jonathan Borwein, Marcos López de Prado and Qiji
Jim Zhu. The mathematics is theirs and has been public since 2014. What is here is the
packaging, the adversarial framing, the causality check, the universe-selection test, the
live monitor and the attestation format.
