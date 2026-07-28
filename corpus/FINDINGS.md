# The published canon, prosecuted

*Generated 2026-07-28 from `corpus/study.db`, run `main`.*

Eighteen of the best-known published trading rules, each at the parameters its own source named, run against every instrument in a fixed universe and attacked with the same six tests. No optimisation, no parameter selection, no choosing the window afterwards.

The question is not whether these rules can be made to look good -- anything can -- but whether the versions people actually trade survive being told how large the search behind them was.

## What the study found

**890 verdicts.** Each is one published rule, at the parameters its source named, on one instrument, over as much history as a free data feed will give.

- **Median score 3.7 / 100.** Quartiles 0.4 and 17.1; the best cell scored 95.0 and the worst 0.1.
- **8% reached PLAUSIBLE or better** (73 of 890). 80% came back NO EDGE FOUND -- indistinguishable from having tested noise.
- **51% could not clear their own trading costs** at retail rates, before any question of overfitting arises.
- **No cell failed the causality gate.** Expected, and worth stating plainly: these are clean-room implementations written against a truncation test. The lookahead rate in *published implementations* is a different study, and this one does not measure it.

**Read that median with `Home turf` below, not on its own.** It pools every rule over every instrument, including rules run in markets their sources never claimed. Scored where their own authors tested them, the rules with a matching venue here have a median of **56.9**, and 7 of the eighteen have no matching venue in this universe at all. The pooled figure is the right answer to "what happens if you take the canon and point it at whatever you can download", which is what most people do -- and the wrong answer to "does this rule work".

## Verdicts

| Verdict | Cells | Share |
|---|---:|---:|
| SURVIVED | 28 | 3% |
| PLAUSIBLE | 45 | 5% |
| UNPROVEN | 37 | 4% |
| LIKELY OVERFIT | 68 | 8% |
| NO EDGE FOUND | 712 | 80% |

## Which test does the killing

Each check scores 0 to 1. The verdict is their weighted geometric mean, so the column that matters is how often a check lands near zero -- one fatal leg drags the whole score down regardless of the others.

| Check | Median | Failed (< 0.5) | Near-fatal (< 0.1) |
|---|---:|---:|---:|
| Causality (lookahead) | 1.00 | 0% | 0% |
| Cost breakeven | 0.45 | 51% | 46% |
| Deflated Sharpe | 0.16 | 78% | 43% |
| Backtest overfitting (PBO) | 0.46 | 52% | 29% |
| Monte-Carlo permutation | 0.00 | 89% | 82% |
| Regime concentration | 0.43 | 52% | 48% |

## By family

| Family | Cells | Median | 25th | 75th | Best | PLAUSIBLE+ |
|---|---:|---:|---:|---:|---:|---:|
| seasonal | 40 | 19.5 | 12.4 | 22.0 | 92.5 | 12% |
| trend | 350 | 10.1 | 0.7 | 18.0 | 94.3 | 8% |
| reversion | 300 | 1.4 | 0.4 | 17.0 | 95.0 | 12% |
| breakout | 200 | 0.9 | 0.4 | 11.1 | 84.3 | 2% |

## The short leg

| Rules | Distinct | Cells | Median | 75th | PLAUSIBLE+ | Median SR |
|---|---:|---:|---:|---:|---:|---:|
| Long only | 5 | 240 | 18.6 | 37.9 | 20% | 0.32 |
| Long and short | 13 | 650 | 1.0 | 12.6 | 4% | -0.11 |

The same trend idea is in this study twice: `golden-cross` goes long above the crossover and flat below it, `dual-ma` goes short instead of flat. They are not the same strategy with a switch flipped -- different periods, different sources -- so this is an observation rather than an experiment. But the direction of the gap is consistent and the mechanism is not subtle: a symmetric rule spends much of its life short an asset class with a positive expected return, and pays financing to do it.

Worth stating because the symmetric version is what gets taught. The long-only reading is what the press reports and what the tactical-allocation literature actually tested.

## What the search is worth

Every cell sweeps a modest grid around the published parameters -- a few dozen combinations, far fewer than the literature has tried. `search premium` is the annualised Sharpe of the best combination minus the Sharpe of the one the source actually published, on identical data. It is the size of the free lunch available to anyone willing to report their best run.

- Median premium **0.30 Sharpe**; 75th percentile 0.52; largest 2.37.
- **27% of cells** had a best-in-grid at least 0.5 Sharpe above the published version.

That is the entire gap between a strategy that looks publishable and one that does not, and it is available on pure noise. It is also a *lower* bound: the grids here are small, and nobody stops at one grid.

## Choosing the window

The same rules, the same instruments, the same daily bars — scored over all available history, and then over only the stretch the hourly feed also covers.

- Median score **11.5 on the full history** vs **3.7 on the recent window**, across 180 pairs.
- **13 of 19** cases that reached PLAUSIBLE or better on the full history failed to on the shorter one.

The median history lost is only 0.7 years, and that number is misleading. Most of this universe listed recently and loses almost nothing; the long-lived pairs lose 3.6 years at the 90th percentile — and what they lose is the 2021 bull market, which is where a crypto trend rule earned everything it earned.

That split is also the control. In the 18 pairs whose two windows differ by under three months, **17 agree within five points** — so the comparison is measuring the history that was removed, not an error in removing it.

In the 72 pairs that lose a year or more, the median goes **15.0 → 3.3**.

Two effects are tangled here and cannot be separated with this design: the shorter window is also the *more recent* one, and a shorter sample is penalised on its own merits, because deflation is less forgiving when there is less evidence to deflate. Both are real, and both are things a backtester chooses.

This section exists because it changed a conclusion. The cadence comparison below initially looked like a rout — a rule scoring 82 daily and 6 hourly — until the same rule over the same window at daily scored 7.5. Nearly all of that gap was the window, and attributing it to the bar size would have been wrong. The date range is a researcher degree of freedom like any other, and it is the one nobody reports.

## Trading the same rule faster

Crypto is run twice on **the same calendar window**: once on daily bars and once on hourly. Same rule, same parameters, same instrument, same dates, same cost per unit of turnover. The only difference is how often the rule is allowed to act.

The matched window is not a nicety. Hourly history from a free endpoint runs out after a few years, so comparing it against a full daily history compares two different markets and blames the bar size. See the section above for how much damage that does -- it was enough to reverse this study's first conclusion.

- Median score **3.7 daily** vs **0.6 hourly** across 170 matched pairs.
- Median annualised Sharpe **0.04 daily** vs **-0.53 hourly**.
- **63% of rules scored worse hourly.** Median change -0.8 points.

Bar size is not a free parameter. A rule pays its costs per decision, and moving to a bar twenty-four times shorter multiplies the decisions without multiplying the signal.

## By market

| Market | Cadence | Cells | Median | PLAUSIBLE+ | Median years |
|---|---|---:|---:|---:|---:|
| crypto | daily | 180 | 11.5 | 11% | 5.3 |
| crypto | daily-matched | 180 | 3.7 | 6% | 4.6 |
| crypto | hourly | 170 | 0.6 | 0% | 4.6 |
| equity | daily | 360 | 11.6 | 12% | 23.8 |

## Home turf

A rule tested somewhere its author never claimed it worked is not being tested. Every citation was read and labelled with the market **the source itself used**, before any of these scores were looked at:

| Domain in the source | Rules |
|---|---|
| `equity-index` | `golden-cross`, `n-down-days`, `price-vs-ma`, `rsi2-connors`, `turn-of-month`, `vol-target-trend` |
| `futures` | `chandelier`, `donchian`, `keltner-breakout`, `rsi-reversion`, `stochastic`, `williams-r` |
| `cross-asset` | `tsmom` |
| `unstated` | `bollinger-breakout`, `bollinger-reversion`, `dual-ma`, `macd`, `triple-ma` |

For the six rules whose sources tested equity indices, this universe contains the matching venue. On daily bars, scored on index ETFs against everywhere else:

- **On home turf: median 56.9**, 50% reaching PLAUSIBLE or better, across 36 cells.
- Everywhere else: median 18.8, across 144 cells.
- The study-wide median is 3.7.

**That is the single largest effect in this study, and it qualifies the headline number rather than sitting beside it.** A good part of the overall median is rules being scored in markets they never claimed.

**Six of the eighteen rules have no home turf here at all.** `chandelier`, `donchian`, `keltner-breakout`, `rsi-reversion`, `stochastic`, `williams-r` come from commodity and financial futures — Wilder and Lane developed on commodities, the Turtles traded futures, and LeBeau's book has it in the title. **This universe contains no futures.** Their median of 0.7 across 180 daily cells is therefore not a verdict on them; it is a measurement of what happens when you take a futures system to equities and crypto, which is what most retail platforms invite you to do.

Stated plainly because it cuts against the study: a third of the canon here was never given a fair test, and fixing that needs futures data this does not have.

The rules whose sources name no market at all — `bollinger-breakout`, `bollinger-reversion`, `dual-ma`, `macd`, `triple-ma` — sit at a median of 3.5. There is nowhere to move them to. A rule that never said where it worked cannot be defended on the grounds that it was being tested in the wrong place.

## By strategy

Every rule at the parameters its own source published, across every instrument and cadence it was run on.

| Strategy | Family | Cells | Median | Best cell | PLAUSIBLE+ | Median SR |
|---|---|---:|---:|---:|---:|---:|
| `turn-of-month` | seasonal | 40 | 19.5 | 92.5 | 12% | 0.34 |
| `golden-cross` | trend | 50 | 19.2 | 92.7 | 10% | 0.33 |
| `n-down-days` | reversion | 50 | 19.1 | 95.0 | 36% | 0.28 |
| `price-vs-ma` | trend | 50 | 18.4 | 94.3 | 18% | 0.28 |
| `rsi2-connors` | reversion | 50 | 17.1 | 93.4 | 20% | 0.37 |
| `tsmom` | trend | 50 | 9.5 | 53.6 | 0% | 0.04 |
| `triple-ma` | trend | 50 | 7.5 | 87.1 | 10% | 0.03 |
| `dual-ma` | trend | 50 | 4.5 | 75.5 | 4% | 0.10 |
| `vol-target-trend` | trend | 50 | 4.1 | 69.5 | 6% | 0.03 |
| `chandelier` | breakout | 50 | 1.5 | 73.6 | 2% | -0.13 |
| `keltner-breakout` | breakout | 50 | 1.5 | 68.6 | 2% | -0.25 |
| `donchian` | breakout | 50 | 0.8 | 84.3 | 6% | -0.12 |
| `macd` | trend | 50 | 0.7 | 82.8 | 8% | -0.12 |
| `bollinger-reversion` | reversion | 50 | 0.6 | 68.2 | 4% | -0.31 |
| `rsi-reversion` | reversion | 50 | 0.6 | 54.3 | 0% | -0.18 |
| `stochastic` | reversion | 50 | 0.6 | 69.8 | 2% | -0.34 |
| `bollinger-breakout` | breakout | 50 | 0.5 | 52.4 | 0% | -0.28 |
| `williams-r` | reversion | 50 | 0.5 | 73.6 | 8% | -0.28 |

## What this does not establish

**The true number of trials is far larger than the grid.** Each cell deflates by a few dozen combinations. The real search behind `RSI(14) at 30/70` is fifty years of practitioners trying everything and publishing what worked. Every score here is therefore an **upper bound**.

**Survivorship runs through the whole asset list.** Every instrument still trades. Companies that went to zero are absent; delisted crypto is absent twice over, because exchanges remove the pair and the history with it. This flatters long-biased rules. Index ETFs mitigate it and nothing free eliminates it.

**These are clean-room implementations, not the sources' code.** They were written from published descriptions against a truncation test, so the causality results say nothing about the lookahead rate in real implementations -- which is the more interesting question and a different study.

**A low score is not proof a rule does not work.** It is a statement that this sample cannot distinguish it from noise once the search that produced it is priced in. That is a weaker claim, and it is the only one the arithmetic supports.

**Costs are modelled, not realised.** Retail rates on liquid instruments, with no market impact, no partial fills and no borrow that ever goes special. Real execution is worse, so this errs toward flattering.

## Reproducing it

```bash
python -m corpus.run --out corpus/study.db
python -m corpus.aggregate
```

- falsify `0.1.0`, commit `a7ed306`
- Python 3.12.7, numpy 2.5.1, scipy 1.18.0
- 100 permutations per cell, study seed `20260727`
- Per-cell seeds derive from the study seed and the cell's identity, so a single cell run alone reproduces its number exactly.

Data fingerprints (SHA-256 over closes and timestamps, first 16 hex) for all 50 series are in `results.csv`. Free feeds revise history; if your fingerprints differ, your data differs, and the numbers should be expected to.
