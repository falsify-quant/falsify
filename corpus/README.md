# The corpus study

A study built with `falsify`, kept in the repository so its numbers can be reproduced
rather than believed.

Every well-known published trading rule, at **the parameters its own source named**, run
against a fixed universe of instruments and attacked with the same six tests. The
question is not whether these rules can be made to look good — anything can — but whether
the versions people actually trade survive being told how large the search behind them
was.

```bash
python -m corpus.run                    # the whole study, resumable
python -m corpus.aggregate              # writes FINDINGS.md and results.csv
```

## What is in it

| Part | File | What it is |
|---|---|---|
| Strategies | [`strategies/canon.py`](../strategies/canon.py) | 18 published rules, causal implementations, with citations |
| Assets | [`assets.py`](assets.py) | 20 equities and ETFs, 10 crypto pairs, chosen by rule |
| Data | [`cache.py`](cache.py) | frozen snapshot with SHA-256 fingerprints |
| Runner | [`run.py`](run.py) | resumable, deterministic per cell, sqlite |
| Aggregation | [`aggregate.py`](aggregate.py) | aggregate statistics only |

## Method, and why each choice was made

**The shipped parameters are scored, not the best of the grid.** Scoring the grid winner
measures how well the search did, which is easy and uninteresting. 50/200 is what the
press announces, RSI(14) at 30/70 is what the book says, and those are the versions under
examination.

**The grid exists so the deflation has a number to work with.** `falsify` deflates by the
trials it is handed, so each cell sweeps a few dozen combinations around the published
ones. The gap between the grid's best and the published version — the *search premium* —
is recorded for every cell and is one of the study's clearest results.

**Crypto runs at two bar sizes over one window.** The same rule, the same asset, the same
dates, the same cost per unit of turnover, differing only in how often the rule may act.
The `daily-matched` cadence exists because hourly history from a free feed runs out after
a few years; comparing it against a full daily history compares two different markets and
blames the bar size.

**Assets are chosen by constraint, not by taste.** Broad ETFs first, because an index
cannot survive its way into a sample. Single names across sectors, deliberately including
four whose last two decades were bad. No date-range choices: every series takes as much
history as the source will give, ending today.

**Everything is deterministic and fingerprinted.** A cell's seed derives from the study
seed and the cell's own identity, so a resumed run, a partial run and a single cell run
alone all produce the same number. Library version, commit, interpreter, numeric library
versions, grid, shipped parameters and a hash of the price series are stored next to
every verdict.

## What it does not establish

Repeated in `FINDINGS.md`, and worth repeating here.

**The true number of trials is far larger than the grid.** The real search behind
`RSI(14) at 30/70` is fifty years of practitioners trying everything and publishing what
worked. Every score is an **upper bound**.

**Survivorship runs through the whole asset list.** Every instrument still trades.
Companies that went to zero are absent; delisted crypto is absent twice over, because
exchanges remove the pair and its history together. This flatters long-biased rules.

**These are clean-room implementations.** Written from published descriptions against a
truncation test, so the causality results say nothing about the lookahead rate in real
implementations. That is the more interesting question and a different study — one that
needs a corpus of actual code.

**A low score is not proof a rule does not work.** It is a statement that this sample
cannot distinguish it from noise once the search behind it is priced in.

## On naming names

The aggregation publishes **aggregate statistics and per-strategy medians**, and the
per-cell CSV, because the strategies are public knowledge with public citations.

It does not publish verdicts on commercial products, courses or signal services, and the
tooling here is not pointed at them. Prosecuting the canon is scholarship. Publishing "we
tested this seller's system and it does not work" is a different activity with different
consequences, and it is not what this study is for.
