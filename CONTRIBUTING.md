# Contributing

## The one rule

**Every change must keep `python selftest.py` at 9/9.**

That suite is the only thing standing between this project and uselessness. There are
exactly two ways a tool like this fails, and it fails silently in both directions:

1. It misses a fake — a strategy fit to noise scores well.
2. It refuses to pass a real one — everything is "overfit", which sounds rigorous and is
   exactly as informative as a stopped clock.

Cases A, B and B2 guard the first. Case C guards the second, and it is the one that
actually constrains the design. Making the tool stricter is easy; keeping it *calibrated*
is the work.

## Setup

```bash
git clone https://github.com/falsify/falsify
cd falsify
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

pytest                # 57 unit tests: is the arithmetic right?
python selftest.py    # 9 known-answer cases: does it measure anything?
```

Both must pass. They answer different questions and neither substitutes for the other —
a wrong constant in the deflation formula would leave all nine known-answer cases green
while making every reported number wrong.

## Adding a test to the prosecution

New checks are welcome. They need three things:

**A known-answer case in both directions.** Construct a situation where your test must
fire, and one where it must stay quiet. Add both to `selftest.py`. A test that only ever
says "bad" is not evidence of anything.

**A `Finding` with a score in [0, 1] and a headline someone will actually read.** The
headline is the product. `"PBO 0.54"` is not a finding; `"Choosing the best backtest is
no better than choosing at random"` is.

**Fail-closed behaviour.** If your test cannot be computed, it scores 0, not 1. In a tool
whose entire job is scepticism, "uncomputable" must never read as "passed".

Name it `check_*`, never `test_*` — pytest collects `test_*` from imported modules, which
breaks the test suite of anyone who imports falsify.

## Things that will be rejected

**Anything that makes a strategy score better.** There is no optimiser here and there
will not be one. If a change's effect is that more strategies pass, it needs to be
justified as a correction to a false positive, with the case that proves it.

**Statistical claims without a reference.** The unit tests check the maths against things
true independently of the implementation — Monte Carlo simulation of the quantity being
approximated, closed forms, round trips between separately written formulas. New maths
needs the same.

**Silent tolerance widening.** If a threshold moves, say why in the commit and add the
case that motivated it.

## Reporting a false positive

The most valuable bug report is a strategy that genuinely works which falsify calls
overfit. If you have one, open an issue with a reproducible construction — synthetic data
is fine and preferred, since it needs no explanation of what you trade.

The second most valuable is a strategy with a real lookahead bug that the causality check
misses. That check has been wrong before: it was once blind to a strategy reading exactly
one bar ahead, which is the most common form of the bug it exists to catch.

## Style

Match the surrounding code. Comments explain *why*, especially where a choice is not the
obvious one — most of the non-obvious choices here exist because the obvious one was
tried first and produced a wrong answer on real data. Those comments are load-bearing;
please keep them accurate rather than tidy.
