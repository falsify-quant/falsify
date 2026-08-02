# Changelog

## 0.1.3

Bug fixes and onboarding. Nothing here changes what a verdict means; several things
change whether you ever see one.

### Failures that looked like the tool was broken

- **The GUI could hang forever.** A strategy file missing its `GRID` — the likeliest
  thing to be wrong with a ported strategy — left the page spinning with no error and
  no result. Every user error the CLI recognises it reports by raising `SystemExit`,
  which is a `BaseException` and slipped past the worker's `except Exception`, so the
  thread died without ever setting a state.
- **A mistyped strategy filename printed a traceback.** Eight frames of importlib
  internals ending in `FileNotFoundError`. It now names the file, lists the `.py` files
  actually in that directory, and points at `--new`. Same fix for `--csv`, which had the
  same bug in the sibling function.
- **A syntax error, a missing dependency and a directory** were also tracebacks and are
  now sentences. The syntax error keeps Python's own file/line/caret, which is the useful
  part, and drops only the importlib frames above it.
- **`falsify-quant-gui` and `falsify-quant-watch` could die while printing.** Only the
  main CLI reconfigured the console for UTF-8. The line the GUI died on was the one
  telling you not to expose the port, printed after the socket was already listening.

### Attestation

- **A document could be reported `TAMPERED` because of the reader's codepage.** The file
  was written with `ensure_ascii=False`, so a single em-dash in a finding headline — which
  the report emits routinely — read as three different characters under a Windows default
  of cp1252, and a different body hashes differently. Attestations are now pure ASCII and
  verify identically under utf-8, cp1252, latin-1 and ascii. Documents written by earlier
  versions still verify; the hash is taken over parsed values, so escaping cannot change it.
- **Attesting a strategy file that was not UTF-8 crashed.** CPython imports a module whose
  comments are cp1252 without complaint, so the strategy ran, the whole analysis ran, and
  then it died on a quotation mark in a comment. Fingerprints are taken over bytes now,
  and agree with the old values on anything valid UTF-8.

### `falsify-quant-watch`

- **One failing sink could end the daemon.** A full log volume raised before the webhook
  was ever called, so the local convenience took out the delivery that reaches a person,
  skipped the state save, and — `run_once` being unguarded in the loop — stopped the
  watching. Each sink now fails alone.
- **A full state volume could end it too**, after exactly one cycle. It now carries on
  without a memory and says why; alerts repeat until it is fixed, which is the signal you
  want, where exiting produces silence.
- **A webhook URL with no scheme now fails at `--check`** instead of dropping every batch
  forever while the operator believes they are covered.

### Correctness

- **Stale cache could silently change a verdict.** `get()` returned a cached series
  whenever the file existed, ignoring the `fetched_utc`, window and fingerprint the sidecar
  had been writing all along. Two machines holding different data under the same filename
  scored the same strategy 41/100 and 10.7/100.
- **A CSV with no readable date column no longer guesses the calendar in silence.** Every
  annualised figure is scaled by `bars_per_year`; falling back to the `--market` preset
  without saying so put the headline out by a constant factor with nothing looking wrong.
  Common export date formats are now read, and ambiguous ones are rejected rather than
  used, if they do not come out strictly increasing.
- **The permutation test says why it could not run.** Synthetic paths carry a close series
  only, so a strategy reading `bars.volume` or `bars.high` fails on every null path. It is
  one of the six weighted checks and it was moving the score while saying nothing.

### Getting started

- `--new mystrategy.py` writes a working strategy instead of leaving you a blank file and
  a spec to reverse-engineer.
- `from_signals(entries, exits)` and `from_positions(...)` convert the event model almost
  everyone already has — vectorbt, backtrader, Pine — into the weight series this wants.
  Last event wins, forward-fill only, so it is causal by construction.
- The CSV loader reads what Yahoo and Nasdaq actually export, including `Adj Close`,
  `Close/Last` and prices written as `$1,234.56`.
- Failures during a sweep name which of the four common mistakes you made rather than
  reporting "check the strategy signature" for all of them.
- The GUI derives its market list from the presets, so `futures` is reachable, and can
  score a local CSV instead of only fetching.
- The report's equity curve carries a scale, and the detail tables say which return they
  are reporting — the summed and compounded figures differ by volatility drag and the
  document used to show both without labelling either.

### Research

- The corpus study scores each rule on the market its source actually tested. Six rules
  whose sources used equity indices median 56.9 there against a pooled 3.7.
- Seven futures contracts added, 1,016 cells in total, closing the gap that finding named.

### Internal

- pandas is a test dependency now, not a runtime one. The porting docs promise a pandas
  Series works wherever an array does; that is checked rather than assumed.
- Calibration is measured. The permutation p-value is uniform under the null across 1,500
  searches on structureless data, and deflation takes the best of 100 worthless strategies
  from a 99% false-positive rate undeflated to 0%, while still rising monotonically with a
  real edge.

## 0.1.2 and earlier

See the release history on GitHub.
