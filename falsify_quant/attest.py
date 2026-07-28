"""Attested verdicts: a report that cannot be quietly edited after the fact.

A falsify verdict is more useful as a *credential* than as a tool. Nobody pays to be told
their own strategy is nothing; people do care what a stranger's strategy is worth before
wiring money at it. That only works if the stranger cannot doctor the report, and if the
recipient can check without taking anyone's word.

So: every attestation is a canonical JSON body plus a SHA-256 over it, and verification
does not merely re-hash. It **recomputes the score from the findings** using the published
weighting. Editing a finding changes the hash; editing the score alone survives the hash
if you recompute it, and then fails the arithmetic. Both have to be forged consistently,
and doing so requires forging the individual test statistics, which is the point where
forgery becomes as much work as being honest.

## What this proves, and what it does not

**Proves, on its own:**

- the body has not changed since the hash was taken;
- the headline score follows from the individual findings by the documented weights;
- everything needed to re-run the analysis is recorded -- library version, seed, grid,
  the parameters under examination, a fingerprint of the price series, the cost model.

**Does not prove, and cannot:**

- **When it was made.** `created_utc` is self-reported. Anyone can put any date in it.
  This is the one that matters, because a verdict is only impressive if it predates the
  performance it is being used to justify. Fixing it requires an *anchor*: publish the
  hash somewhere you do not control -- a commit in a public repository, a blockchain
  timestamp, anything with an independent clock -- and record the reference. `verify`
  reports loudly when there is no anchor rather than letting a bare timestamp pass for
  evidence.
- **That the strategy is what its author says.** The source is fingerprinted, not
  embedded, so an author can attest without publishing their code. That means a verifier
  learns "this code, whatever it is, produced this" -- and can confirm the match later if
  the code is ever handed over.
- **That the data is real.** The price series is fingerprinted. A verifier who fetches the
  same series from the same source and gets the same fingerprint has checked it; one who
  does not, has not.

The honest summary is that this makes tampering detectable and pre-commitment possible.
It is not a proof of skill, and a `verify` that passes says only that the arithmetic in
front of you is the arithmetic that was done.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .prosecute import Finding
from .score import WEIGHTS, Verdict, score_findings

__all__ = [
    "Attestation",
    "Check",
    "VerifyResult",
    "attest",
    "verify",
    "canonical_bytes",
    "content_hash",
    "fingerprint_array",
    "fingerprint_text",
    "write_attestation",
    "read_attestation",
    "ANCHOR_HELP",
]

SCHEMA_VERSION = 1

ANCHOR_HELP = """\
An attestation dates itself, which is worth nothing on its own -- the author picks the
date. To make the date mean something, publish the content hash somewhere with a clock
you do not control, then record where:

  git       commit the hash to a public repository; the commit is timestamped by the host
  release   put it in a tagged release or a published package
  chain     an OpenTimestamps proof, or any blockchain anchor
  post      anywhere public, dated, and not editable after the fact

Then re-attest with the reference:

  anchor={"kind": "git", "ref": "https://github.com/you/repo/commit/<sha>"}

The order matters. Publish the hash *first*, then let time pass, then show the results.
A hash published after the fact proves only that you can use a hash function.
"""


# --------------------------------------------------------------------------------------
# Canonical form
# --------------------------------------------------------------------------------------


def _plain(obj: Any) -> Any:
    """Reduce to JSON-native types, deterministically.

    Non-finite numbers become null: a NaN written literally is valid to Python's own
    parser and invalid to every other one, which is a poor property for a document whose
    purpose is being checked by someone else's tooling.

    Negative zero is normalised to zero. It compares equal to zero, hashes differently as
    text, and no statistic here distinguishes them -- so leaving it in means two identical
    analyses can produce two different hashes.
    """
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_plain(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if not np.isfinite(f):
            return None
        return 0.0 if f == 0.0 else f
    if isinstance(obj, (int, str)) or obj is None:
        return obj
    return str(obj)


def canonical_bytes(body: dict) -> bytes:
    """The exact bytes the hash is taken over.

    Sorted keys, no incidental whitespace, UTF-8. Floats go through `repr`, which since
    Python 3.1 emits the shortest string that round-trips to the same IEEE-754 double --
    identical on every platform that has IEEE-754 doubles, which is every platform this
    runs on.
    """
    return json.dumps(
        _plain(body), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def content_hash(body: dict) -> str:
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def fingerprint_array(x) -> str:
    """A 32-hex identity for a numeric series, stable across platforms."""
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()[:32]


def fingerprint_text(text: str) -> str:
    """A 32-hex identity for source code, normalised for line endings.

    Without the normalisation the same file attests to two different values depending on
    whether it last touched Windows, which would make the check useless exactly when
    somebody is trying to confirm a match across machines.
    """
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:32]


# --------------------------------------------------------------------------------------
# Building one
# --------------------------------------------------------------------------------------


@dataclass
class Attestation:
    body: dict
    content_hash: str

    @property
    def created_utc(self) -> str:
        return self.body.get("created_utc", "")

    @property
    def score(self) -> float:
        return float(self.body.get("verdict", {}).get("score", float("nan")))

    @property
    def label(self) -> str:
        return str(self.body.get("verdict", {}).get("label", ""))

    @property
    def anchor(self) -> dict | None:
        return self.body.get("anchor")

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            {"falsify_attestation": SCHEMA_VERSION,
             "content_hash": self.content_hash,
             "body": _plain(self.body)},
            indent=indent, ensure_ascii=False, allow_nan=False,
        ) + "\n"


def attest(
    verdict: Verdict,
    *,
    strategy_source: str | Path | None = None,
    anchor: dict | None = None,
    note: str = "",
    created_utc: str | None = None,
) -> Attestation:
    """Turn a verdict into a tamper-evident document.

    `strategy_source` is fingerprinted, never embedded -- attesting should not require
    publishing the strategy. `anchor` records where the hash was published; see
    `ANCHOR_HELP` for why a document without one cannot vouch for its own date.

    `created_utc` is settable so that regenerating an attestation from the same analysis
    reproduces the same hash. It is not a way to backdate anything: the date is inside the
    hashed body precisely so that an anchor binds a specific claimed date, and a date
    nobody anchored was never evidence in the first place.
    """
    import falsify_quant

    meta = dict(verdict.meta)
    equity = meta.pop("equity", None)
    ts = meta.pop("ts", None)

    if isinstance(strategy_source, Path):
        strategy_source = strategy_source.read_text(encoding="utf-8")

    body: dict = {
        "schema": SCHEMA_VERSION,
        "created_utc": created_utc or datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "falsify_version": falsify_quant.__version__,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "subject": {
            "symbol": meta.get("symbol"),
            "market": meta.get("market"),
            "asset_class": meta.get("asset_class"),
            "bars": meta.get("bars"),
            "bars_per_year": meta.get("bars_per_year"),
            "years": meta.get("years"),
            "params": meta.get("params"),
            "grid": meta.get("grid"),
            "n_trials": meta.get("n_trials"),
        },
        "evidence": {
            "sharpe_annual": meta.get("sharpe_annual"),
            "equity_fingerprint": fingerprint_array(equity) if equity is not None else None,
            "series_fingerprint": fingerprint_array(ts) if ts is not None else None,
            "strategy_fingerprint": (fingerprint_text(strategy_source)
                                     if strategy_source else None),
        },
        "verdict": {
            "score": float(verdict.score),
            "label": verdict.label,
            "broken": bool(verdict.broken),
            "headline_failure": verdict.headline_failure,
        },
        # Sorted by name so the document does not depend on the order the checks ran in.
        "findings": [
            {"name": f.name, "title": f.title, "score": float(f.score),
             "fatal": bool(f.fatal), "headline": f.headline, "detail": f.detail}
            for f in sorted(verdict.findings, key=lambda f: f.name)
        ],
        "weights": dict(sorted(WEIGHTS.items())),
        "anchor": anchor,
        "note": note,
    }
    # Canonicalise at construction, not at write time. Findings carry numpy arrays and
    # numpy scalars in their `detail`, so a body that keeps them is a body that cannot be
    # compared, copied or round-tripped -- and one that differs from the bytes actually
    # hashed. The document in memory should be the document on disk.
    body = _plain(body)
    return Attestation(body=body, content_hash=content_hash(body))


# --------------------------------------------------------------------------------------
# Checking one
# --------------------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    severity: str = "error"  # error | warning | info


@dataclass
class VerifyResult:
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks if c.severity == "error")

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if not c.ok and c.severity == "warning"]

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok and c.severity == "error"]

    def summary(self) -> str:
        if not self.ok:
            return f"TAMPERED — {len(self.failures)} check(s) failed"
        if self.warnings:
            return f"INTACT, {len(self.warnings)} caveat(s)"
        return "INTACT"


def verify(att: Attestation) -> VerifyResult:
    """Check an attestation against itself and against the published arithmetic."""
    import falsify_quant

    out = VerifyResult()
    body = att.body

    recomputed = content_hash(body)
    out.checks.append(Check(
        "content hash", recomputed == att.content_hash,
        "the body matches its hash" if recomputed == att.content_hash
        else f"hash mismatch: body hashes to {recomputed}, document claims "
             f"{att.content_hash}",
    ))

    findings = body.get("findings") or []
    if not findings:
        out.checks.append(Check("findings", False, "no findings recorded"))
        return out

    # Recompute the headline from the parts. A doctored score that was re-hashed still
    # has to survive this, and surviving it means forging the individual test statistics.
    try:
        rebuilt = score_findings(
            [Finding(name=f["name"], title=f.get("title", f["name"]),
                     score=float(f["score"]), headline=f.get("headline", ""),
                     fatal=bool(f.get("fatal", False)))
             for f in findings]
        )
        claimed = float(body["verdict"]["score"])
        agrees = abs(rebuilt.score - claimed) < 0.05
        out.checks.append(Check(
            "score arithmetic", agrees,
            f"{claimed:.2f} follows from the findings" if agrees
            else f"the findings imply {rebuilt.score:.2f}, the document claims "
                 f"{claimed:.2f}",
        ))
        label_ok = rebuilt.label == body["verdict"]["label"]
        out.checks.append(Check(
            "label", label_ok,
            f"{rebuilt.label} is the band for {claimed:.2f}" if label_ok
            else f"score {claimed:.2f} belongs in band {rebuilt.label}, document says "
                 f"{body['verdict']['label']}",
        ))
    except (KeyError, TypeError, ValueError) as exc:
        out.checks.append(Check("score arithmetic", False, f"could not recompute: {exc}"))

    # Deleting the check that failed is the cheapest forgery available: recompute the
    # score over what is left and the arithmetic is consistent again. So the set of
    # checks is itself part of what gets verified.
    present = {f.get("name") for f in findings}
    missing = sorted(set(WEIGHTS) - present)
    out.checks.append(Check(
        "coverage", not missing,
        f"all {len(WEIGHTS)} scored checks are present" if not missing
        else f"missing finding(s): {', '.join(missing)}. A score computed without them "
             f"is not comparable to one that includes them.",
    ))
    if "causality" not in present:
        out.checks.append(Check(
            "causality gate", False,
            "the lookahead gate is absent; no verdict is meaningful without it",
        ))

    stored_w = body.get("weights") or {}
    same_w = stored_w == dict(WEIGHTS)
    out.checks.append(Check(
        "weights", same_w,
        "scored with the weights this version publishes" if same_w
        else f"attested under different weights than this falsify uses: "
             f"{stored_w} vs {dict(WEIGHTS)}",
        severity="warning",
    ))

    version = body.get("falsify_version")
    same_v = version == falsify_quant.__version__
    out.checks.append(Check(
        "version", same_v,
        f"produced by falsify {version}" if same_v
        else f"produced by falsify {version}; verifying with {falsify_quant.__version__}",
        severity="warning" if not same_v else "info",
    ))

    anchor = body.get("anchor")
    out.checks.append(Check(
        "anchor", bool(anchor and anchor.get("ref")),
        f"hash published at {anchor['ref']} ({anchor.get('kind', 'unknown')}); "
        f"confirm that reference independently -- this tool cannot"
        if anchor and anchor.get("ref")
        else f"no anchor. The date {body.get('created_utc', '?')} is self-reported and "
             f"proves nothing about when this was produced.",
        severity="warning",
    ))

    return out


# --------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------


def write_attestation(att: Attestation, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(att.to_json(), encoding="utf-8")
    return path


def read_attestation(path: str | Path) -> Attestation:
    """Load an attestation, keeping the hash *as written* rather than recomputing it.

    Recomputing on load would make every document verify, which is the one thing this
    must never do.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if "body" not in raw or "content_hash" not in raw:
        raise ValueError(f"{path} is not a falsify attestation")
    version = raw.get("falsify_attestation")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{path} uses attestation schema {version}, this falsify speaks "
            f"{SCHEMA_VERSION}"
        )
    return Attestation(body=raw["body"], content_hash=raw["content_hash"])
