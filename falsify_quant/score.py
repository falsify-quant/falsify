"""Turning six findings into one number and one sentence.

The score is a *weighted geometric* mean, not an arithmetic one. That is a deliberate
epistemic choice: a strategy is only as good as its weakest leg, and averaging lets a
strategy hide a fatal flaw behind five comfortable passes. Under a geometric mean, any
component near zero drags the whole score to zero, which is the correct behaviour --
an edge that does not clear its trading costs is worth nothing no matter how stable its
parameter surface is.

Findings marked `fatal` (currently only lookahead) are gates rather than terms. A
strategy that reads the future does not get a score, because there is nothing to score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .prosecute import Finding

__all__ = ["Verdict", "score_findings", "WEIGHTS"]

# How much each surviving test counts. Costs and the three overfitting tests dominate;
# regime spread is a tiebreaker rather than a verdict.
WEIGHTS: dict[str, float] = {
    "costs": 1.3,
    "deflation": 1.2,
    "pbo": 1.2,
    "permutation": 1.2,
    "regime": 0.8,
}

_BANDS: list[tuple[float, str, str]] = [
    (80, "SURVIVED", "Nothing here killed it. That is not proof it works — it is the absence of proof that it does not."),
    (60, "PLAUSIBLE", "Survived the serious tests with room to spare on some and not others. Worth paper trading."),
    (40, "UNPROVEN", "Not obviously broken, not demonstrably real. The sample cannot tell the difference yet."),
    (20, "LIKELY OVERFIT", "Most of what looks like edge is explained by the size of the search."),
    (0, "NO EDGE FOUND", "The backtest is consistent with there being nothing here at all."),
]


@dataclass
class Verdict:
    score: float  # 0-100
    label: str
    summary: str
    findings: list[Finding]
    broken: bool = False
    headline_failure: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def ordered_findings(self) -> list[Finding]:
        """Worst first — the reason you opened the report should be at the top."""
        return sorted(self.findings, key=lambda f: (not f.fatal, f.score))

    @property
    def advice(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for f in self.ordered_findings:
            if f.advice and f.advice not in seen:
                seen.add(f.advice)
                out.append(f.advice)
        return out


def score_findings(findings: list[Finding], meta: dict | None = None) -> Verdict:
    """Collapse the prosecution into a verdict."""
    fatal = [f for f in findings if f.fatal and f.score <= 0.0]
    if fatal:
        return Verdict(
            score=0.0,
            label="BROKEN",
            summary=(
                "This strategy uses information it could not have had at the time. "
                "The backtest does not describe a possible trading history, so none of the "
                "other statistics mean anything. Fix the leak and run again."
            ),
            findings=findings,
            broken=True,
            headline_failure=fatal[0].headline,
            meta=meta or {},
        )

    num = 0.0
    den = 0.0
    for f in findings:
        w = WEIGHTS.get(f.name)
        if w is None:
            continue  # gate-only findings (causality) do not contribute a term
        num += w * np.log(max(f.score, 1e-3))
        den += w

    score = 100.0 * float(np.exp(num / den)) if den else 0.0
    if not np.isfinite(score):
        score = 0.0

    label, summary = next(
        ((lbl, txt) for lo, lbl, txt in _BANDS if score >= lo), _BANDS[-1][1:]
    )

    scored = [f for f in findings if f.name in WEIGHTS]
    worst = min(scored, key=lambda f: f.score) if scored else None

    return Verdict(
        score=score,
        label=label,
        summary=summary,
        findings=findings,
        headline_failure=worst.headline if worst and worst.score < 0.5 else None,
        meta=meta or {},
    )
