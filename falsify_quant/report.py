"""Single-file HTML verdict. No CDN, no build step, no network -- open it anywhere.

The report is deliberately blunt at the top and detailed underneath. The number and the
one damning sentence are what someone reads; everything below is what they need when
they want to argue with it.
"""

from __future__ import annotations

import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from .score import Verdict

__all__ = ["render_report", "write_report"]


def _hue(score: float) -> str:
    if score >= 0.8:
        return "ok"
    if score >= 0.5:
        return "warn"
    return "bad"


def _fmt(x: object, digits: int = 3) -> str:
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        if math.isinf(x):
            return "∞"
        if math.isnan(x):
            return "—"
        if abs(x) >= 10000:
            return f"{x:,.0f}"
        return f"{x:,.{digits}g}"
    if isinstance(x, (list, dict)):
        return html.escape(json.dumps(x, default=str)[:200])
    return html.escape(str(x))


def _curve_scale(equity: list[float]) -> str:
    """Total return and worst drawdown, next to the chart heading.

    Without these the curve is shape with no magnitude: the reader cannot tell
    +2% from +200%, and both figures existed only inside a collapsed toggle under
    the *costs* finding, which is not where anyone looks for them. This does not
    promote return to the headline -- the score stays the headline -- it just
    makes the chart that is already on the page readable.
    """
    if not equity:
        return ""
    # equity is (1+net).cumprod(), so it starts from an implied 1.0 BEFORE the
    # first bar -- the endpoint alone is the compounded return. Dividing by
    # equity[0] would silently discard the first bar's contribution.
    #
    # Said "compounded" rather than "total" on purpose: the detail tables report
    # `net_return` as a plain SUM of bar returns, and on this sample those differ
    # by 3.8 points (15.8% summed against 12.0% compounded, the gap being
    # volatility drag). Two different quantities under one word, in a report
    # about not overstating results, is exactly the kind of thing this tool
    # exists to catch elsewhere.
    total = equity[-1] - 1.0

    peak, worst = equity[0], 0.0
    for v in equity:
        peak = max(peak, v)
        if peak:
            worst = min(worst, v / peak - 1.0)

    return (f'<span class="scale">{total:+.1%} compounded &middot; '
            f'{worst:.1%} worst drawdown</span>')


def _sparkline(values: list[float], width: int = 720, height: int = 160) -> str:
    """Equity curve as an inline SVG path, downsampled to keep the file small."""
    if not values:
        return ""
    v = list(values)
    if len(v) > 1200:
        step = len(v) / 1200
        v = [v[int(i * step)] for i in range(1200)]

    lo, hi = min(v), max(v)
    span = (hi - lo) or 1.0
    n = len(v)
    pad = 6

    def pt(i: int, val: float) -> tuple[float, float]:
        x = pad + (width - 2 * pad) * (i / max(1, n - 1))
        y = height - pad - (height - 2 * pad) * ((val - lo) / span)
        return x, y

    pts = [pt(i, val) for i, val in enumerate(v)]
    path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = path + f" L {pts[-1][0]:.1f},{height - pad} L {pts[0][0]:.1f},{height - pad} Z"

    # Where equity crosses its starting value, for a reference line.
    y0 = pt(0, 1.0)[1] if lo <= 1.0 <= hi else None
    base = (
        f'<line x1="{pad}" y1="{y0:.1f}" x2="{width - pad}" y2="{y0:.1f}" class="axis"/>'
        if y0 is not None else ""
    )

    return (
        f'<svg viewBox="0 0 {width} {height}" class="spark" preserveAspectRatio="none" '
        f'role="img" aria-label="equity curve">'
        f'<path d="{area}" class="sparkfill"/>{base}<path d="{path}" class="sparkline"/></svg>'
    )


def _score_bar(score: float) -> str:
    pct = max(0.0, min(1.0, score)) * 100
    return (
        f'<div class="bar"><div class="bar-fill {_hue(score)}" style="width:{pct:.1f}%"></div></div>'
    )


_DETAIL_LABELS = {
    "edge_per_turnover_bps": "Edge per unit turnover",
    "cost_per_turnover_bps": "Cost per unit turnover",
    "margin": "Cost margin (edge ÷ cost)",
    "breakeven_cost_bps": "Breakeven cost",
    "turnover_per_year": "Turnover per year",
    "n_trades": "Trades",
    "gross_return": "Gross return (summed)",
    "net_return": "Net return (summed)",
    "cost_drag": "Paid in costs",
    "dsr": "Deflated Sharpe (probability real)",
    "n_trials": "Variants searched",
    "observed_sharpe_annual": "Reported Sharpe (annual)",
    "deflated_benchmark_annual": "Sharpe expected from search alone",
    "min_track_record_bars": "Bars of live trading needed to prove it",
    "max_drawdown": "Max drawdown",
    "pbo": "Probability of backtest overfitting",
    "oos_profitable_rate": "Splits where the pick made money OOS",
    "median_oos_sharpe_annual": "Median out-of-sample Sharpe",
    "n_splits": "Train/test splits",
    "n_variants": "Variants ranked",
    "mean_oos_percentile": "Mean out-of-sample percentile",
    "p_value": "p-value against noise",
    "n_runs": "Synthetic histories",
    "real_best_sharpe_annual": "Best on real data (annual)",
    "null_best_median_annual": "Median best on noise (annual)",
    "null_best_p95_annual": "95th pct best on noise (annual)",
    "consistency": "Fraction of periods profitable",
    "top1pct_pnl_share": "P&L from best 1% of bars",
    "max_relative_change": "Largest changed decision",
    "first_leak_bar": "First leaking bar",
}

_SKIP_DETAILS = {"curve", "chunks", "cuts_tested", "cuts_leaking", "method", "block_size",
                 "trial_sharpe_variance", "n_blocks", "n_periods", "periods_profitable"}


def _detail_table(detail: dict) -> str:
    rows = []
    for k, v in detail.items():
        if k in _SKIP_DETAILS:
            continue
        label = _DETAIL_LABELS.get(k, k.replace("_", " ").capitalize())
        suffix = ""
        if k.endswith("_bps"):
            suffix = " bps"
        elif k in {"dsr", "pbo", "consistency", "top1pct_pnl_share", "p_value",
                   "max_drawdown", "mean_oos_percentile", "max_relative_change",
                   "oos_profitable_rate"}:
            if isinstance(v, float) and math.isfinite(v):
                rows.append(f"<tr><td>{html.escape(label)}</td><td>{v:.3f}</td></tr>")
                continue
        rows.append(f"<tr><td>{html.escape(label)}</td><td>{_fmt(v)}{suffix}</td></tr>")
    return "<table class='detail'>" + "".join(rows) + "</table>" if rows else ""


def _cost_curve(detail: dict) -> str:
    curve = detail.get("curve") or []
    if not curve:
        return ""
    rows = "".join(
        f"<tr><td>{c['multiple']:g}×</td><td>{c['sharpe']:.2f}</td>"
        f"<td>{c['total_return']*100:+.1f}%</td></tr>"
        for c in curve
    )
    return (
        "<table class='detail'><thead><tr><th>Cost level</th><th>Sharpe</th>"
        f"<th>Total return</th></tr></thead>{rows}</table>"
    )


def _regime_table(detail: dict) -> str:
    chunks = detail.get("chunks") or []
    if not chunks:
        return ""
    rows = "".join(
        f"<tr><td>Period {c['index'] + 1}</td><td>{c['sharpe']:.2f}</td>"
        f"<td class='{'pos' if c['total_return'] > 0 else 'neg'}'>{c['total_return']*100:+.1f}%</td></tr>"
        for c in chunks
    )
    return (
        "<table class='detail'><thead><tr><th></th><th>Sharpe</th><th>Return</th></tr></thead>"
        f"{rows}</table>"
    )


CSS = """
:root{--bg:#fbfbfa;--fg:#1a1a19;--dim:#6b6b66;--line:#e3e3df;--card:#fff;
--ok:#1a7f4f;--warn:#b06a00;--bad:#c0392b;--accent:#1a1a19}
@media(prefers-color-scheme:dark){:root{--bg:#131313;--fg:#eceae6;--dim:#9a978f;
--line:#2b2b2a;--card:#1a1a19;--ok:#4ec98a;--warn:#e0a03a;--bad:#f0685a;--accent:#eceae6}}
:root[data-theme=dark]{--bg:#131313;--fg:#eceae6;--dim:#9a978f;--line:#2b2b2a;
--card:#1a1a19;--ok:#4ec98a;--warn:#e0a03a;--bad:#f0685a;--accent:#eceae6}
:root[data-theme=light]{--bg:#fbfbfa;--fg:#1a1a19;--dim:#6b6b66;--line:#e3e3df;
--card:#fff;--ok:#1a7f4f;--warn:#b06a00;--bad:#c0392b;--accent:#1a1a19}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Inter,system-ui,sans-serif;
-webkit-font-smoothing:antialiased}
.wrap{max-width:820px;margin:0 auto;padding:56px 24px 96px}
.brand{font:600 13px ui-monospace,"SF Mono",Menlo,Consolas,monospace;letter-spacing:.14em;
text-transform:uppercase;color:var(--dim);margin:0 0 40px}
.verdict{display:flex;gap:28px;align-items:baseline;flex-wrap:wrap;margin-bottom:8px}
.score{font:700 76px/1 ui-monospace,"SF Mono",Menlo,Consolas,monospace;letter-spacing:-.03em}
.score.ok{color:var(--ok)}.score.warn{color:var(--warn)}.score.bad{color:var(--bad)}
.score small{font-size:24px;color:var(--dim);font-weight:400}
.label{font:600 26px/1.2 inherit;letter-spacing:-.01em}
.summary{color:var(--dim);max-width:60ch;margin:12px 0 0}
.damning{margin:28px 0 0;padding:16px 18px;border-left:3px solid var(--bad);
background:color-mix(in srgb,var(--bad) 7%,transparent);border-radius:0 6px 6px 0}
.meta{display:flex;flex-wrap:wrap;gap:0 32px;margin:36px 0 0;padding-top:24px;
border-top:1px solid var(--line);font-size:13px}
.meta div{margin-bottom:10px}.meta dt{color:var(--dim);margin-bottom:2px}
.meta dd{margin:0;font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
h2{font:600 13px ui-monospace,"SF Mono",Menlo,Consolas,monospace;letter-spacing:.14em;
text-transform:uppercase;color:var(--dim);margin:56px 0 16px}
/* Sits on the curve's heading. Deliberately quiet -- the score is the headline,
   this only stops the chart being a shape with no magnitude. Wraps under the
   heading on a narrow screen rather than pushing it out of view. */
.scale{float:right;text-transform:none;letter-spacing:.02em;color:var(--dim);
font-weight:400}
@media(max-width:520px){.scale{float:none;display:block;margin-top:6px}}
.spark{width:100%;height:160px;display:block}
.sparkline{fill:none;stroke:var(--accent);stroke-width:1.5;vector-effect:non-scaling-stroke}
.sparkfill{fill:color-mix(in srgb,var(--accent) 8%,transparent);stroke:none}
.axis{stroke:var(--dim);stroke-width:1;stroke-dasharray:3 3;opacity:.5;vector-effect:non-scaling-stroke}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:20px 22px;margin-bottom:14px}
.card-head{display:flex;align-items:center;gap:12px;margin-bottom:10px}
.card-head h3{margin:0;font:600 16px inherit;flex:1}
.pill{font:600 11px ui-monospace,Menlo,Consolas,monospace;letter-spacing:.08em;
padding:3px 8px;border-radius:99px;text-transform:uppercase}
.pill.ok{background:color-mix(in srgb,var(--ok) 15%,transparent);color:var(--ok)}
.pill.warn{background:color-mix(in srgb,var(--warn) 15%,transparent);color:var(--warn)}
.pill.bad{background:color-mix(in srgb,var(--bad) 15%,transparent);color:var(--bad)}
.bar{height:4px;background:var(--line);border-radius:99px;overflow:hidden;margin:0 0 14px}
.bar-fill{height:100%;border-radius:99px}
.bar-fill.ok{background:var(--ok)}.bar-fill.warn{background:var(--warn)}
.bar-fill.bad{background:var(--bad)}
.headline{margin:0 0 12px}
.advice{margin:12px 0 0;padding:12px 14px;background:color-mix(in srgb,var(--fg) 4%,transparent);
border-radius:6px;font-size:14px;color:var(--dim)}
.advice b{color:var(--fg);font-weight:600}
details{margin-top:12px}summary{cursor:pointer;font-size:13px;color:var(--dim);
user-select:none;padding:4px 0}summary:hover{color:var(--fg)}
table.detail{width:100%;border-collapse:collapse;margin-top:10px;font-size:13px;
font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
table.detail td,table.detail th{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left}
table.detail th{color:var(--dim);font-weight:600}
table.detail td:first-child{color:var(--dim);font-family:inherit;width:55%}
table.detail td.pos{color:var(--ok)}table.detail td.neg{color:var(--bad)}
.fix{counter-reset:f;list-style:none;padding:0}
.fix li{counter-increment:f;position:relative;padding-left:34px;margin-bottom:14px;max-width:66ch}
.fix li::before{content:counter(f);position:absolute;left:0;top:1px;width:22px;height:22px;
border-radius:99px;background:var(--line);color:var(--dim);font:600 12px ui-monospace,monospace;
display:grid;place-items:center}
footer{margin-top:64px;padding-top:20px;border-top:1px solid var(--line);
font-size:12px;color:var(--dim)}
footer code{font-family:ui-monospace,Menlo,Consolas,monospace}
"""


def render_report(verdict: Verdict, title: str | None = None) -> str:
    m = verdict.meta
    band = _hue(verdict.score / 100.0)
    sym = html.escape(str(m.get("symbol", "?")))
    doc_title = title or f"falsify — {sym}"

    meta_items = [
        ("Symbol", sym),
        ("Market", html.escape(str(m.get("market", "—")))),
        ("Sample", f"{m.get('bars', 0):,} bars &middot; {m.get('years', 0):.1f} yr"),
        ("Variants searched", f"{m.get('n_trials', 0):,}"),
        ("Reported Sharpe", f"{m.get('sharpe_annual', 0):.2f}"),
        ("Parameters", html.escape(", ".join(f"{k}={v:g}" for k, v in (m.get("params") or {}).items()) or "—")),
    ]
    meta_html = "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in meta_items)

    damning = (
        f'<p class="damning">{html.escape(verdict.headline_failure)}</p>'
        if verdict.headline_failure else ""
    )

    equity = m.get("equity") or []
    curve_html = (
        f'<h2>Net equity curve, after costs{_curve_scale(equity)}</h2>{_sparkline(equity)}'
        if equity and not verdict.broken else ""
    )

    cards = []
    for f in verdict.ordered_findings:
        hue = _hue(f.score)
        pill = "fatal" if f.fatal and f.score <= 0 else f"{f.score * 100:.0f}"
        extra = ""
        if f.name == "costs":
            extra = _cost_curve(f.detail)
        elif f.name == "regime":
            extra = _regime_table(f.detail)
        body = _detail_table(f.detail)
        advice = f'<p class="advice">{html.escape(f.advice)}</p>' if f.advice else ""
        details = (
            f"<details><summary>Numbers</summary>{extra}{body}</details>"
            if (body or extra) else ""
        )
        cards.append(
            f'<div class="card"><div class="card-head"><h3>{html.escape(f.title)}</h3>'
            f'<span class="pill {hue}">{pill}</span></div>'
            f"{_score_bar(f.score)}"
            f'<p class="headline">{html.escape(f.headline)}</p>{advice}{details}</div>'
        )

    fixes = verdict.advice
    fixes_html = (
        "<h2>What to fix, in order</h2><ol class='fix'>"
        + "".join(f"<li>{html.escape(a)}</li>" for a in fixes)
        + "</ol>"
    ) if fixes else ""

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(doc_title)}</title><style>{CSS}</style></head>
<body><div class="wrap">
<p class="brand">falsify</p>

<div class="verdict">
  <div class="score {band}">{verdict.score:.0f}<small>/100</small></div>
  <div><div class="label">{html.escape(verdict.label)}</div></div>
</div>
<p class="summary">{html.escape(verdict.summary)}</p>
{damning}

<dl class="meta">{meta_html}</dl>

{curve_html}

<h2>The prosecution</h2>
{"".join(cards)}

{fixes_html}

<footer>
Generated by <code>falsify {__import__("falsify_quant").__version__}</code> on {stamp}.
Deflated Sharpe and PBO after Bailey &amp; López de Prado. A high score is not a
prediction — it means these particular tests failed to kill the strategy.
</footer>
</div></body></html>"""


def write_report(verdict: Verdict, path: str | Path, title: str | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_report(verdict, title), encoding="utf-8")
    return p
