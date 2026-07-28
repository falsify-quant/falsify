"""A local app, in your browser, that counts how many times you have asked.

    falsify-gui

Starts a small server on 127.0.0.1 and opens a page. Pick a market, point at a strategy
file, press Run, read the verdict. No npm, no build step, no CDN -- the page is a string
in this file and the server is `http.server`, because a tool whose selling point is that
you can audit the arithmetic should not arrive with a node_modules directory.

## Why a GUI is dangerous here, and what is done about it

Every other part of falsify is designed so there is no knob that improves your score. A
graphical interface undermines that by accident: it makes re-running *free*. Adjust the
grid, press Run again, watch the number move. Do that six times and keep the best, and you
have performed exactly the search the tool exists to charge you for -- except now the
charge does not appear, because each run looks like a fresh question.

So the session remembers. Every search inside an *investigation* contributes its trials to
the next verdict's deflation: run four grids and the fifth is deflated by all five. The
counter is on screen the whole time, and the score you see is the score after paying for
everything you have tried.

There is a reset, deliberately. Sometimes the next question genuinely is unrelated -- a
different market, a different idea -- and pretending otherwise would be its own kind of
dishonesty. It is a button you have to press, with the count you are discarding written on
it, which is the difference between an escape hatch and a default.

## Scope

Binds to the loopback interface only, and there is no authentication because there is
nothing to authenticate against. It executes the strategy file you point it at, in this
process, exactly as the command line does -- so do not expose the port, and do not run
strategy files you have not read. Hosting this for other people needs sandboxing that is
not here.
"""

from __future__ import annotations

import json
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

__all__ = ["Investigation", "Run", "serve", "main"]

MAX_BODY = 1 << 20  # a config blob, not an upload endpoint

# How much of an over-long body to read and throw away before answering 413.
#
# Answering without draining looks correct and is not: the client is still writing when
# the socket closes, so it sees the connection vanish rather than the refusal. Linux and
# Windows happen to buffer enough to hide it; macOS returns ECONNRESET and the caller
# never learns why it was rejected. Draining a bounded amount makes the refusal legible on
# every platform without turning the cap into a suggestion.
DRAIN_CAP = 8 << 20


# --------------------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------------------


@dataclass
class Run:
    id: str
    label: str
    n_trials: int
    score: float
    label_band: str
    report: str | None = None


@dataclass
class Investigation:
    """Every search asked inside one line of enquiry.

    `sharpes` is the pooled cross-section of trial Sharpes from previous runs. It is what
    gets handed to the deflation as `prior_sharpes`, and it is why the fifth run of a
    tweaked grid is harder to pass than the first.
    """

    runs: list[Run] = field(default_factory=list)
    sharpes: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    discarded_searches: int = 0
    discarded_trials: int = 0

    @property
    def n_searches(self) -> int:
        return len(self.runs)

    @property
    def n_trials(self) -> int:
        return int(len(self.sharpes))

    def record(self, run: Run, trial_sharpes: np.ndarray) -> None:
        self.runs.append(run)
        finite = np.asarray(trial_sharpes, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        self.sharpes = np.concatenate([self.sharpes, finite])

    def reset(self) -> None:
        self.discarded_searches += self.n_searches
        self.discarded_trials += self.n_trials
        self.runs = []
        self.sharpes = np.array([], dtype=np.float64)

    def to_dict(self) -> dict:
        return {
            "searches": self.n_searches,
            "trials": self.n_trials,
            "discarded_searches": self.discarded_searches,
            "discarded_trials": self.discarded_trials,
            "runs": [{"id": r.id, "label": r.label, "trials": r.n_trials,
                      "score": r.score, "band": r.label_band} for r in self.runs],
        }


@dataclass
class Job:
    id: str
    state: str = "running"  # running | done | error
    progress: list[str] = field(default_factory=list)
    verdict: dict | None = None
    error: str | None = None
    report: str | None = None


class App:
    def __init__(self, root: Path):
        self.root = root
        self.investigation = Investigation()
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    # -- discovery ------------------------------------------------------------------

    def strategy_files(self) -> list[str]:
        found: list[str] = []
        for pattern in ("strategies/*.py", "*.py"):
            for p in sorted(self.root.glob(pattern)):
                if p.name.startswith("_") or p.name in ("setup.py", "conftest.py"):
                    continue
                text = p.read_text(encoding="utf-8", errors="ignore")
                if "def strategy" in text and "GRID" in text:
                    found.append(str(p.relative_to(self.root)).replace("\\", "/"))
        return found

    # -- running --------------------------------------------------------------------

    def start(self, cfg: dict) -> str:
        job = Job(id=uuid.uuid4().hex[:12])
        with self.lock:
            self.jobs[job.id] = job
        threading.Thread(target=self._run, args=(job, cfg), daemon=True).start()
        return job.id

    def _run(self, job: Job, cfg: dict) -> None:
        try:
            import falsify_quant
            from falsify_quant.cli import _load_module
            from falsify_quant.data import GRANULARITY, load
            from falsify_quant.harness import sweep
            from falsify_quant.report import write_report
            from falsify_quant.spec import PRESETS

            # Confine to the served root. The strategy path arrives from the page, and
            # this executes it -- a value that can walk out of the tree with `..` turns a
            # loopback convenience into "run any file on this machine".
            root = self.root.resolve()
            path = (root / str(cfg.get("strategy", ""))).resolve()
            if root not in path.parents or not path.is_file():
                raise FileNotFoundError(
                    f"no strategy file at {cfg.get('strategy')!r} inside {root}")
            mod = _load_module(path)

            spec = PRESETS[cfg.get("market", "equity")]
            symbol = (cfg.get("symbol") or "SPY").strip()
            interval = cfg.get("interval", "1h")
            n_bars = int(cfg.get("bars", 5000))

            job.progress.append(f"fetching {symbol}")
            bars = load(symbol, asset_class=spec.asset_class,
                        interval=interval, bars=n_bars)
            if spec.asset_class == "crypto" and interval in GRANULARITY:
                spec = spec.at_bars_per_year(365.25 * 86400 / GRANULARITY[interval])

            job.progress.append(f"{len(bars):,} bars · {spec.name}")
            sw = sweep(mod.strategy, bars, spec, mod.GRID,
                       valid=getattr(mod, "valid", None))

            with self.lock:
                prior = self.investigation.sharpes.copy()
            n_here = int((~sw.failed).sum())
            if len(prior):
                job.progress.append(
                    f"{n_here} combinations here, {len(prior)} already tried this "
                    f"investigation — deflating by all {n_here + len(prior)}")

            verdict = falsify_quant.run_on_sweep(
                sw, sw.best_index,
                n_permutations=int(cfg.get("permutations", 100)),
                permutation_method=cfg.get("null", "iid"),
                prior_sharpes=prior,
                progress=job.progress.append,
            )

            reports = root / "reports"
            reports.mkdir(exist_ok=True)
            out = write_report(verdict, reports / f"gui-{job.id}.html")

            run = Run(id=job.id, label=f"{symbol} · {path.name}", n_trials=n_here,
                      score=round(float(verdict.score), 1), label_band=verdict.label,
                      report=out.name)
            with self.lock:
                self.investigation.record(run, sw.sharpes[~sw.failed])
                job.verdict = _verdict_json(verdict, n_here, len(prior))
                job.report = out.name
                job.state = "done"
        except Exception as exc:  # noqa: BLE001 -- surfaced in the page, not the console
            job.error = f"{type(exc).__name__}: {exc}"
            job.state = "error"
            traceback.print_exc()


def _verdict_json(v, n_here: int, n_prior: int) -> dict:
    return {
        "score": round(float(v.score), 1),
        "label": v.label,
        "summary": v.summary,
        "broken": bool(v.broken),
        "trials_this_run": n_here,
        "trials_charged": n_here + n_prior,
        "findings": [
            {"name": f.name, "title": f.title, "score": round(float(f.score), 2),
             "headline": f.headline, "fatal": bool(f.fatal)}
            for f in v.ordered_findings
        ],
        "advice": list(v.advice),
    }


# --------------------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    app: App = None  # type: ignore[assignment]
    server_version = "falsify"

    def log_message(self, *_args):  # noqa: D102 -- the page is the interface, not the log
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path)
        query = parse_qs(route.query)

        if route.path in ("/", "/index.html"):
            return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")

        if route.path == "/api/session":
            with self.app.lock:
                return self._json({"session": self.app.investigation.to_dict(),
                                   "strategies": self.app.strategy_files()})

        if route.path == "/api/job":
            job = self.app.jobs.get((query.get("id") or [""])[0])
            if job is None:
                return self._json({"error": "no such job"}, 404)
            return self._json({
                "state": job.state, "progress": job.progress,
                "verdict": job.verdict, "error": job.error, "report": job.report,
            })

        if route.path == "/report":
            name = (query.get("id") or [""])[0]
            # Resolve and confine: this serves from a directory, and a path parameter
            # that reaches outside it is the oldest bug a file server has.
            reports = (self.app.root / "reports").resolve()
            target = (reports / Path(name).name).resolve()
            if target.parent != reports or not target.is_file():
                return self._send(404, b"no such report", "text/plain")
            return self._send(200, target.read_bytes(), "text/html; charset=utf-8")

        return self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            remaining = min(length, DRAIN_CAP)
            while remaining > 0:
                chunk = self.rfile.read(min(1 << 16, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
            return self._json({"error": "body too large"}, 413)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad json"}, 400)

        if route.path == "/api/run":
            return self._json({"job": self.app.start(payload)})

        if route.path == "/api/reset":
            with self.app.lock:
                self.app.investigation.reset()
                return self._json({"session": self.app.investigation.to_dict()})

        return self._json({"error": "not found"}, 404)


def serve(root: Path, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"app": App(root)})
    return ThreadingHTTPServer((host, port), handler)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import webbrowser

    p = argparse.ArgumentParser(
        prog="falsify-gui",
        description="Run falsify from a local page in your browser.")
    p.add_argument("--root", type=Path, default=Path.cwd(),
                   help="directory to look for strategy files in (default: cwd)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args(argv)

    server = serve(args.root.resolve(), port=args.port)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"falsify is at {url}   (ctrl-c to stop)")
    print("loopback only, and it executes the strategy file you point it at — "
          "do not expose this port.")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


# --------------------------------------------------------------------------------------
# The page
#
# Everything the page renders is escaped through `esc()` before it reaches innerHTML.
# Most of it originates in this library, but not all: strategy names come from the
# filesystem, the symbol comes from the input box, and error text can carry either. A
# single-user loopback app is a weak excuse for building the injection in anyway, and the
# fix is one function.
# --------------------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>falsify</title>
<style>
:root{--bg:#fbfbfa;--fg:#1a1a19;--dim:#6b6b66;--line:#e2e1dd;--card:#fff;
      --good:#1a7f4b;--mid:#a86b12;--bad:#b3261e;--accent:#1a1a19}
@media(prefers-color-scheme:dark){:root{--bg:#141413;--fg:#eeeeec;--dim:#9a9a94;
      --line:#2c2c2a;--card:#1c1c1b;--good:#4ec38a;--mid:#e0a53c;--bad:#f0736a;
      --accent:#eeeeec}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 ui-sans-serif,
     -apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:860px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:22px;margin:0;letter-spacing:-.01em}
h1 span{color:var(--dim);font-weight:400}
.sub{color:var(--dim);margin:6px 0 28px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
      padding:18px;margin-bottom:16px}
label{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.06em;
      color:var(--dim);margin-bottom:5px}
input,select{width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:7px;
      background:var(--bg);color:var(--fg);font:inherit;font-size:14px}
.row{display:grid;gap:12px;margin-bottom:12px}
.r3{grid-template-columns:2fr 1fr 1fr}
@media(max-width:620px){.r3{grid-template-columns:1fr}}
button{font:inherit;font-weight:600;padding:9px 18px;border-radius:7px;cursor:pointer;
      border:1px solid var(--accent);background:var(--accent);color:var(--bg)}
button.ghost{background:transparent;color:var(--dim);border-color:var(--line);
      font-weight:400}
button:disabled{opacity:.45;cursor:default}
.counter{display:flex;align-items:center;justify-content:space-between;gap:12px;
      flex-wrap:wrap;font-size:14px}
.counter b{font-variant-numeric:tabular-nums}
.warn{color:var(--mid)}
pre{margin:0;font:12.5px/1.7 ui-monospace,SFMono-Regular,Menlo,monospace;
      color:var(--dim);white-space:pre-wrap}
.score{display:flex;align-items:baseline;gap:14px;margin-bottom:4px}
.score b{font-size:44px;line-height:1;font-variant-numeric:tabular-nums}
.band{font-size:15px;font-weight:600;letter-spacing:.04em}
.good{color:var(--good)}.mid{color:var(--mid)}.bad{color:var(--bad)}
.f{display:grid;grid-template-columns:52px 1fr;gap:12px;padding:11px 0;
      border-top:1px solid var(--line)}
.f .s{font-variant-numeric:tabular-nums;font-weight:600;font-size:13px;text-align:right}
.f .t{font-weight:600;font-size:13px}
.f .h{color:var(--dim);font-size:13px}
ol{margin:10px 0 0;padding-left:20px;color:var(--dim);font-size:13px}
a{color:inherit}
.hist{font-size:13px;color:var(--dim);border-top:1px solid var(--line);padding-top:10px;
      margin-top:14px}
.hist div{display:flex;justify-content:space-between;gap:12px;padding:3px 0}
.err{color:var(--bad);font-size:13px;white-space:pre-wrap}
</style></head><body><div class="wrap">

<h1>falsify <span>— try to prove it is nothing</span></h1>
<p class="sub">Runs your parameter search itself, then spends seven tests trying to kill
the result.</p>

<div class="card counter" id="counter"></div>

<div class="card">
  <div class="row r3">
    <div><label for="strategy">Strategy file</label>
      <select id="strategy"></select></div>
    <div><label for="symbol">Symbol</label>
      <input id="symbol" value="SPY"></div>
    <div><label for="market">Market</label>
      <select id="market">
        <option value="equity">US equity</option>
        <option value="equity-smallcap">US equity, small cap</option>
        <option value="crypto-spot">Crypto spot</option>
        <option value="crypto-perp">Crypto perp</option>
      </select></div>
  </div>
  <div class="row r3">
    <div><label for="bars">Bars of history</label>
      <input id="bars" type="number" value="5000" min="200" step="500"></div>
    <div><label for="interval">Crypto bar size</label>
      <select id="interval">
        <option>1h</option><option>6h</option><option>1d</option><option>15m</option>
      </select></div>
    <div><label for="perms">Noise runs</label>
      <input id="perms" type="number" value="100" min="10" step="10"></div>
  </div>
  <button id="go">Run</button>
</div>

<div class="card" id="out" hidden></div>

</div><script>
const $ = id => document.getElementById(id);
const ENT = {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'};
const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ENT[c]);

let session = {searches:0, trials:0, runs:[], discarded_searches:0, discarded_trials:0};

const bandClass = s => s >= 60 ? 'good' : s >= 40 ? 'mid' : 'bad';
const num = n => Number(n).toLocaleString();

function drawCounter(){
  const c = $('counter');
  c.innerHTML = '<div>' + (session.searches === 0
    ? 'Nothing tried yet in this investigation.'
    : '<b>' + num(session.searches) + '</b> search' + (session.searches > 1 ? 'es' : '')
      + ' this investigation, <b>' + num(session.trials) + '</b> combination'
      + (session.trials === 1 ? '' : 's') + ' total.'
      + (session.searches >= 2
         ? ' <span class="warn">The next verdict is deflated by all of them.</span>' : '')
  ) + '</div>';

  if (session.searches) {
    const b = document.createElement('button');
    b.className = 'ghost';
    b.textContent = 'New investigation (discards ' + num(session.trials) + ')';
    b.onclick = async () => {
      const ok = confirm('Start a new investigation?\n\n' + session.trials
        + ' combinations across ' + session.searches + ' searches stop counting against '
        + 'your next verdict.\n\nDo this when the next question is genuinely unrelated — '
        + 'not when you dislike the answer.');
      if (!ok) return;
      const r = await fetch('/api/reset', {method:'POST', body:'{}'});
      session = (await r.json()).session;
      drawCounter(); renderHistory();
    };
    c.appendChild(b);
  }
}

function renderHistory(){
  const out = $('out');
  const old = out.querySelector('.hist');
  if (old) old.remove();
  if (out.hidden || session.runs.length < 2) return;
  const h = document.createElement('div');
  h.className = 'hist';
  h.innerHTML = '<div style="font-weight:600;color:var(--fg)">This investigation</div>'
    + session.runs.map(r => '<div><span>' + esc(r.label) + '</span><span class="'
        + bandClass(r.score) + '">' + esc(r.score) + ' ' + esc(r.band)
        + '</span></div>').join('');
  out.appendChild(h);
}

async function refresh(){
  const d = await (await fetch('/api/session')).json();
  session = d.session;
  const sel = $('strategy');
  if (!sel.options.length) {
    sel.innerHTML = d.strategies.length
      ? d.strategies.map(s => '<option>' + esc(s) + '</option>').join('')
      : '<option value="">no strategy files found here</option>';
  }
  drawCounter();
}

$('go').onclick = async () => {
  const out = $('out'); out.hidden = false;
  $('go').disabled = true;
  out.innerHTML = '<pre id="log">starting…</pre>';

  const body = {
    strategy: $('strategy').value, symbol: $('symbol').value,
    market: $('market').value, interval: $('interval').value,
    bars: +$('bars').value, permutations: +$('perms').value,
  };
  const {job} = await (await fetch('/api/run',
    {method:'POST', body: JSON.stringify(body)})).json();

  const poll = setInterval(async () => {
    const j = await (await fetch('/api/job?id=' + encodeURIComponent(job))).json();
    if ($('log')) $('log').textContent = j.progress.join('\n') || 'starting…';
    if (j.state === 'running') return;
    clearInterval(poll);
    $('go').disabled = false;

    if (j.state === 'error') {
      const d = document.createElement('div');
      d.className = 'err';
      d.textContent = j.error;
      out.replaceChildren(d);
      return;
    }

    const v = j.verdict, cls = v.broken ? 'bad' : bandClass(v.score);
    out.innerHTML =
      '<div class="score"><b class="' + cls + '">' + esc(v.score) + '</b>'
      + '<span class="band ' + cls + '">' + esc(v.label) + '</span></div>'
      + '<p class="sub" style="margin:8px 0 14px">' + esc(v.summary) + '</p>'
      + (v.trials_charged > v.trials_this_run
          ? '<p class="warn" style="font-size:13px;margin:-6px 0 14px">Deflated by '
            + num(v.trials_charged) + ' combinations — ' + num(v.trials_this_run)
            + ' from this run and ' + num(v.trials_charged - v.trials_this_run)
            + ' from earlier searches in this investigation.</p>'
          : '')
      + v.findings.map(f => '<div class="f"><div class="s '
          + (f.fatal && f.score <= 0 ? 'bad' : bandClass(f.score * 100)) + '">'
          + (f.fatal && f.score <= 0 ? 'FATAL' : esc(f.score.toFixed(2)))
          + '</div><div><div class="t">' + esc(f.title) + '</div>'
          + '<div class="h">' + esc(f.headline) + '</div></div></div>').join('')
      + (v.advice.length
          ? '<ol>' + v.advice.map(a => '<li>' + esc(a) + '</li>').join('') + '</ol>' : '')
      + '<p style="margin:16px 0 0;font-size:13px"><a href="/report?id='
      + encodeURIComponent(j.report) + '" target="_blank">Open the full report →</a></p>';

    await refresh();
    renderHistory();
  }, 700);
};

refresh();
</script></body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
