"""The local app, tested on the two things that could make it worse than no app.

A graphical interface makes re-running free, and free re-running is the exact search this
whole library exists to charge people for. So most of this file is about the counter: it
has to accumulate, it has to actually reach the deflation rather than just being drawn on
screen, and the reset has to be a deliberate act rather than a default.

The rest is the ordinary hazards of putting an HTTP server in front of code execution --
path traversal on the strategy argument and on the report argument, and markup arriving
through a filename.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from falsify.gui import PAGE, Investigation, Run, serve

STRATEGY = '''
import numpy as np
from falsify.indicators import rolling_mean

GRID = {"fast": [3, 5], "slow": [10, 20]}

def valid(p):
    return p["fast"] < p["slow"]

def strategy(bars, fast=3, slow=10):
    f, s = rolling_mean(bars.close, int(fast)), rolling_mean(bars.close, int(slow))
    w = np.where(f > s, 1.0, -1.0)
    w[np.isnan(f) | np.isnan(s)] = np.nan
    return w
'''


# --------------------------------------------------------------------------------------
# The counter
# --------------------------------------------------------------------------------------


def test_an_investigation_accumulates_trials():
    inv = Investigation()
    assert inv.n_searches == 0 and inv.n_trials == 0

    inv.record(Run("a", "SPY", 4, 61.0, "PLAUSIBLE"), np.array([0.1, 0.2, 0.3, 0.4]))
    inv.record(Run("b", "SPY", 6, 44.0, "UNPROVEN"), np.arange(6) / 10.0)

    assert inv.n_searches == 2
    assert inv.n_trials == 10


def test_non_finite_trial_sharpes_are_not_counted():
    """Failed grid points are recorded as -inf. Charging for them would overstate N."""
    inv = Investigation()
    inv.record(Run("a", "x", 4, 0.0, "NO EDGE FOUND"),
               np.array([0.1, np.inf, -np.inf, np.nan, 0.2]))
    assert inv.n_trials == 2


def test_reset_clears_the_count_but_remembers_that_it_happened():
    """Discarding silently would make the escape hatch invisible in the record."""
    inv = Investigation()
    inv.record(Run("a", "x", 4, 50.0, "UNPROVEN"), np.arange(4) / 10.0)
    inv.reset()

    assert inv.n_searches == 0 and inv.n_trials == 0
    assert inv.discarded_searches == 1 and inv.discarded_trials == 4

    inv.record(Run("b", "y", 3, 50.0, "UNPROVEN"), np.arange(3) / 10.0)
    inv.reset()
    assert inv.discarded_searches == 2 and inv.discarded_trials == 7


def test_the_counter_reaches_the_deflation_rather_than_only_the_screen():
    """The load-bearing claim of the whole design.

    If accumulated trials did not make the next verdict harder, the counter would be
    decoration and the interface would be exactly the free-re-run machine it is meant not
    to be.
    """
    from falsify.harness import Sweep
    from falsify.prosecute import check_deflation
    from falsify.spec import MarketSpec
    from falsify.stats import sharpe_columns

    rng = np.random.default_rng(21)
    returns = 0.0007 + 0.01 * rng.standard_normal((3000, 12))
    sw = Sweep(params=[{"i": float(i)} for i in range(12)],
               returns=returns.astype(np.float32), sharpes=sharpe_columns(returns),
               gross=returns.sum(axis=0), churn=np.ones(12),
               failed=np.zeros(12, dtype=bool),
               bars=__import__("falsify").Bars(close=100.0 * np.ones(3000)),
               spec=MarketSpec(name="t", asset_class="equity", bars_per_year=252,
                               fee=0.0, half_spread=0.0),
               strategy=lambda b, **p: np.zeros(len(b)), grid={"i": list(range(12))})

    inv = Investigation()
    first = check_deflation(sw, 0, prior_sharpes=inv.sharpes)
    for k in range(5):
        inv.record(Run(str(k), "x", 12, 0.0, "x"), sw.sharpes)
    later = check_deflation(sw, 0, prior_sharpes=inv.sharpes)

    assert later.detail["n_trials"] > first.detail["n_trials"]
    assert later.score < first.score


def test_session_serialises_for_the_page():
    inv = Investigation()
    inv.record(Run("a", "SPY · trend.py", 4, 61.0, "PLAUSIBLE"), np.arange(4) / 10.0)
    d = inv.to_dict()

    assert d["searches"] == 1 and d["trials"] == 4
    assert d["runs"][0]["band"] == "PLAUSIBLE"
    json.dumps(d)  # must survive the wire


# --------------------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------------------


def test_the_page_carries_no_external_references():
    """No CDN, no npm, no build step -- it has to work on a machine with no network."""
    for token in ("http://", "https://", "cdn", "<script src", "<link "):
        assert token not in PAGE.lower().replace("http://127.0.0.1", ""), token


def test_everything_interpolated_into_the_page_is_escaped():
    """Strategy names come from the filesystem and the symbol from an input box."""
    assert "const esc =" in PAGE
    # Values that originate outside this library must not reach innerHTML unescaped.
    for raw in ("+ r.label +", "+ v.summary +", "+ f.headline +", "+ j.error +",
                "+ s + '</option>"):
        assert raw not in PAGE, f"unescaped interpolation: {raw}"


def test_the_reset_button_states_what_it_discards():
    assert "discards" in PAGE
    assert "not when you dislike the answer" in PAGE


# --------------------------------------------------------------------------------------
# The server
# --------------------------------------------------------------------------------------


@pytest.fixture()
def live(tmp_path):
    (tmp_path / "strategies").mkdir()
    (tmp_path / "strategies" / "trend.py").write_text(STRATEGY, encoding="utf-8")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "ok.html").write_text("<h1>report</h1>", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("not a report", encoding="utf-8")

    srv = serve(tmp_path, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", tmp_path
    srv.shutdown()
    srv.server_close()


def _get(url: str):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.read()


def _post(url: str, payload: dict):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def test_it_serves_the_page_and_finds_strategy_files(live):
    base, _ = live
    status, body = _get(base + "/")
    assert status == 200 and b"falsify" in body

    d = json.loads(_get(base + "/api/session")[1])
    assert d["strategies"] == ["strategies/trend.py"]
    assert d["session"]["searches"] == 0


def test_reset_over_http(live):
    base, _ = live
    d = _post(base + "/api/reset", {})
    assert d["session"]["searches"] == 0


def test_reports_cannot_be_escaped_with_a_path(live):
    """The oldest bug a file server has."""
    base, _ = live
    assert _get(base + "/report?id=ok.html")[1] == b"<h1>report</h1>"

    for attack in ("../secret.txt", "..%2Fsecret.txt", "/etc/passwd",
                   "..%5C..%5Csecret.txt"):
        with pytest.raises(urllib.error.HTTPError) as e:
            _get(base + "/report?id=" + attack)
        assert e.value.code == 404


def test_a_strategy_path_cannot_escape_the_served_root(live):
    """This endpoint executes the file it is given. Confinement is not optional."""
    base, tmp = live
    outside = tmp.parent / "evil.py"
    outside.write_text("raise SystemExit('should never run')", encoding="utf-8")

    job = _post(base + "/api/run", {"strategy": "../evil.py", "symbol": "SPY"})["job"]
    for _ in range(200):
        j = json.loads(_get(base + f"/api/job?id={job}")[1])
        if j["state"] != "running":
            break
    assert j["state"] == "error"
    assert "no strategy file" in j["error"]


def test_an_unknown_job_is_a_404(live):
    base, _ = live
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + "/api/job?id=nope")
    assert e.value.code == 404


def test_a_malformed_body_does_not_take_the_server_down(live):
    base, _ = live
    req = urllib.request.Request(base + "/api/run", data=b"{not json",
                                 headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)
    assert e.value.code == 400

    assert _get(base + "/api/session")[0] == 200  # still alive


def test_an_oversized_body_is_refused_with_an_answer_not_a_dropped_socket(live):
    """The refusal has to reach the client, which means draining what it is still sending.

    Answering 413 and closing looks correct and is not: on macOS the client's write fails
    with ECONNRESET before it can read the response, so the caller learns only that the
    connection died. Linux and Windows buffer enough to hide it, which is how this shipped
    green on four runners and red on the fifth.
    """
    base, _ = live
    req = urllib.request.Request(base + "/api/run", data=b"x" * ((1 << 20) + 1024),
                                 headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=30)
    assert e.value.code == 413
    assert json.loads(e.value.read())["error"] == "body too large"

    assert _get(base + "/api/session")[0] == 200  # and the server is still up


def test_a_bad_strategy_surfaces_in_the_page_not_the_console(live):
    """A traceback in a terminal nobody is looking at is not an error message.

    The file fails at import, which is both a realistic user error and the one that
    happens *before* any prices are fetched -- so this suite never touches the network.
    """
    base, tmp = live
    (tmp / "strategies" / "broken.py").write_text(
        "import a_module_that_is_not_installed\nGRID = {'a': [1]}\n"
        "def strategy(bars, a=1):\n    return bars.close * 0\n",
        encoding="utf-8")

    job = _post(base + "/api/run",
                {"strategy": "strategies/broken.py", "symbol": "SPY",
                 "market": "equity", "bars": 300})["job"]
    for _ in range(400):
        j = json.loads(_get(base + f"/api/job?id={job}")[1])
        if j["state"] != "running":
            break
    assert j["state"] == "error"
    assert "ModuleNotFoundError" in j["error"]


def test_the_suite_never_reaches_the_network(live, monkeypatch):
    """CI must not depend on Yahoo being up. Pinned so a future test cannot add it."""
    import falsify.data

    def refuse(*_a, **_k):
        raise AssertionError("a test tried to fetch prices")

    monkeypatch.setattr(falsify.data, "_get", refuse)

    base, _ = live
    assert _get(base + "/api/session")[0] == 200
    job = _post(base + "/api/run", {"strategy": "../evil.py"})["job"]
    for _ in range(200):
        j = json.loads(_get(base + f"/api/job?id={job}")[1])
        if j["state"] != "running":
            break
    assert j["state"] == "error"
