"""The watch daemon, tested on the thing that actually makes it useful: staying quiet.

Any loop can call `monitor()` on a timer. The value is entirely in what it declines to
say, so most of this file is about silence -- a condition that is still true does not
notify, a restart does not re-announce, a flapping check gets muted after being named
once. The two exceptions are the cases where silence would be a lie: a resolved alarm,
and a daemon that has stopped working.

The clock is injected everywhere. A scheduler that only runs in real time only gets
tested in real time, which means it does not get tested.
"""

from __future__ import annotations

import io
import json

import pytest

from falsify_quant.monitor import Alert, MonitorVerdict
from falsify_quant.watch import (
    AlertState,
    Event,
    StateStore,
    WatchConfig,
    diff_alerts,
    file_sink,
    log_sink,
    run_forever,
    run_once,
    webhook_sink,
)

HOUR = 3600.0


def _alerts(**kw) -> list[Alert]:
    return [Alert(name=n, severity=s, headline=f"{n} is {s}") for n, s in kw.items()]


def _verdict(alerts) -> MonitorVerdict:
    worst = max((a.severity for a in alerts), key={"ok": 0, "watch": 1, "alarm": 2}.get,
                default="ok")
    return MonitorVerdict(status=worst, alerts=alerts, record=None)


# --------------------------------------------------------------------------------------
# Transitions
# --------------------------------------------------------------------------------------


def test_a_new_alarm_notifies():
    events, state = diff_alerts({}, _alerts(divergence="alarm"), now=1000.0)
    assert [(e.kind, e.name) for e in events] == [("raised", "divergence")]
    assert state["divergence"].severity == "alarm"
    assert state["divergence"].since == 1000.0


def test_the_same_alarm_next_poll_says_nothing():
    """The rule the whole daemon exists for. Six hours of alarm is one piece of news."""
    _, state = diff_alerts({}, _alerts(divergence="alarm"), now=1000.0)
    for t in (1300.0, 1600.0, 1900.0):
        events, state = diff_alerts(state, _alerts(divergence="alarm"), now=t)
        assert events == []
    assert state["divergence"].since == 1000.0  # still dated from when it started


def test_clearing_notifies_because_silence_is_ambiguous():
    """An alarm that resolved and a daemon that died emit the same nothing."""
    _, state = diff_alerts({}, _alerts(divergence="alarm"), now=1000.0)
    events, _ = diff_alerts(state, _alerts(divergence="ok"), now=2000.0)
    assert [(e.kind, e.severity) for e in events] == [("cleared", "ok")]


def test_worsening_notifies_but_is_not_a_new_alert():
    _, state = diff_alerts({}, _alerts(costs="watch"), now=1000.0)
    events, _ = diff_alerts(state, _alerts(costs="alarm"), now=2000.0)
    assert [e.kind for e in events] == ["worsened"]


def test_easing_off_counts_as_good_news():
    _, state = diff_alerts({}, _alerts(costs="alarm"), now=1000.0)
    events, _ = diff_alerts(state, _alerts(costs="watch"), now=2000.0)
    assert [e.kind for e in events] == ["cleared"]


def test_a_steady_ok_never_notifies():
    state: dict[str, AlertState] = {}
    for t in range(0, 10):
        events, state = diff_alerts(state, _alerts(costs="ok", heartbeat="ok"),
                                    now=float(t) * 300)
        assert events == []


def test_checks_are_tracked_independently():
    _, state = diff_alerts({}, _alerts(a="alarm", b="ok"), now=1000.0)
    events, _ = diff_alerts(state, _alerts(a="alarm", b="alarm"), now=2000.0)
    assert [e.name for e in events] == ["b"]


def test_a_check_that_disappears_is_announced_if_it_was_bad():
    _, state = diff_alerts({}, _alerts(divergence="alarm", costs="ok"), now=1000.0)
    events, _ = diff_alerts(state, _alerts(costs="ok"), now=2000.0)
    assert [(e.kind, e.name) for e in events] == [("cleared", "divergence")]


def test_a_healthy_check_that_disappears_says_nothing():
    _, state = diff_alerts({}, _alerts(costs="ok"), now=1000.0)
    events, _ = diff_alerts(state, [], now=2000.0)
    assert events == []


# --------------------------------------------------------------------------------------
# Flapping
# --------------------------------------------------------------------------------------


def test_a_flapping_check_is_named_once_then_muted():
    """Nine toggles in an hour is one message about a bad threshold, not nine alerts."""
    state: dict[str, AlertState] = {}
    kinds = []
    for i in range(12):
        sev = "alarm" if i % 2 else "ok"
        events, state = diff_alerts(state, _alerts(edge=sev), now=1000.0 + i * 60,
                                    flap_window_s=HOUR, flap_limit=4, mute_s=HOUR)
        kinds += [e.kind for e in events]

    assert kinds.count("flapping") == 1, kinds
    assert kinds.index("flapping") == 3  # after the limit is reached, not before
    assert kinds.count("raised") + kinds.count("cleared") == 3  # then silence


def test_muting_expires_and_the_check_can_speak_again():
    state: dict[str, AlertState] = {}
    for i in range(9):  # ends on "ok", so the next alarm is a real transition
        _, state = diff_alerts(state, _alerts(edge="alarm" if i % 2 else "ok"),
                               now=1000.0 + i * 60, flap_limit=4, mute_s=HOUR)
    assert state["edge"].muted_until > 0
    assert state["edge"].severity == "ok"

    events, _ = diff_alerts(state, _alerts(edge="alarm"), now=1000.0 + 3 * HOUR,
                            flap_limit=4, mute_s=HOUR)
    assert [e.kind for e in events] == ["raised"]


def test_slow_toggling_is_not_flapping():
    """Twice last month and twice today is not a broken check."""
    state: dict[str, AlertState] = {}
    kinds = []
    for i in range(10):
        events, state = diff_alerts(state, _alerts(edge="alarm" if i % 2 else "ok"),
                                    now=1000.0 + i * 6 * HOUR,
                                    flap_window_s=HOUR, flap_limit=4)
        kinds += [e.kind for e in events]
    assert "flapping" not in kinds
    assert len(kinds) == 9


# --------------------------------------------------------------------------------------
# State across restarts
# --------------------------------------------------------------------------------------


def test_state_survives_a_restart(tmp_path):
    """A deploy must not turn into a wall of alerts."""
    store = StateStore(tmp_path / "state.json")
    _, state = diff_alerts({}, _alerts(divergence="alarm"), now=1000.0)
    store.save(state, last_heartbeat=1000.0)

    reloaded = StateStore(tmp_path / "state.json").load()
    events, _ = diff_alerts(reloaded, _alerts(divergence="alarm"), now=2000.0)
    assert events == []
    assert reloaded["divergence"].since == 1000.0


def test_a_missing_state_file_is_not_an_error(tmp_path):
    assert StateStore(tmp_path / "nope.json").load() == {}
    assert StateStore(tmp_path / "nope.json").last_heartbeat() == 0.0


def test_a_corrupt_state_file_is_not_an_error(tmp_path):
    """A daemon that will not start because its own scratch file is truncated is a
    daemon that stops watching at exactly the wrong moment."""
    p = tmp_path / "state.json"
    p.write_text("{ this is not json", encoding="utf-8")
    assert StateStore(p).load() == {}
    assert StateStore(p).last_heartbeat() == 0.0


def test_saving_is_atomic(tmp_path):
    """A killed daemon must not leave a half-written state file behind."""
    p = tmp_path / "state.json"
    store = StateStore(p)
    store.save({"a": AlertState(severity="alarm", since=1.0)}, last_heartbeat=2.0)
    store.save({"a": AlertState(severity="ok", since=3.0)}, last_heartbeat=4.0)

    assert not p.with_suffix(".json.tmp").exists()
    assert json.loads(p.read_text(encoding="utf-8"))["alerts"]["a"]["severity"] == "ok"
    assert store.last_heartbeat() == 4.0


# --------------------------------------------------------------------------------------
# Heartbeat
# --------------------------------------------------------------------------------------


def test_the_heartbeat_fires_even_when_nothing_is_wrong(tmp_path):
    """Silence has to be earned. A crashed watchdog emits the same nothing as a calm one."""
    store = StateStore(tmp_path / "s.json")
    cfg = WatchConfig(heartbeat_s=6 * HOUR)
    poll = lambda: _verdict(_alerts(costs="ok"))
    t0 = 1.7e9

    first = run_once(poll, store, [], cfg, now=t0)
    assert [e.kind for e in first] == ["heartbeat"]
    assert "started watching" in first[0].headline

    quiet = run_once(poll, store, [], cfg, now=t0 + HOUR)
    assert quiet == []

    later = run_once(poll, store, [], cfg, now=t0 + 7 * HOUR)
    assert [e.kind for e in later] == ["heartbeat"]
    assert "still watching" in later[0].headline


def test_the_first_run_always_beats_whatever_the_clock_says(tmp_path):
    """`the watchdog is up` must not depend on how far away the unix epoch happens to be."""
    events = run_once(lambda: _verdict(_alerts(x="ok")), StateStore(tmp_path / "s.json"),
                      [], WatchConfig(heartbeat_s=10 * HOUR), now=5.0)
    assert [e.kind for e in events] == ["heartbeat"]
    assert events[0].detail["first"] is True


def test_the_heartbeat_reports_the_worst_state(tmp_path):
    store = StateStore(tmp_path / "s.json")
    events = run_once(lambda: _verdict(_alerts(a="ok", b="alarm")), store, [],
                      WatchConfig(), now=1.7e9)
    beat = next(e for e in events if e.kind == "heartbeat")
    assert beat.detail["worst"] == "alarm"
    assert beat.detail["checks"] == 2


# --------------------------------------------------------------------------------------
# Surviving what it watches
# --------------------------------------------------------------------------------------


def test_a_failing_monitor_becomes_an_event_not_a_crash(tmp_path):
    """A locked trading database is not a reason to stop watching it."""
    store = StateStore(tmp_path / "s.json")

    def broken():
        raise sqlite_locked()

    class sqlite_locked(RuntimeError):
        def __init__(self):
            super().__init__("database is locked")

    events = run_once(broken, store, [], WatchConfig(), now=1000.0)
    err = next(e for e in events if e.kind == "error")
    assert err.severity == "alarm"
    assert "database is locked" in err.headline


def test_a_failing_monitor_does_not_wipe_remembered_state(tmp_path):
    store = StateStore(tmp_path / "s.json")
    run_once(lambda: _verdict(_alerts(divergence="alarm")), store, [], WatchConfig(),
             now=1000.0)

    def broken():
        raise RuntimeError("boom")

    run_once(broken, store, [], WatchConfig(), now=2000.0)
    assert store.load()["divergence"].severity == "alarm"

    # ... so when it recovers, the still-true alarm is still not re-announced.
    events = run_once(lambda: _verdict(_alerts(divergence="alarm")), store, [],
                      WatchConfig(), now=3000.0)
    assert events == []


# --------------------------------------------------------------------------------------
# Sinks
# --------------------------------------------------------------------------------------


def test_log_sink_writes_one_readable_line_each():
    buf = io.StringIO()
    log_sink(buf)([Event(kind="raised", name="costs", severity="alarm",
                         headline="costs are 3x the model", at=1.7e9)])
    line = buf.getvalue().strip()
    assert "RAISED" in line and "costs" in line and "3x the model" in line
    assert line.startswith("2023-")


def test_file_sink_appends_one_json_object_per_line(tmp_path):
    p = tmp_path / "events.jsonl"
    sink = file_sink(p)
    sink([Event(kind="raised", name="a", severity="alarm", headline="x", at=1.0)])
    sink([Event(kind="cleared", name="a", severity="ok", headline="y", at=2.0)])

    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(x)["kind"] for x in lines] == ["raised", "cleared"]


def test_file_sink_creates_its_directory(tmp_path):
    p = tmp_path / "deep" / "nested" / "events.jsonl"
    file_sink(p)([Event(kind="raised", name="a", severity="alarm", headline="x")])
    assert p.exists()


def test_webhook_failure_is_logged_and_dropped_not_raised():
    """A watchdog that spends its time managing a delivery queue is not watching."""
    said = []
    sink = webhook_sink("http://127.0.0.1:9/never", timeout=0.2, log=said.append)
    sink([Event(kind="raised", name="a", severity="alarm", headline="x")])

    assert len(said) == 1 and "failed" in said[0]
    assert "1 event(s) dropped" in said[0]


def test_webhook_does_not_call_out_on_an_empty_batch():
    said = []
    webhook_sink("http://127.0.0.1:9/never", log=said.append)([])
    assert said == []


def test_every_sink_receives_every_event(tmp_path):
    a, b = io.StringIO(), tmp_path / "e.jsonl"
    run_once(lambda: _verdict(_alerts(x="alarm")), StateStore(tmp_path / "s.json"),
             [log_sink(a), file_sink(b)], WatchConfig(), now=1.7e9)

    assert "RAISED" in a.getvalue() and "HEARTBEAT" in a.getvalue()
    assert len(b.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_sinks_are_not_called_when_there_is_nothing_to_say(tmp_path):
    calls = []
    store = StateStore(tmp_path / "s.json")
    cfg = WatchConfig(heartbeat_s=6 * HOUR)

    run_once(lambda: _verdict(_alerts(x="ok")), store, [calls.append], cfg, now=1.7e9)
    run_once(lambda: _verdict(_alerts(x="ok")), store, [calls.append], cfg,
             now=1.7e9 + HOUR)
    assert len(calls) == 1  # the opening heartbeat only


# --------------------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------------------


def test_run_forever_stops_when_told(tmp_path):
    polls = []
    slept = []

    def poll():
        polls.append(1)
        return _verdict(_alerts(x="ok"))

    run_forever(poll, StateStore(tmp_path / "s.json"), [], WatchConfig(poll_s=42.0),
                stop=lambda: len(polls) >= 3, sleep=slept.append)

    assert len(polls) == 3
    assert slept == [42.0, 42.0]  # no trailing sleep after the last poll


def test_run_forever_does_not_poll_at_all_if_already_stopped(tmp_path):
    polls = []
    run_forever(lambda: polls.append(1), StateStore(tmp_path / "s.json"), [],
                WatchConfig(), stop=lambda: True, sleep=lambda _: None)
    assert polls == []


@pytest.mark.parametrize("severity", ["ok", "watch", "alarm"])
def test_event_line_is_printable_for_every_severity(severity):
    e = Event(kind="raised", name="n", severity=severity, headline="h", at=1.7e9)
    assert severity in (e.severity,) and e.line


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


def test_the_example_config_is_valid_json():
    """It is the first thing anyone copies. It must not need fixing first."""
    from falsify_quant.watch import CONFIG_EXAMPLE

    cfg = json.loads(CONFIG_EXAMPLE)
    assert {"database", "state", "sinks"} <= set(cfg)
    assert {s["kind"] for s in cfg["sinks"]} == {"log", "file", "webhook"}


@pytest.mark.parametrize("value,want", [
    ("2026-07-23", 1784764800.0),
    ("2026-07-23T12:00:00+00:00", 1784808000.0),
    (1784764800, 1784764800.0),
    (None, None),
    ("", None),
])
def test_since_accepts_dates_and_timestamps(value, want):
    from falsify_quant.watch import _as_epoch

    assert _as_epoch(value) == want


def test_build_sinks_defaults_to_logging():
    from falsify_quant.watch import build_sinks

    assert len(build_sinks([])) == 1


def test_build_sinks_rejects_an_unknown_kind():
    from falsify_quant.watch import build_sinks

    with pytest.raises(ValueError, match="unknown sink"):
        build_sinks([{"kind": "carrier-pigeon"}])


@pytest.mark.parametrize("url", [
    "hooks.slack.com/services/XXX",   # the pasted-without-scheme case
    "//hooks.slack.com/x",
    "www.example.com/hook",
])
def test_a_webhook_url_without_a_scheme_is_refused_at_config_time(url):
    """urllib raises on this from inside the sink, where it is one drop per cycle.

    An alert channel that has never worked, and says so only in a log nobody reads,
    is worse than no webhook at all -- the operator believes they are covered. This
    is the one config mistake whose punishment is silence, so it fails the deploy.
    """
    from falsify_quant.watch import build_sinks

    with pytest.raises(ValueError, match="http"):
        build_sinks([{"kind": "webhook", "url": url}])


def test_ordinary_webhook_urls_still_build():
    from falsify_quant.watch import build_sinks

    assert len(build_sinks([{"kind": "webhook", "url": "https://example.invalid/h"},
                            {"kind": "webhook", "url": "http://127.0.0.1:9000/h"}])) == 2


@pytest.mark.parametrize("spec,missing", [
    ({"kind": "webhook"}, "url"),
    ({"kind": "webhook", "url": "   "}, "url"),
    ({"kind": "file"}, "path"),
    ({"kind": "file", "path": ""}, "path"),
])
def test_a_sink_missing_its_target_says_which_one(spec, missing):
    """The bare KeyError this replaces said `'url'` and nothing else."""
    from falsify_quant.watch import build_sinks

    with pytest.raises(ValueError, match=missing):
        build_sinks([spec])


# --------------------------------------------------------------------------------------
# One sink must not be able to take down the watching
#
# The useful configuration is more than one sink: a local file to grep, and a webhook
# that reaches a person. A bare `for sink in sinks: sink(events)` couples them, and the
# wrong way round -- the local convenience is the one most likely to fail, and it was
# taking the delivery that matters with it.
# --------------------------------------------------------------------------------------


def _explodes(events):
    raise OSError("log volume is full")


def test_a_failing_sink_does_not_end_the_cycle(tmp_path):
    run_once(lambda: _verdict(_alerts(x="alarm")), StateStore(tmp_path / "s.json"),
             [_explodes], WatchConfig(), now=1.7e9)      # must not raise


def test_a_failing_sink_does_not_starve_the_others(tmp_path):
    got: list = []
    run_once(lambda: _verdict(_alerts(x="alarm")), StateStore(tmp_path / "s.json"),
             [_explodes, got.append], WatchConfig(), now=1.7e9)

    assert len(got) == 1, "the webhook that reaches a human never fired"
    assert {e.kind for e in got[0]} == {"raised", "heartbeat"}


def test_a_failing_sink_does_not_lose_the_state(tmp_path):
    """Skipping the save means every alert is announced again on the next cycle."""
    store = StateStore(tmp_path / "s.json")
    run_once(lambda: _verdict(_alerts(x="alarm")), store, [_explodes], WatchConfig(),
             now=1.7e9)

    assert store.path.exists()
    assert set(store.load()) == {"x"}

    got: list = []
    run_once(lambda: _verdict(_alerts(x="alarm")), store, [got.append], WatchConfig(),
             now=1.7e9 + 60)
    assert got == [], "an unchanged alarm was announced twice"


def test_the_failure_is_reported_rather_than_swallowed(tmp_path, capsys):
    run_once(lambda: _verdict(_alerts(x="alarm")), StateStore(tmp_path / "s.json"),
             [_explodes], WatchConfig(), now=1.7e9)

    err = capsys.readouterr().err
    assert "log volume is full" in err and "dropped" in err


def test_a_sink_that_fails_while_reporting_a_failure_is_still_survivable(tmp_path):
    """Belt and braces: the report path is itself inside a try for a reason."""
    class Hostile:
        def __call__(self, events):
            raise OSError("boom")

        def __str__(self):
            raise RuntimeError("even my repr is broken")

    run_once(lambda: _verdict(_alerts(x="alarm")), StateStore(tmp_path / "s.json"),
             [Hostile()], WatchConfig(), now=1.7e9)      # must not raise


class _FullDisk(StateStore):
    def save(self, state, last_heartbeat):
        raise OSError("no space left on device")


def test_a_state_volume_that_fills_up_does_not_end_the_daemon(tmp_path):
    """One volume further out than the sinks, and the same argument.

    Carrying on means announcing every standing alert again next cycle. That is
    noisy, and the noise is the right signal -- an operator seeing the same alarm
    every five minutes goes and looks. Exiting produces silence, which is the one
    thing this module exists to stop being ambiguous.
    """
    got: list = []
    cycles: list = []

    def poll():
        cycles.append(1)
        return _verdict(_alerts(x="alarm"))

    run_forever(poll, _FullDisk(tmp_path / "s.json"), [got.append],
                WatchConfig(poll_s=0.0),
                stop=lambda: len(cycles) >= 3, sleep=lambda _: None)

    assert len(cycles) == 3
    assert len(got) == 3, "the alarm should be re-announced, not swallowed"


def test_a_failed_state_save_says_why_the_alerts_repeat(tmp_path, capsys):
    run_once(lambda: _verdict(_alerts(x="alarm")), _FullDisk(tmp_path / "s.json"),
             [], WatchConfig(), now=1.7e9)

    err = capsys.readouterr().err
    assert "no space left on device" in err
    assert "repeat" in err


def test_the_events_still_go_out_when_the_state_cannot_be_saved(tmp_path):
    """Delivery comes first on purpose: the alert matters more than the bookkeeping."""
    got: list = []
    run_once(lambda: _verdict(_alerts(x="alarm")), _FullDisk(tmp_path / "s.json"),
             [got.append], WatchConfig(), now=1.7e9)

    assert len(got) == 1
    assert {e.kind for e in got[0]} == {"raised", "heartbeat"}


def test_run_forever_survives_a_sink_that_always_fails(tmp_path):
    """The whole point. A daemon that exits on a full disk stopped watching."""
    cycles = []

    def poll():
        cycles.append(1)
        return _verdict(_alerts(x="alarm"))

    run_forever(poll, StateStore(tmp_path / "s.json"), [_explodes],
                WatchConfig(poll_s=0.0),
                stop=lambda: len(cycles) >= 3, sleep=lambda _: None)

    assert len(cycles) == 3


def test_check_mode_validates_without_polling(tmp_path, capsys):
    """`--check` is what you run after editing a config on a box you cannot watch."""
    from falsify_quant.watch import main

    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({
        "database": str(tmp_path / "trader.db"),
        "state": str(tmp_path / "s.json"),
        "market": "crypto-spot",
        "sinks": [{"kind": "log"}],
    }), encoding="utf-8")

    assert main([str(cfg), "--check"]) == 0
    assert "config ok" in capsys.readouterr().out


def test_a_config_missing_required_keys_says_which(tmp_path, capsys):
    from falsify_quant.watch import main

    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"market": "crypto-spot"}), encoding="utf-8")

    assert main([str(cfg), "--check"]) == 2
    err = capsys.readouterr().err
    assert "database" in err and "state" in err


def test_unreadable_config_exits_two(tmp_path, capsys):
    from falsify_quant.watch import main

    assert main([str(tmp_path / "nope.json"), "--check"]) == 2

    broken = tmp_path / "b.json"
    broken.write_text("{not json", encoding="utf-8")
    assert main([str(broken), "--check"]) == 2
    assert "cannot read" in capsys.readouterr().err


def test_example_flag_needs_no_config(capsys):
    from falsify_quant.watch import main

    assert main(["--example"]) == 0
    assert json.loads(capsys.readouterr().out)["market"] == "crypto-spot"


def test_no_arguments_at_all_explains_itself(capsys):
    from falsify_quant.watch import main

    with pytest.raises(SystemExit):
        main([])
    assert "--example" in capsys.readouterr().err
