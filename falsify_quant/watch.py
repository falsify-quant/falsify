"""A daemon that keeps running the monitor, and only speaks when something changes.

`monitor()` answers "is the bot doing what the strategy says, right now". Running it once
is a spot check. Running it every five minutes and printing the result is a spot check
nobody reads. The difference between those and something worth leaving on is entirely in
what it decides *not* to say.

Four rules, each of which exists because the alternative is an alert channel people mute:

**Only transitions notify.** A divergence that has been alarming for six hours is one
piece of news, not seventy-two. State is remembered between polls, and a notification goes
out when an alert appears, worsens, or clears -- never for a condition that is simply
still true.

**Recoveries are news too.** An alarm that stops firing because it resolved and one that
stops firing because the daemon died look identical from the outside. Clearing is
announced explicitly.

**Silence is a claim, so it has to be earned.** A watchdog that has crashed emits exactly
the same nothing as a watchdog with nothing to report. A heartbeat goes out on a fixed
schedule regardless of alert state, so the absence of one is itself the signal.

**Flapping is throttled, not hidden.** A check that toggles every poll is a broken check
or a boundary condition, and either way the useful message is "this flapped nine times in
an hour", once, rather than nine messages.

Nothing here writes to the trading database. The adapters open it read-only and this adds
no path that could change that.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from .monitor import SEVERITY, Alert, MonitorVerdict

__all__ = [
    "AlertState",
    "Event",
    "StateStore",
    "WatchConfig",
    "diff_alerts",
    "log_sink",
    "file_sink",
    "webhook_sink",
    "build_poll",
    "build_sinks",
    "run_forever",
    "run_once",
    "main",
]

_ORDER = {"ok": 0, "watch": 1, "alarm": 2}


# --------------------------------------------------------------------------------------
# What changed
# --------------------------------------------------------------------------------------


@dataclass
class AlertState:
    """What the daemon remembers about one check between polls."""

    severity: str = "ok"
    since: float = 0.0
    headline: str = ""
    transitions: int = 0
    last_notified: float = 0.0
    muted_until: float = 0.0

    def to_dict(self) -> dict:
        return {
            "severity": self.severity, "since": self.since, "headline": self.headline,
            "transitions": self.transitions, "last_notified": self.last_notified,
            "muted_until": self.muted_until,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AlertState":
        return cls(**{k: d[k] for k in
                      ("severity", "since", "headline", "transitions",
                       "last_notified", "muted_until") if k in d})


@dataclass
class Event:
    """Something worth telling a human about."""

    kind: str  # raised | worsened | cleared | heartbeat | flapping | error
    name: str
    severity: str
    headline: str
    detail: dict = field(default_factory=dict)
    at: float = 0.0

    @property
    def line(self) -> str:
        when = datetime.fromtimestamp(self.at or time.time(), timezone.utc)
        return (f"{when.isoformat(timespec='seconds')}  {self.kind.upper():<9} "
                f"{self.name:<20} {self.headline}")


def diff_alerts(
    previous: dict[str, AlertState],
    alerts: Iterable[Alert],
    *,
    now: float,
    flap_window_s: float = 3600.0,
    flap_limit: int = 4,
    mute_s: float = 3600.0,
) -> tuple[list[Event], dict[str, AlertState]]:
    """Compare this poll against the last one and decide what is worth saying.

    Returns the events to send and the state to remember. Pure -- no clock, no I/O, no
    globals -- because the interesting behaviour here is all in the edge cases, and edge
    cases in a daemon that owns its own clock are not testable.
    """
    events: list[Event] = []
    state: dict[str, AlertState] = {}

    for a in alerts:
        prev = previous.get(a.name, AlertState())
        cur = AlertState(
            severity=a.severity,
            since=prev.since if prev.severity == a.severity else now,
            headline=a.headline,
            transitions=prev.transitions,
            last_notified=prev.last_notified,
            muted_until=prev.muted_until,
        )

        changed = a.severity != prev.severity
        if changed:
            cur.transitions = prev.transitions + 1

            # Count only recent flapping. A check that toggled twice last month and twice
            # today is not flapping; one that toggled four times in an hour is, and the
            # useful message is that fact rather than each toggle.
            if (prev.since and now - prev.since < flap_window_s
                    and cur.transitions >= flap_limit
                    and now >= cur.muted_until):
                cur.muted_until = now + mute_s
                events.append(Event(
                    kind="flapping", name=a.name, severity=a.severity,
                    headline=(f"{a.name} changed state {cur.transitions} times; muting for "
                              f"{mute_s / 60:.0f} min. A check that cannot make up its mind "
                              f"is usually a threshold sitting on top of the data."),
                    detail={"transitions": cur.transitions}, at=now,
                ))
            elif now < cur.muted_until:
                pass  # inside a mute window: remember the change, say nothing
            else:
                if _ORDER[a.severity] == 0:
                    kind = "cleared"
                elif _ORDER[a.severity] > _ORDER[prev.severity]:
                    kind = "raised" if prev.severity == "ok" else "worsened"
                else:
                    kind = "cleared"  # alarm -> watch is still good news
                events.append(Event(kind=kind, name=a.name, severity=a.severity,
                                    headline=a.headline, detail=dict(a.detail), at=now))
                cur.last_notified = now

        state[a.name] = cur

    # A check that stopped being reported has, for the reader, stopped being a problem.
    # Saying so beats leaving a stale alarm in someone's memory of the system.
    for name, prev in previous.items():
        if name not in state and prev.severity != "ok":
            events.append(Event(
                kind="cleared", name=name, severity="ok",
                headline=f"{name} is no longer being checked", at=now))

    return events, state


# --------------------------------------------------------------------------------------
# Where events go
# --------------------------------------------------------------------------------------

Sink = Callable[[list[Event]], None]


def log_sink(stream=None) -> Sink:
    import sys

    out = stream if stream is not None else sys.stdout

    def send(events: list[Event]) -> None:
        for e in events:
            print(e.line, file=out, flush=True)

    return send


def file_sink(path: str | Path) -> Sink:
    """Append one JSON object per line. Survives restarts, greps cleanly."""
    p = Path(path)

    def send(events: list[Event]) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            for e in events:
                fh.write(json.dumps({
                    "at": e.at, "kind": e.kind, "name": e.name,
                    "severity": e.severity, "headline": e.headline, "detail": e.detail,
                }, default=str) + "\n")

    return send


def webhook_sink(url: str, *, timeout: float = 10.0, log=print) -> Sink:
    """POST a JSON batch. Deliberately dumb: no retries, no queue, no dependencies.

    A failed notification is logged and dropped rather than retried, because the
    alternative is a watchdog that spends its time managing a delivery queue instead of
    watching. The heartbeat is what catches a silently broken sink -- if notifications
    stop arriving, the missing heartbeat says so.
    """
    def send(events: list[Event]) -> None:
        if not events:
            return
        payload = json.dumps({"events": [
            {"at": e.at, "kind": e.kind, "name": e.name, "severity": e.severity,
             "headline": e.headline} for e in events]}).encode()
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=timeout).close()
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            log(f"webhook delivery failed ({type(exc).__name__}: {exc}); "
                f"{len(events)} event(s) dropped")

    return send


# --------------------------------------------------------------------------------------
# Remembering across restarts
# --------------------------------------------------------------------------------------


class StateStore:
    """Alert state on disk, so a restart does not re-announce everything.

    A daemon that forgets on restart turns every deploy into a wall of alerts, and an
    alert channel that shouts after every deploy is one people learn to ignore -- which
    costs more than the restart did.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, AlertState]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        alerts = raw.get("alerts", {})
        return {k: AlertState.from_dict(v) for k, v in alerts.items()
                if isinstance(v, dict)}

    def last_heartbeat(self) -> float:
        try:
            return float(json.loads(self.path.read_text(encoding="utf-8"))
                         .get("last_heartbeat", 0.0))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return 0.0

    def save(self, state: dict[str, AlertState], last_heartbeat: float) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({
            "alerts": {k: v.to_dict() for k, v in state.items()},
            "last_heartbeat": last_heartbeat,
            "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, indent=2), encoding="utf-8")
        tmp.replace(self.path)  # atomic: a killed daemon must not leave a truncated file


# --------------------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------------------


@dataclass
class WatchConfig:
    poll_s: float = 300.0
    heartbeat_s: float = 21600.0  # six hours
    flap_window_s: float = 3600.0
    flap_limit: int = 4
    mute_s: float = 3600.0


def _deliver(sink: Sink, events: list[Event]) -> None:
    """Hand a batch to one sink, and let that sink fail alone.

    The sinks are a list because the useful configuration is more than one: a local
    file to grep after the fact, and a webhook that reaches a person. Calling them in
    a bare loop couples them, and couples them the wrong way round -- a full disk
    under the file sink raised OSError before the webhook was ever called, so the
    local convenience took out the delivery that matters. It also skipped the state
    save below it and, `run_once` being unguarded in `run_forever`, ended the daemon.
    One log volume filling up should not be able to stop the watching.

    Dropped rather than retried, on purpose, and the reasoning is `webhook_sink`'s:
    a watchdog that manages a delivery queue is spending its time on something other
    than watching. The heartbeat is the backstop -- notifications that stop arriving
    show up as a missing heartbeat, which is the one signal that does not depend on
    any of this working.
    """
    try:
        sink(events)
    except Exception as exc:  # noqa: BLE001 -- a sink cannot be trusted with the loop
        _warn(lambda: f"sink {getattr(sink, '__name__', sink)!s} failed "
                      f"({type(exc).__name__}: {exc}); {len(events)} event(s) dropped")


def _warn(build: Callable[[], str]) -> None:
    """Say something went wrong, and never become the thing that goes wrong.

    Takes a callable rather than a string because building the message is itself
    part of what can fail: the subject is an object that has just misbehaved, and
    interpolating it calls its `__str__`. Formatting outside the guard moves the
    crash rather than preventing it.
    """
    import sys

    try:
        print(build(), file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 -- reporting a failure must not become one
        pass


def run_once(
    poll: Callable[[], MonitorVerdict],
    store: StateStore,
    sinks: list[Sink],
    config: WatchConfig,
    *,
    now: float | None = None,
) -> list[Event]:
    """One cycle: run the monitor, work out what changed, notify, remember.

    A failure inside `poll` becomes an event rather than an exception. The point of a
    watchdog is to survive the thing it is watching, and a daemon that exits when the
    trading database is briefly locked has chosen the wrong moment to stop watching.
    """
    now = time.time() if now is None else now
    previous = store.load()

    try:
        verdict = poll()
        events, state = diff_alerts(
            previous, verdict.alerts, now=now,
            flap_window_s=config.flap_window_s, flap_limit=config.flap_limit,
            mute_s=config.mute_s,
        )
        worst = max((a.severity for a in verdict.alerts), key=lambda s: _ORDER[s],
                    default="ok")
    except Exception as exc:  # noqa: BLE001 -- the watchdog outlives what it watches
        events = [Event(kind="error", name="monitor", severity="alarm",
                        headline=f"the monitor itself failed: {type(exc).__name__}: {exc}",
                        at=now)]
        state = previous
        worst = "alarm"

    # A first run always beats. "The watchdog is up" is the single most useful thing it
    # ever says, and leaving it to fall out of `now - 0 >= interval` means it works only
    # because the unix epoch is far away -- true in production, and quietly false anywhere
    # the clock is controlled, which is every test and every replay.
    last_beat = store.last_heartbeat()
    first_run = last_beat == 0.0
    if first_run or now - last_beat >= config.heartbeat_s:
        events.append(Event(
            kind="heartbeat", name="watch", severity="ok",
            headline=(f"{'started watching' if first_run else 'still watching'}; "
                      f"{len(state)} check(s), worst is {worst}"),
            detail={"checks": len(state), "worst": worst, "first": first_run}, at=now))
        last_beat = now

    if events:
        for sink in sinks:
            _deliver(sink, events)

    # Same reasoning as `_deliver`, one volume further out. If the state cannot be
    # written the daemon carries on without a memory: every standing alert is announced
    # again next cycle, which is noisy, and the noise is the correct signal -- an
    # operator seeing the same alarm every five minutes goes and looks. A daemon that
    # exited here would produce silence instead, and silence is what this whole module
    # exists to stop being ambiguous.
    try:
        store.save(state, last_beat)
    except Exception as exc:  # noqa: BLE001 -- outliving the disk is the job
        _warn(lambda: f"could not save watch state to {store.path} "
                      f"({type(exc).__name__}: {exc}); "
                      f"alerts will repeat until this is fixed")

    return events


def run_forever(
    poll: Callable[[], MonitorVerdict],
    store: StateStore,
    sinks: list[Sink],
    config: WatchConfig | None = None,
    *,
    stop: Callable[[], bool] | None = None,
    sleep=time.sleep,
) -> None:
    """Poll on a schedule until told to stop.

    `sleep` and `stop` are injected so the loop can be driven deterministically in tests.
    A scheduler that only runs in real time only gets tested in real time, which means it
    does not get tested.
    """
    cfg = config or WatchConfig()
    while not (stop and stop()):
        run_once(poll, store, sinks, cfg)
        if stop and stop():
            break
        sleep(cfg.poll_s)


# --------------------------------------------------------------------------------------
# Running it from a config file
# --------------------------------------------------------------------------------------

# Illustrative values only. The whole reason this daemon is configured from a file rather
# than from flags is that a live deployment's paths, instruments and parameters should
# never end up in a repository -- so this example does not contain any.
CONFIG_EXAMPLE = """\
{
  "database": "/srv/bot/trader.db",
  "market": "crypto-spot",
  "state": "/var/lib/falsify/watch-state.json",
  "poll_seconds": 300,
  "heartbeat_seconds": 21600,
  "expected_interval_seconds": 3600,
  "since": "2025-01-01",
  "strategy": {
    "file": "/etc/falsify/live_strategy.py",
    "params": {"fast": 20, "slow": 100}
  },
  "symbols": {"BTC-USD": {"interval": "1h", "bars": 5000}},
  "sinks": [
    {"kind": "log"},
    {"kind": "file", "path": "/var/log/falsify/events.jsonl"},
    {"kind": "webhook", "url": "https://example.invalid/hook"}
  ]
}
"""


def _as_epoch(value) -> float | None:
    """Accept a date, a datetime or a unix timestamp. Dates are read as UTC midnight."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("Z", "+00:00")
    d = datetime.fromisoformat(text if "T" in text or " " in text else f"{text}T00:00:00")
    return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).timestamp()


def _required(spec: dict, key: str, kind: str) -> str:
    """A missing key here is a deploy-time typo, so it should read like one.

    The bare KeyError this replaces said `'url'` and nothing else, which is a poor
    thing to be handed by the tool whose job is telling you when something is wrong.
    """
    value = spec.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{kind} sink needs a {key!r}; got {spec!r}")
    return value.strip()


def build_sinks(specs: list[dict], log=print) -> list[Sink]:
    out: list[Sink] = []
    for s in specs or [{"kind": "log"}]:
        kind = s.get("kind")
        if kind == "log":
            out.append(log_sink())
        elif kind == "file":
            out.append(file_sink(_required(s, "path", "file")))
        elif kind == "webhook":
            url = _required(s, "url", "webhook")
            # Checked here rather than at delivery, because the failure mode is silent.
            # urllib raises ValueError on a scheme-less URL from inside the sink, where
            # it becomes one dropped batch per cycle -- an alert channel that has never
            # worked and says so only in a log nobody is reading, which is worse than
            # having configured no webhook at all.
            if not url.lower().startswith(("http://", "https://")):
                raise ValueError(
                    f"webhook url must start with http:// or https://, got {url!r}. "
                    "Without a scheme nothing is ever delivered.")
            out.append(webhook_sink(url, timeout=float(s.get("timeout", 10.0)), log=log))
        else:
            raise ValueError(f"unknown sink {kind!r}; have log, file, webhook")
    return out


def build_poll(cfg: dict) -> Callable[[], MonitorVerdict]:
    """Turn a config dict into the callable the loop polls.

    The bars are re-fetched on every poll rather than cached. A replay against stale
    prices compares the bot's live decisions to a strategy that cannot see the bars the
    bot was actually looking at, and reports the difference as divergence -- an alert that
    is entirely the monitor's own fault, arriving at exactly the moment the market moved.
    """
    from .adapters import load_bot_db
    from .monitor import monitor
    from .spec import PRESETS

    spec = PRESETS[cfg.get("market", "crypto-spot")]
    since = _as_epoch(cfg.get("since"))
    db = cfg["database"]

    strategy = None
    params = cfg.get("strategy", {}).get("params")
    if (path := cfg.get("strategy", {}).get("file")):
        from .cli import _load_module

        strategy = _load_module(Path(path)).strategy

    symbols: dict = cfg.get("symbols") or {}

    def poll() -> MonitorVerdict:
        record = load_bot_db(db, since=since)
        bars_by_symbol = None
        if strategy is not None and symbols:
            from .data import load

            bars_by_symbol = {
                sym: load(sym, asset_class=spec.asset_class,
                          interval=opt.get("interval", "1h"),
                          bars=int(opt.get("bars", 5000)))
                for sym, opt in symbols.items()
            }
        return monitor(
            record, spec, strategy=strategy, bars_by_symbol=bars_by_symbol,
            params=params, since=since,
            expected_interval_s=float(cfg.get("expected_interval_seconds", 3600.0)),
        )

    return poll


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    from .cli import _utf8_console

    # A daemon that dies while printing an alert is the one failure a monitor cannot
    # have, and the alert text is prose with dashes in it. See `_utf8_console`.
    _utf8_console()

    p = argparse.ArgumentParser(
        prog="falsify-quant-watch",
        description="Keep running the live monitor and speak only when something changes.",
        epilog=f"example config:\n\n{CONFIG_EXAMPLE}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("config", type=Path, nargs="?", help="JSON config file")
    p.add_argument("--once", action="store_true", help="one cycle, then exit (for cron)")
    p.add_argument("--check", action="store_true",
                   help="load the config and the strategy, then exit without polling")
    p.add_argument("--example", action="store_true", help="print an example config")
    args = p.parse_args(argv)

    if args.example:
        print(CONFIG_EXAMPLE)
        return 0
    if args.config is None:
        p.error("need a config file, or --example")

    try:
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read {args.config}: {exc}", file=sys.stderr)
        return 2

    missing = [k for k in ("database", "state") if not cfg.get(k)]
    if missing:
        print(f"{args.config} is missing: {', '.join(missing)}", file=sys.stderr)
        return 2

    try:
        poll = build_poll(cfg)
        sinks = build_sinks(cfg.get("sinks", []))
    except ValueError as exc:  # raised deliberately, with the explanation already in it
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 -- a bad config should explain itself
        print(f"config error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.check:
        print(f"config ok: watching {cfg['database']}, "
              f"{len(cfg.get('symbols') or {})} symbol(s), {len(sinks)} sink(s)")
        return 0

    store = StateStore(cfg["state"])
    watch_cfg = WatchConfig(
        poll_s=float(cfg.get("poll_seconds", 300.0)),
        heartbeat_s=float(cfg.get("heartbeat_seconds", 21600.0)),
    )

    if args.once:
        run_once(poll, store, sinks, watch_cfg)
        return 0

    try:
        run_forever(poll, store, sinks, watch_cfg)
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
