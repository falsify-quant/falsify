"""Attestations, tested from the forger's side.

Every test that matters here is an attack. A tamper-evident document is only worth
carrying if the obvious edits are caught, so the suite works through them in order of how
cheap they are to attempt: raise the score, raise a finding, delete the finding that
failed, swap the weights, backdate the whole thing.

The last one is not caught, and there is a test asserting that it is not. An attestation
cannot vouch for its own date, and a suite that quietly implied otherwise would be worse
than no suite.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from falsify_quant.attest import (
    SCHEMA_VERSION,
    Attestation,
    attest,
    canonical_bytes,
    content_hash,
    fingerprint_array,
    fingerprint_bytes,
    fingerprint_text,
    read_attestation,
    verify,
    write_attestation,
)
from falsify_quant.prosecute import Finding
from falsify_quant.score import WEIGHTS, score_findings


def _verdict(scores=None):
    scores = scores or {"causality": 1.0, "costs": 0.8, "deflation": 0.7,
                        "pbo": 0.6, "permutation": 0.5, "regime": 0.9}
    findings = [
        Finding(name=n, title=n.title(), score=s, headline=f"{n} said {s}",
                fatal=(n == "causality"),
                detail={"stat": s * 2, "n": 17, "series": np.arange(3)})
        for n, s in scores.items()
    ]
    v = score_findings(findings, meta={
        "symbol": "SPY", "market": "US equity", "asset_class": "equity",
        "bars": 5000, "bars_per_year": 252.0, "years": 19.8,
        "params": {"fast": 50, "slow": 200},
        "grid": {"fast": [10, 50], "slow": [100, 200]},
        "n_trials": 4, "sharpe_annual": 0.71,
        "equity": (1.0 + np.full(500, 0.0004)).cumprod().tolist(),
        "ts": (1.6e9 + np.arange(500) * 86400.0).tolist(),
    })
    return v


def _tamper(att: Attestation, mutate, *, rehash: bool) -> Attestation:
    """Produce a forged document. `rehash` is the difference between a lazy forger
    and a competent one."""
    body = json.loads(json.dumps(att.body))
    mutate(body)
    return Attestation(body=body,
                       content_hash=content_hash(body) if rehash else att.content_hash)


# --------------------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------------------


def test_a_fresh_attestation_verifies():
    r = verify(attest(_verdict()))
    assert r.ok
    assert [c.name for c in r.failures] == []


def test_the_only_caveat_on_an_unanchored_document_is_the_date():
    r = verify(attest(_verdict()))
    assert {c.name for c in r.warnings} == {"anchor"}
    assert "self-reported" in next(c for c in r.warnings if c.name == "anchor").detail


def test_an_anchored_document_has_no_caveats():
    att = attest(_verdict(), anchor={"kind": "git", "ref": "https://example.com/c/abc"})
    r = verify(att)
    assert r.ok and not r.warnings
    assert r.summary() == "INTACT"


def test_the_strategy_is_fingerprinted_not_embedded():
    """Attesting must not require publishing the strategy."""
    source = "def strategy(bars, fast=50):\n    return bars.close * 0 + 1\n"
    att = attest(_verdict(), strategy_source=source)

    assert att.body["evidence"]["strategy_fingerprint"] == fingerprint_text(source)
    assert "def strategy" not in att.to_json()


def test_the_equity_curve_is_hashed_not_carried():
    """Keeps the document small and still binds it to a specific track record."""
    att = attest(_verdict())
    assert att.body["evidence"]["equity_fingerprint"]
    assert "equity" not in att.body["subject"]
    assert len(att.to_json()) < 8000


# --------------------------------------------------------------------------------------
# Forgeries
# --------------------------------------------------------------------------------------


def test_raising_the_score_breaks_the_hash():
    def raise_it(b):
        b["verdict"]["score"] = 95.0

    r = verify(_tamper(attest(_verdict()), raise_it, rehash=False))
    assert not r.ok
    assert "content hash" in {c.name for c in r.failures}


def test_raising_the_score_and_rehashing_breaks_the_arithmetic():
    """The check that does the work: the headline is recomputed from the findings."""
    def raise_it(b):
        b["verdict"]["score"] = 95.0
        b["verdict"]["label"] = "SURVIVED"

    r = verify(_tamper(attest(_verdict()), raise_it, rehash=True))
    assert not r.ok
    assert "score arithmetic" in {c.name for c in r.failures}
    assert "content hash" not in {c.name for c in r.failures}  # the rehash worked


def test_raising_one_finding_and_rehashing_still_breaks_the_arithmetic():
    def flatter(b):
        for f in b["findings"]:
            if f["name"] == "costs":
                f["score"] = 1.0

    r = verify(_tamper(attest(_verdict()), flatter, rehash=True))
    assert not r.ok
    assert "score arithmetic" in {c.name for c in r.failures}


def test_deleting_the_failing_check_is_caught_by_coverage():
    """The cheapest forgery: drop the bad news and recompute over what is left.

    The arithmetic then agrees with itself -- the score really is the geometric mean of
    the findings that remain -- so only the *set* of checks gives it away.
    """
    honest = _verdict({"causality": 1.0, "costs": 0.02, "deflation": 0.8,
                       "pbo": 0.7, "permutation": 0.9, "regime": 0.9})

    def drop_costs(b):
        b["findings"] = [f for f in b["findings"] if f["name"] != "costs"]
        rebuilt = score_findings([
            Finding(name=f["name"], title=f["title"], score=f["score"],
                    headline=f["headline"], fatal=f["fatal"])
            for f in b["findings"]])
        b["verdict"]["score"] = rebuilt.score
        b["verdict"]["label"] = rebuilt.label

    forged = _tamper(attest(honest), drop_costs, rehash=True)
    r = verify(forged)

    # Deleting one line turns LIKELY OVERFIT into SURVIVED. This is the forgery worth
    # catching, and the arithmetic alone cannot: it is a correct mean of what remains.
    truth = attest(honest)
    assert truth.label == "LIKELY OVERFIT" and forged.label == "SURVIVED"
    assert forged.score - truth.score > 40
    assert not r.ok
    assert "coverage" in {c.name for c in r.failures}
    assert "score arithmetic" not in {c.name for c in r.failures}  # internally consistent


def test_un_breaking_a_broken_verdict_is_caught():
    """Deleting the lookahead finding is one forgery; passing it is the other.

    Coverage catches the deletion, so the forger's next move is to leave the gate in
    place and mark it clean. That is the single most valuable edit in the document --
    it turns a 0.00 BROKEN into a respectable number -- and it is caught only because
    the headline is recomputed from the parts rather than read off the top.
    """
    broken = attest(_verdict({"causality": 0.0, "costs": 0.9, "deflation": 0.9,
                              "pbo": 0.9, "permutation": 0.9, "regime": 0.9}))
    assert broken.score == 0.0 and broken.body["verdict"]["broken"] is True

    def pass_the_gate(b):
        for f in b["findings"]:
            if f["name"] == "causality":
                f["score"], f["fatal"] = 1.0, False

    r = verify(_tamper(broken, pass_the_gate, rehash=True))
    assert not r.ok
    assert "score arithmetic" in {c.name for c in r.failures}


def test_deleting_the_causality_gate_is_called_out_separately():
    """Causality is a gate, not a weighted term, so coverage does not cover it.

    It gets its own check for exactly that reason -- a lookahead failure is the single
    most attractive finding to make disappear, and it is the one that is not in the
    weighting for the machine to miss.
    """
    def drop_gate(b):
        b["findings"] = [f for f in b["findings"] if f["name"] != "causality"]

    r = verify(_tamper(attest(_verdict()), drop_gate, rehash=True))
    assert not r.ok
    assert "causality gate" in {c.name for c in r.failures}
    assert "coverage" not in {c.name for c in r.failures}  # it is not a weighted term


def test_swapping_the_weights_is_flagged():
    def reweight(b):
        b["weights"] = {k: 1.0 for k in b["weights"]}

    r = verify(_tamper(attest(_verdict()), reweight, rehash=True))
    assert "weights" in {c.name for c in r.warnings}


def test_editing_the_subject_breaks_the_hash():
    """Everything is under the hash, not just the verdict."""
    for edit in ({"symbol": "NVDA"}, {"years": 99.0}, {"n_trials": 1}):
        r = verify(_tamper(attest(_verdict()),
                           lambda b, e=edit: b["subject"].update(e), rehash=False))
        assert not r.ok, f"editing {edit} went unnoticed"


def test_backdating_is_not_detectable_and_the_result_says_so():
    """The limit, asserted rather than implied.

    An attestation carries whatever date its author typed. Nothing inside the document
    can contradict it, which is exactly why `verify` reports an unanchored document as a
    caveat instead of passing it silently.
    """
    def backdate(b):
        b["created_utc"] = "2019-01-01T00:00:00+00:00"

    forged = _tamper(attest(_verdict()), backdate, rehash=True)
    r = verify(forged)

    assert r.ok  # nothing here can catch it
    assert forged.created_utc.startswith("2019")
    anchor = next(c for c in r.warnings if c.name == "anchor")
    assert "proves nothing about when" in anchor.detail


# --------------------------------------------------------------------------------------
# Canonical form
# --------------------------------------------------------------------------------------


def test_the_hash_does_not_depend_on_key_order():
    a = {"z": 1, "a": {"y": 2, "b": 3}}
    b = {"a": {"b": 3, "y": 2}, "z": 1}
    assert content_hash(a) == content_hash(b)


def test_the_same_analysis_hashes_the_same_twice():
    v = _verdict()
    when = "2026-07-27T12:00:00+00:00"
    assert attest(v, created_utc=when).content_hash == attest(v, created_utc=when).content_hash


def test_findings_are_ordered_so_the_hash_does_not_depend_on_run_order():
    forward = {"causality": 1.0, "costs": 0.8, "deflation": 0.7,
               "pbo": 0.6, "permutation": 0.5, "regime": 0.9}
    backward = dict(reversed(list(forward.items())))
    when = "2026-07-27T12:00:00+00:00"
    assert (attest(_verdict(forward), created_utc=when).content_hash
            == attest(_verdict(backward), created_utc=when).content_hash)


def test_negative_zero_hashes_the_same_as_zero():
    """They compare equal and no statistic here distinguishes them, so two identical
    analyses must not produce two hashes."""
    assert content_hash({"x": -0.0}) == content_hash({"x": 0.0})


def test_non_finite_numbers_become_null_not_an_unparseable_token():
    raw = canonical_bytes({"a": float("nan"), "b": float("inf"), "c": 1.5}).decode()
    assert "NaN" not in raw and "Infinity" not in raw
    assert json.loads(raw) == {"a": None, "b": None, "c": 1.5}


def test_numpy_types_survive_canonicalisation():
    body = {"i": np.int64(3), "f": np.float32(0.5), "b": np.bool_(True),
            "a": np.arange(3), "n": np.float64("nan")}
    assert json.loads(canonical_bytes(body)) == {
        "i": 3, "f": 0.5, "b": True, "a": [0, 1, 2], "n": None}


def test_findings_carrying_numpy_detail_do_not_break_attestation():
    att = attest(_verdict())
    assert json.loads(att.to_json())["body"]["findings"][0]["detail"]["series"] == [0, 1, 2]


def test_source_fingerprint_ignores_line_endings():
    """The same file must not attest to two values depending on which machine it was on."""
    assert fingerprint_text("a\r\nb\r\n") == fingerprint_text("a\nb\n")
    assert fingerprint_text("a\rb\r") == fingerprint_text("a\nb\n")
    assert fingerprint_text("a\nb\n") != fingerprint_text("a\nb\nc\n")


def test_array_fingerprint_is_sensitive_to_one_bit():
    x = np.linspace(0, 1, 500)
    y = x.copy()
    y[321] = np.nextafter(y[321], 1.0)
    assert fingerprint_array(x) != fingerprint_array(y)
    assert len(fingerprint_array(x)) == 32


# --------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------


def test_write_then_read_round_trips(tmp_path):
    att = attest(_verdict(), note="quarterly review",
                 anchor={"kind": "git", "ref": "https://example.com/c/abc"})
    p = write_attestation(att, tmp_path / "a.json")

    back = read_attestation(p)
    assert back.content_hash == att.content_hash
    assert back.body == att.body
    assert verify(back).ok
    assert back.anchor["kind"] == "git"


def test_reading_keeps_the_written_hash_rather_than_recomputing_it(tmp_path):
    """Recomputing on load would make every document verify, including forged ones."""
    att = attest(_verdict())
    p = write_attestation(att, tmp_path / "a.json")

    raw = json.loads(p.read_text(encoding="utf-8"))
    raw["body"]["verdict"]["score"] = 99.0
    p.write_text(json.dumps(raw), encoding="utf-8")

    assert not verify(read_attestation(p)).ok


def test_reading_a_foreign_file_fails_clearly(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a falsify attestation"):
        read_attestation(p)


def test_reading_a_future_schema_fails_clearly(tmp_path):
    att = attest(_verdict())
    p = tmp_path / "a.json"
    raw = json.loads(att.to_json())
    raw["falsify_attestation"] = SCHEMA_VERSION + 1
    p.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="schema"):
        read_attestation(p)


def test_the_document_is_human_readable(tmp_path):
    """A credential nobody can read without tooling is not much of a credential."""
    text = write_attestation(attest(_verdict()), tmp_path / "a.json").read_text("utf-8")
    for token in ["falsify_attestation", "content_hash", "SPY", "causality", "score"]:
        assert token in text
    assert text.endswith("\n")


# --------------------------------------------------------------------------------------
# Encoding
#
# These exist because a real attestation of a real strategy carried three non-ASCII
# bytes -- one em-dash in a finding headline -- and that was enough for
# `json.load(open(path))` on a Windows box to read a different body and report the
# document TAMPERED. Calling someone a forger because their platform defaulted to
# cp1252 is the worst thing this module can do, so the file is now pure ASCII and
# these pin it there.
# --------------------------------------------------------------------------------------


def _non_ascii_verdict():
    """A verdict whose headlines carry the characters the report actually emits."""
    findings = [
        Finding(name=n, title=n.title(), score=s,
                headline=f"{n} — Sharpe 0.71 ± 0.09, cost drag ≈3 bps",
                fatal=(n == "causality"), detail={"stat": s})
        for n, s in {"causality": 1.0, "costs": 0.8, "deflation": 0.7,
                     "pbo": 0.6, "permutation": 0.5, "regime": 0.9}.items()
    ]
    return score_findings(findings, meta={
        "symbol": "SPY", "market": "US equity", "asset_class": "equity",
        "bars": 5000, "bars_per_year": 252.0, "years": 19.8,
        "params": {"fast": 50, "slow": 200}, "n_trials": 4, "sharpe_annual": 0.71,
    })


def test_the_written_file_is_pure_ascii(tmp_path):
    """Every encoding anyone might open this with has to agree on its bytes."""
    p = write_attestation(attest(_non_ascii_verdict()), tmp_path / "a.json")
    raw = p.read_bytes()
    assert not [b for b in raw if b > 127], "non-ASCII bytes make the hash reader-dependent"


@pytest.mark.parametrize("encoding", ["utf-8", "cp1252", "latin-1", "ascii"])
def test_it_verifies_whatever_encoding_the_reader_guesses(tmp_path, encoding):
    """A third party's `open()` default is not something this library gets to choose."""
    p = write_attestation(attest(_non_ascii_verdict()), tmp_path / "a.json")
    raw = json.loads(p.read_text(encoding=encoding))
    assert verify(Attestation(body=raw["body"],
                              content_hash=raw["content_hash"])).ok


def test_escaping_does_not_change_the_hash(tmp_path):
    """Serialisation is not canonicalisation -- the hash is over parsed values.

    If these ever diverged, every document written before the escaping change would
    stop verifying, which is a far more expensive failure than the one it fixed.
    """
    att = attest(_non_ascii_verdict())
    p = write_attestation(att, tmp_path / "a.json")
    assert read_attestation(p).content_hash == att.content_hash
    assert content_hash(json.loads(p.read_text("utf-8"))["body"]) == att.content_hash


def test_the_characters_are_preserved_not_stripped(tmp_path):
    """Escaping keeps the text. Dropping it would silently rewrite the evidence."""
    p = write_attestation(attest(_non_ascii_verdict()), tmp_path / "a.json")
    headline = json.loads(p.read_text("utf-8"))["body"]["findings"][0]["headline"]
    assert "—" in headline and "±" in headline and "≈" in headline


# --------------------------------------------------------------------------------------
# The strategy file's own encoding
#
# CPython imports a module whose comments are cp1252 without complaint, so the file
# runs, the whole analysis runs, and only then did attesting it fail. Failing at the
# end of a long job, on the flagship feature, over a quotation mark in a comment.
# --------------------------------------------------------------------------------------


_SMART_QUOTES = b"# a \x93smart quote\x94 comment \x97 from Notepad\n"


def test_a_strategy_that_is_not_utf8_can_still_be_attested(tmp_path):
    p = tmp_path / "s.py"
    p.write_bytes(b"def strategy(bars):\n    return None\n" + _SMART_QUOTES)

    att = attest(_verdict(), strategy_source=p)
    assert att.body["evidence"]["strategy_fingerprint"] == fingerprint_bytes(p.read_bytes())
    assert verify(att).ok


def test_the_byte_fingerprint_matches_the_text_one_for_utf8():
    """Backwards compatibility: documents written before this change must still match.

    If these two ever diverged, every previously attested strategy would look like a
    different strategy -- a far more expensive failure than the crash it replaced.
    """
    for text in ["def strategy(bars):\n    return None\n",
                 "# em dash — and pi π\nx = 1\n",
                 "", "a\r\nb\r\n", "a\rb\r"]:
        assert fingerprint_bytes(text.encode("utf-8")) == fingerprint_text(text)


def test_line_endings_are_normalised_at_the_byte_level_too():
    """The cross-machine guarantee has to survive reading the file as bytes."""
    assert fingerprint_bytes(b"a\r\nb\r\n") == fingerprint_bytes(b"a\nb\n")
    assert fingerprint_bytes(b"a\rb\r") == fingerprint_bytes(b"a\nb\n")
    assert fingerprint_bytes(b"a\nb\n") != fingerprint_bytes(b"a\nb\nc\n")


def test_reading_the_file_is_what_is_fingerprinted_not_its_name(tmp_path):
    """`--strategy` arrives as a Path; fingerprinting the path would attest nothing."""
    a, b = tmp_path / "one.py", tmp_path / "two.py"
    body = b"def strategy(bars):\n    return None\n"
    a.write_bytes(body)
    b.write_bytes(body)

    fp = lambda p: attest(_verdict(), strategy_source=p).body["evidence"][
        "strategy_fingerprint"]
    assert fp(a) == fp(b), "same code under two names must fingerprint the same"

    b.write_bytes(body + b"# changed\n")
    assert fp(a) != fp(b), "different code must not fingerprint the same"


def test_a_broken_verdict_still_attests():
    """A lookahead failure is exactly the verdict someone would want to suppress."""
    v = _verdict({"causality": 0.0, "costs": 0.9, "deflation": 0.9,
                  "pbo": 0.9, "permutation": 0.9, "regime": 0.9})
    att = attest(v)
    assert att.body["verdict"]["broken"] is True
    assert att.score == 0.0
    assert verify(att).ok


def test_weights_recorded_match_the_library():
    assert attest(_verdict()).body["weights"] == dict(WEIGHTS)


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------


def test_verify_exit_codes_are_scriptable(tmp_path, capsys):
    """The exit code is the whole interface for anyone wiring this into a pipeline."""
    from falsify_quant.cli import main

    good = write_attestation(attest(_verdict()), tmp_path / "good.json")
    assert main(["--verify", str(good)]) == 0

    raw = json.loads(good.read_text(encoding="utf-8"))
    raw["body"]["verdict"]["score"] = 99.0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    assert main(["--verify", str(bad)]) == 1

    missing = tmp_path / "nope.json"
    assert main(["--verify", str(missing)]) == 2

    out = capsys.readouterr().out
    assert "INTACT" in out and "TAMPERED" in out


def test_verify_needs_no_strategy_argument(tmp_path):
    """`falsify --verify file` has to work for someone who was only sent the document."""
    from falsify_quant.cli import main

    p = write_attestation(attest(_verdict()), tmp_path / "a.json")
    assert main(["--verify", str(p)]) == 0


def test_running_with_no_arguments_at_all_explains_itself(capsys):
    from falsify_quant.cli import main

    with pytest.raises(SystemExit):
        main([])
    assert "--verify" in capsys.readouterr().err


@pytest.mark.parametrize("text,want", [
    ("git=https://example.com/c/abc", {"kind": "git", "ref": "https://example.com/c/abc"}),
    (" chain = otsproof ", {"kind": "chain", "ref": "otsproof"}),
    (None, None),
])
def test_anchor_parsing(text, want):
    from falsify_quant.cli import _parse_anchor

    assert _parse_anchor(text, None) == want


def test_a_malformed_anchor_is_rejected_rather_than_guessed():
    import argparse

    from falsify_quant.cli import _parse_anchor

    with pytest.raises(SystemExit):
        _parse_anchor("just-a-url", argparse.ArgumentParser())
