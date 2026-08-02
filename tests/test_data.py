"""Failures in the data layer have to name the actual cause.

`--symbol` is one of the two ways anyone gets data in, and a fetch that comes back
empty is indistinguishable from a pair that does not exist unless the message says
otherwise. These pin the one case where the difference is a single character.
"""

from __future__ import annotations

import pytest

from falsify_quant.data import _quote_hint


@pytest.mark.parametrize("symbol,expected", [
    ("BTC-USDC", "BTC-USD"),
    ("eth-usdc", "ETH-USD"),
    ("SOL-USDT", "SOL-USD"),
    ("AAVE-USDC", "AAVE-USD"),
])
def test_stablecoin_quotes_name_the_usd_equivalent(symbol, expected):
    """Exchange treats these as the same market as USD and serves no history.

    The request succeeds and returns nothing, which reads as "this pair does not
    exist" when the fix is to drop one letter.
    """
    hint = _quote_hint(symbol)
    assert expected in hint
    assert "no history" in hint


def test_the_hint_does_not_fire_on_ordinary_pairs():
    """A genuinely missing product must not be blamed on its quote currency."""
    assert _quote_hint("BTC-USD") == ""
    assert _quote_hint("NOTREAL-USD") == ""
    assert _quote_hint("ETH-EUR") == ""
    assert _quote_hint("ETH-GBP") == ""


def test_it_does_not_fire_on_a_base_that_merely_contains_usdc():
    """Suffix match, not substring -- USDC-USD is a real book and quoted in USD."""
    assert _quote_hint("USDC-USD") == ""


def test_hint_is_appended_to_the_real_error(monkeypatch):
    """The hint supplements the count, it does not replace it.

    Losing "only got 0 candles" would hide whether the fetch failed or the product
    is simply young.
    """
    import falsify_quant.data as data

    monkeypatch.setattr(data, "_get", lambda *a, **k: b"[]")
    monkeypatch.setattr(data.time, "sleep", lambda *_: None)   # no rate-limit pause
    with pytest.raises(RuntimeError) as exc:
        data.load_crypto("BTC-USDC", interval="1h", bars=100)
    msg = str(exc.value)
    assert "only got 0 candles" in msg
    assert "BTC-USD" in msg
