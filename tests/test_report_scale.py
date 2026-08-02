"""The figures printed on the equity curve heading.

The chart was previously unlabelled, so a reader could not tell +2% from +200%,
and the only return figure in the document sat inside a collapsed toggle under
the *costs* finding. These pin the arithmetic, because a number printed next to
a chart is one people quote.
"""

from __future__ import annotations

import numpy as np

from falsify_quant.report import _curve_scale


def equity(net):
    return list((1.0 + np.asarray(net, dtype=float)).cumprod())


def test_compounded_not_summed():
    """The two differ by volatility drag and the chart must show the compounded one.

    +10% then -10% sums to zero but compounds to -1%. The curve ends at 0.99, so
    the label has to say -1.0%; anything else is describing a different series
    from the one drawn.
    """
    out = _curve_scale(equity([0.10, -0.10]))
    assert "-1.0% compounded" in out
    assert "total" not in out          # 'total' was ambiguous against the summed rows


def test_first_bar_is_not_discarded():
    """Dividing by equity[0] silently drops the first bar's contribution.

    A single +50% bar must read +50%, not 0%.
    """
    assert "+50.0% compounded" in _curve_scale(equity([0.5]))


def test_worst_drawdown_is_peak_to_trough():
    # up 100%, then down 50% from that peak, then partial recovery
    out = _curve_scale(equity([1.0, -0.5, 0.2]))
    assert "-50.0% worst drawdown" in out


def test_flat_series():
    out = _curve_scale(equity([0.0, 0.0, 0.0]))
    assert "+0.0% compounded" in out
    assert "0.0% worst drawdown" in out


def test_empty_is_omitted_rather_than_zero():
    assert _curve_scale([]) == ""


def test_a_losing_curve_reads_negative():
    out = _curve_scale(equity([-0.2, -0.2]))
    assert "-36.0% compounded" in out      # 0.8 * 0.8 = 0.64
