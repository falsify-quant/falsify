"""A moving-average crossover. The most-backtested idea in retail trading.

Run it against a real market:

    python -m falsify strategies/trend.py --symbol SPY     --market equity
    python -m falsify strategies/trend.py --symbol BTC-USD --market crypto-perp --interval 1h
"""

import numpy as np

from falsify_quant.examples import rolling_mean

GRID = {
    "fast": [5, 8, 12, 20, 30, 50],
    "slow": [60, 90, 120, 160, 200, 250],
}


def valid(p):
    return p["fast"] < p["slow"]


def strategy(bars, fast=20, slow=100):
    """Long above the slow mean, short below. Trailing windows only."""
    c = bars.close
    f = rolling_mean(c, int(fast))
    s = rolling_mean(c, int(slow))
    w = np.where(f > s, 1.0, -1.0)
    w[np.isnan(f) | np.isnan(s)] = np.nan  # warmup -> flat
    return w
