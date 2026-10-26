"""
American odds math: payouts, implied probabilities and the bookmaker's cut.

Every function accepts a single number or a numpy array / pandas Series, so
the same code prices one bet in the dashboard or thousands in a backtest.

American odds refresher:
    -110  -> risk 110 to win 100 (profit 0.909 per unit staked)
    +150  -> risk 100 to win 150 (profit 1.5 per unit staked)
"""

import numpy as np


def _arr(price):
    return np.asarray(price, dtype=float)


def _same_shape(result, price):
    """Return a plain float for scalar input, an array otherwise."""
    return float(result) if np.ndim(price) == 0 else result


def payout(price):
    """Profit on a winning 1-unit bet (the 'b' in the Kelly formula)."""
    p = _arr(price)
    if np.any(np.abs(p) < 100):
        raise ValueError(f"American odds must be <= -100 or >= +100, got {price}")
    return _same_shape(np.where(p > 0, p / 100, 100 / np.abs(p)), price)


def to_decimal(price):
    """Decimal odds: total return per unit staked, including the stake."""
    return _same_shape(_arr(payout(price)) + 1, price)


def from_decimal(decimal_odds):
    """American odds from decimal odds (2.5 -> +150, 1.909 -> -110)."""
    d = _arr(decimal_odds)
    if np.any(d <= 1):
        raise ValueError("Decimal odds must be greater than 1")
    return _same_shape(np.where(d >= 2, (d - 1) * 100, -100 / (d - 1)), decimal_odds)


def implied_probability(price):
    """Break-even win rate the price implies, with the book's cut still in it."""
    return _same_shape(1 / _arr(to_decimal(price)), price)


def overround(over_price, under_price):
    """How far the two sides' implied probabilities add up past 100%.
    -110 / -110 -> 0.0476 (4.76%). This is the bookmaker's margin."""
    return implied_probability(over_price) + implied_probability(under_price) - 1


def no_vig_probabilities(over_price, under_price):
    """The market's fair over/under probabilities with the cut removed
    (proportional method: scale both implied probabilities to sum to 1).
    Useful as a sanity check: the model disagreeing hugely with the market
    is more often a model problem than a free lunch."""
    p_over = implied_probability(over_price)
    p_under = implied_probability(under_price)
    total = p_over + p_under
    return p_over / total, p_under / total
