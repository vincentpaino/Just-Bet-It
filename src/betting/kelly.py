"""
Expected value and Kelly Criterion bet sizing.

Kelly answers "what fraction of my bankroll maximizes long-run growth?"
For a bet that wins with probability p, loses with q, pushes otherwise, and
pays b units profit per unit staked, maximizing p*ln(1 + b*f) + q*ln(1 - f) gives

    f* = (b*p - q) / (b * (p + q))

(with no pushes, p + q = 1 and this is the textbook (b*p - q) / b).

Full Kelly assumes the probabilities are exactly right. Ours are estimates,
and overestimating your edge with full Kelly loses money fast, so the
standard practice is a fraction of it (quarter Kelly by default) plus a cap
on any single bet.
"""

from .odds import payout


def expected_value(p_win, p_lose, price):
    """Expected profit per 1 unit staked. A push returns the stake (0 profit)."""
    return p_win * payout(price) - p_lose


def kelly_fraction(p_win: float, p_lose: float, price: float) -> float:
    """Full-Kelly fraction of bankroll to stake. 0 when the bet has no edge."""
    b = payout(price)
    decided = p_win + p_lose
    if decided <= 0:
        return 0.0
    f = (b * p_win - p_lose) / (b * decided)
    return max(0.0, f)


def stake_fraction(p_win: float, p_lose: float, price: float,
                   multiplier: float = 0.25, cap: float = 0.05) -> float:
    """Fractional Kelly, capped: the share of bankroll to actually bet."""
    return min(cap, multiplier * kelly_fraction(p_win, p_lose, price))
