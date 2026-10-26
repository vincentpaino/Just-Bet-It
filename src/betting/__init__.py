"""
Betting math for Just Bet It (Doc II, Section 6): odds conversions, model
probabilities, expected value, Kelly sizing, line shopping and a bankroll
simulation. Used by scripts/predict_upcoming.py, scripts/backtest.py and app.py.
"""

from .kelly import expected_value, kelly_fraction, stake_fraction
from .odds import (
    from_decimal, implied_probability, no_vig_probabilities, overround, payout, to_decimal,
)
from .probability import DEFAULT_SD, Probabilities, ResidualModel, from_bundle, normal_cdf
from .strategy import BetConfig, best_bet, cap_exposure, evaluate_side, simulate_bankroll

__all__ = [
    "BetConfig", "DEFAULT_SD", "cap_exposure", "Probabilities", "ResidualModel", "best_bet", "evaluate_side",
    "expected_value", "from_bundle", "from_decimal", "implied_probability", "kelly_fraction",
    "no_vig_probabilities", "normal_cdf", "overround", "payout", "simulate_bankroll",
    "stake_fraction", "to_decimal",
]
