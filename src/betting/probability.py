"""
Turning a passing-yards prediction into over / under / push probabilities.

The models predict a number of yards. To bet, we need P(actual > line). This
module gets it from how wrong the model has been before: its residuals
(actual - predicted) on games it didn't train on, saved by scripts/train.py.

    P(over) = share of past misses where  prediction + miss > line

That uses the model's real error shape (skew, fat tails) instead of assuming
a bell curve. If no residuals are available (older models file), it falls
back to a normal distribution with a given standard deviation.

Passing yards are whole numbers, so a whole-number line (e.g. 226.0) can push;
a half-yard line (226.5) can't. Both methods return a push probability.

This is the place to plug in a calibrator later (e.g. isotonic regression on
backtest outcomes): anything with a .probabilities(prediction, line) method
works with src/betting/strategy.py.
"""

from dataclasses import dataclass
from math import erf, floor, ceil, sqrt

import numpy as np

# Used only when a models file has no saved residuals
DEFAULT_SD = 80.0
MIN_RESIDUALS = 50


def normal_cdf(z: float) -> float:
    return 0.5 * (1 + erf(z / sqrt(2)))


@dataclass(frozen=True)
class Probabilities:
    over: float
    under: float
    push: float


class ResidualModel:
    """Over/under/push probabilities from a model's past misses.

    Args:
        residuals: actual - predicted yards on held-out games. With fewer than
            MIN_RESIDUALS, the normal fallback is used instead.
        sd: standard deviation for the normal fallback (defaults to the
            residuals' own standard deviation, else DEFAULT_SD).
    """

    def __init__(self, residuals=None, sd: float | None = None):
        r = np.asarray(residuals if residuals is not None else [], dtype=float)
        r = r[np.isfinite(r)]
        self.residuals = np.sort(r) if len(r) >= MIN_RESIDUALS else None
        if sd is not None:
            self.sd = float(sd)
        elif len(r) >= 2:
            self.sd = float(np.std(r, ddof=1))
        else:
            self.sd = DEFAULT_SD

    @property
    def method(self) -> str:
        if self.residuals is not None:
            return f"empirical ({len(self.residuals):,} held-out misses)"
        return f"normal (sd {self.sd:.0f} yds)"

    def probabilities(self, prediction: float, line: float) -> Probabilities:
        if self.residuals is not None:
            # actual = prediction + residual; actuals are whole yards, so round
            outcomes = np.round(prediction + self.residuals)
            n = len(outcomes)
            over = np.count_nonzero(outcomes > line) / n
            under = np.count_nonzero(outcomes < line) / n
            return Probabilities(float(over), float(under), max(0.0, 1 - over - under))

        # Normal, with a continuity correction for whole-yard outcomes:
        # over wins at floor(line)+1 yards or more, under at ceil(line)-1 or fewer.
        over = 1 - normal_cdf((floor(line) + 0.5 - prediction) / self.sd)
        under = normal_cdf((ceil(line) - 0.5 - prediction) / self.sd)
        return Probabilities(over, under, max(0.0, 1 - over - under))


def from_bundle(bundle: dict, model_name: str = "xgboost") -> ResidualModel:
    """The probability model for one trained model in models/qb_models.pkl."""
    residuals = (bundle or {}).get("residuals", {}).get(model_name)
    return ResidualModel(residuals)
