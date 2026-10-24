"""
Betting strategy: line shopping, EV filtering, Kelly sizing and a bankroll
simulation, built on odds.py, probability.py and kelly.py.

    from src.betting import BetConfig, best_bet, from_bundle
    prob = from_bundle(bundle)                      # how wrong the model has been
    bet = best_bet(books, prediction, prob, BetConfig())

`books` is one row per sportsbook with line, over_odds, under_odds (and
optionally sportsbook). The best bet is the side and book with the highest
expected value; it's flagged `qualifies` only if it clears the config's
minimum EV and minimum edge.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .kelly import expected_value, kelly_fraction
from .odds import no_vig_probabilities, payout


@dataclass
class BetConfig:
    min_ev: float = 0.03            # skip bets expected to return < 3% per unit
    min_edge_yards: float = 0.0     # and bets where |prediction - line| is smaller than this
    kelly_multiplier: float = 0.25  # quarter Kelly
    max_stake: float = 0.05         # never more than 5% of bankroll on one bet
    max_daily_exposure: float = 0.30  # all of one day's bets together: at most 30%

    def as_dict(self):
        return asdict(self)


def evaluate_side(prediction, line, price, side, prob_model, config=None):
    """EV and Kelly stake for one side ("OVER" or "UNDER") of one line."""
    config = config or BetConfig()
    p = prob_model.probabilities(prediction, line)
    p_win, p_lose = (p.over, p.under) if side == "OVER" else (p.under, p.over)
    full = kelly_fraction(p_win, p_lose, price)
    ev = expected_value(p_win, p_lose, price)
    edge = prediction - line
    agrees = edge > 0 if side == "OVER" else edge < 0
    qualifies = bool(ev >= config.min_ev and agrees and abs(edge) >= config.min_edge_yards)
    return {
        "side": side, "line": float(line), "price": float(price),
        "p_win": p_win, "p_lose": p_lose, "p_push": p.push,
        "ev": ev, "kelly": full,
        "stake": min(config.max_stake, config.kelly_multiplier * full) if qualifies else 0.0,
        "qualifies": qualifies,
    }


def best_bet(books: pd.DataFrame, prediction: float, prob_model, config=None):
    """Line shopping: the highest-EV bet across every book, on the side the
    prediction points to at that book's line (over if the prediction is above
    it, under if below). None if there are no usable lines."""
    if books is None or len(books) == 0 or prediction is None or not np.isfinite(prediction):
        return None
    best = None
    for row in books.itertuples(index=False):
        if prediction == row.line:
            continue
        side = "OVER" if prediction > row.line else "UNDER"
        price = row.over_odds if side == "OVER" else row.under_odds
        if price is None or not np.isfinite(price):
            continue
        bet = evaluate_side(prediction, row.line, price, side, prob_model, config)
        if best is None or bet["ev"] > best["ev"]:
            fair_over, fair_under = no_vig_probabilities(row.over_odds, row.under_odds)
            bet["book"] = getattr(row, "sportsbook", None)
            bet["market_p_win"] = fair_over if side == "OVER" else fair_under
            best = bet
    return best


def cap_exposure(fractions, max_total: float):
    """Scale a set of simultaneous stakes (bankroll fractions) down together
    so they add up to at most max_total. Leaves them alone if already under."""
    f = np.asarray(fractions, dtype=float)
    total = f.sum()
    return f * (max_total / total) if total > max_total else f


def simulate_bankroll(bets: pd.DataFrame, prediction_col: str, prob_model, config=None,
                      start: float = 100.0):
    """
    Replay bets in date order with fractional Kelly sizing.

    `bets` needs game_date, line, over_odds, under_odds, passing_yards (actual)
    and the prediction column. Stakes for all games on one date are sized from
    that morning's bankroll (they're placed at the same time), then scaled down
    together if they'd exceed max_daily_exposure.

    Returns (log, summary): one log row per bet placed, and totals.
    """
    config = config or BetConfig()
    bankroll = start
    peak, max_drawdown = start, 0.0
    log = []

    df = bets.sort_values("game_date")
    for date, day in df.groupby("game_date", sort=True):
        picks = []
        for row in day.itertuples(index=False):
            pred = getattr(row, prediction_col)
            if not np.isfinite(pred):
                continue
            for side, price in (("OVER", row.over_odds), ("UNDER", row.under_odds)):
                bet = evaluate_side(pred, row.line, price, side, prob_model, config)
                if bet["qualifies"]:
                    picks.append((row, bet))
        if not picks:
            continue

        fractions = cap_exposure([b["stake"] for _, b in picks], config.max_daily_exposure)
        day_start = bankroll
        for (row, bet), frac in zip(picks, fractions):
            stake = day_start * frac
            actual = row.passing_yards
            if actual == row.line:
                result, profit = "push", 0.0
            elif (actual > row.line) == (bet["side"] == "OVER"):
                result, profit = "win", stake * payout(bet["price"])
            else:
                result, profit = "loss", -stake
            bankroll += profit
            log.append({
                "game_date": date, "game_id": row.game_id, "player_id": row.player_id,
                "side": bet["side"], "line": row.line, "price": bet["price"],
                "prediction": pred,
                "actual": actual, "p_win": bet["p_win"], "ev": bet["ev"],
                "stake_fraction": frac, "stake": stake, "result": result,
                "profit": profit, "bankroll": bankroll,
            })
        peak = max(peak, bankroll)
        max_drawdown = max(max_drawdown, (peak - bankroll) / peak)

    log = pd.DataFrame(log)
    staked = float(log["stake"].sum()) if len(log) else 0.0
    profit = bankroll - start
    decided = log["result"].isin(["win", "loss"]).sum() if len(log) else 0
    wins = int((log["result"] == "win").sum()) if len(log) else 0
    summary = {
        "bets": len(log), "wins": wins,
        "losses": int((log["result"] == "loss").sum()) if len(log) else 0,
        "pushes": int((log["result"] == "push").sum()) if len(log) else 0,
        "win_rate": wins / decided if decided else 0.0,
        "staked": staked, "profit": profit,
        "roi": profit / staked if staked else 0.0,       # return on money risked
        "start": start, "final": bankroll,
        "growth": bankroll / start - 1,                   # return on bankroll
        "max_drawdown": max_drawdown,
    }
    return log, summary
