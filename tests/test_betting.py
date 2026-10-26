"""Tests for src/betting. Run from the project root: python -m pytest"""

import numpy as np
import pandas as pd
import pytest

from src.betting import (
    BetConfig, ResidualModel, best_bet, cap_exposure, evaluate_side, expected_value, from_bundle,
    from_decimal, implied_probability, kelly_fraction, no_vig_probabilities, overround,
    payout, simulate_bankroll, stake_fraction, to_decimal,
)


# ---------------------------------------------------------------- odds
@pytest.mark.parametrize("price, expected", [(-110, 100 / 110), (-200, 0.5), (100, 1.0), (150, 1.5)])
def test_payout(price, expected):
    assert payout(price) == pytest.approx(expected)


def test_payout_vectorized():
    np.testing.assert_allclose(payout(np.array([-110, 150])), [100 / 110, 1.5])


def test_payout_rejects_impossible_odds():
    with pytest.raises(ValueError):
        payout(-50)


def test_decimal_round_trip():
    for price in (-250, -110, 100, 120, 300):
        assert from_decimal(to_decimal(price)) == pytest.approx(price)


def test_implied_probability_and_overround():
    assert implied_probability(-110) == pytest.approx(110 / 210)
    assert implied_probability(100) == pytest.approx(0.5)
    assert overround(-110, -110) == pytest.approx(220 / 210 - 1)


def test_no_vig_sums_to_one_and_favors_favorite():
    p_over, p_under = no_vig_probabilities(-130, 110)
    assert p_over + p_under == pytest.approx(1)
    assert p_over > p_under
    assert no_vig_probabilities(-110, -110) == pytest.approx((0.5, 0.5))


# ---------------------------------------------------------------- EV + Kelly
def test_expected_value():
    assert expected_value(0.5, 0.5, 100) == pytest.approx(0)
    assert expected_value(0.55, 0.45, -110) == pytest.approx(0.55 * 100 / 110 - 0.45)
    assert expected_value(110 / 210, 100 / 210, -110) == pytest.approx(0)  # break-even


def test_kelly_textbook_cases():
    # even money, 60% -> 20% of bankroll
    assert kelly_fraction(0.6, 0.4, 100) == pytest.approx(0.2)
    # -110 at 55% -> (0.909*0.55 - 0.45) / 0.909 = 5.5%
    assert kelly_fraction(0.55, 0.45, -110) == pytest.approx(0.055)


def test_kelly_never_negative():
    assert kelly_fraction(0.4, 0.6, -110) == 0.0
    assert kelly_fraction(0.0, 0.0, -110) == 0.0


def test_kelly_with_push_matches_numeric_optimum():
    p, q, price = 0.5, 0.4, -110          # 10% push
    b = payout(price)
    grid = np.linspace(0, 0.99, 99_001)
    growth = p * np.log1p(b * grid) + q * np.log1p(-grid)
    assert kelly_fraction(p, q, price) == pytest.approx(grid[growth.argmax()], abs=1e-4)


def test_stake_fraction_is_fractional_and_capped():
    assert stake_fraction(0.6, 0.4, 100, multiplier=0.25) == pytest.approx(0.05)
    assert stake_fraction(0.9, 0.1, 100, multiplier=1.0, cap=0.05) == 0.05


# ---------------------------------------------------------------- probability
def test_normal_fallback_is_symmetric_and_pushes_on_whole_lines():
    model = ResidualModel(None, sd=80)
    half = model.probabilities(226.5, 226.5)
    assert half.over == pytest.approx(0.5) and half.push == pytest.approx(0)
    whole = model.probabilities(226, 226)
    assert whole.push > 0
    assert whole.over + whole.under + whole.push == pytest.approx(1)


def test_empirical_uses_residual_shape():
    # Model always 20 short -> a line at the prediction is almost always an over
    model = ResidualModel(np.full(500, 20.0))
    p = model.probabilities(200, 210.5)
    assert p.over == 1.0 and p.under == 0.0


def test_too_few_residuals_falls_back_to_normal():
    model = ResidualModel(np.arange(10.0))
    assert model.residuals is None
    assert "normal" in model.method


def test_from_bundle_handles_old_model_files():
    assert "normal" in from_bundle({"models": {}}).method
    bundle = {"residuals": {"xgboost": np.random.default_rng(0).normal(0, 70, 300)}}
    assert "empirical" in from_bundle(bundle).method


# ---------------------------------------------------------------- strategy
def sharp_model():
    return ResidualModel(None, sd=40)


def test_evaluate_side_only_qualifies_in_model_direction():
    cfg = BetConfig(min_ev=0.0)
    over = evaluate_side(260, 230.5, -110, "OVER", sharp_model(), cfg)
    under = evaluate_side(260, 230.5, -110, "UNDER", sharp_model(), cfg)
    assert over["qualifies"] and over["stake"] > 0
    assert not under["qualifies"] and under["stake"] == 0


def test_min_ev_and_min_edge_filters():
    assert not evaluate_side(232, 230.5, -110, "OVER", sharp_model(), BetConfig(min_ev=0.03))["qualifies"]
    assert not evaluate_side(260, 230.5, -110, "OVER", sharp_model(), BetConfig(min_edge_yards=40))["qualifies"]


def test_best_bet_shops_for_the_best_line():
    books = pd.DataFrame({
        "sportsbook": ["A", "B", "C"],
        "line": [230.5, 225.5, 230.5],
        "over_odds": [-110, -110, -105],
        "under_odds": [-110, -110, -115],
    })
    bet = best_bet(books, 260, sharp_model())
    assert bet["side"] == "OVER"
    assert bet["book"] == "B"                      # lowest line to go over
    assert 0 < bet["market_p_win"] < 1


def test_best_bet_follows_the_prediction_even_with_biased_misses():
    # Misses skew high (model usually too low), so an over can look +EV even when
    # the prediction is under the line; the bet must still follow the prediction.
    biased = ResidualModel(np.full(500, 30.0))
    books = pd.DataFrame({"line": [244.5], "over_odds": [-110], "under_odds": [-110]})
    bet = best_bet(books, 242, biased, BetConfig(min_ev=0.0))
    assert bet["side"] == "UNDER" and not bet["qualifies"] and bet["ev"] < 0


def test_best_bet_without_lines_or_prediction():
    empty = pd.DataFrame(columns=["line", "over_odds", "under_odds"])
    assert best_bet(empty, 250, sharp_model()) is None
    books = pd.DataFrame({"line": [230.5], "over_odds": [-110], "under_odds": [-110]})
    assert best_bet(books, float("nan"), sharp_model()) is None


def test_simulate_bankroll_wins_losses_pushes():
    bets = pd.DataFrame({
        "game_date": pd.to_datetime(["2026-10-03", "2026-10-03", "2026-10-10"]),
        "game_id": [1, 2, 3], "player_id": [10, 20, 30],
        "line": [200.5, 200.5, 200.0],
        "over_odds": [100, 100, 100], "under_odds": [-120, -120, -120],
        "pred": [260.0, 260.0, 260.0],
        "passing_yards": [250, 150, 200],          # win, loss, push
    })
    cfg = BetConfig(min_ev=0.0, kelly_multiplier=0.25, max_stake=0.05, max_daily_exposure=1.0)
    log, s = simulate_bankroll(bets, "pred", sharp_model(), cfg, start=100)
    assert list(log["result"]) == ["win", "loss", "push"]
    # Same-day bets are both sized from the morning bankroll, so they net to zero
    assert log["stake"].iloc[0] == pytest.approx(log["stake"].iloc[1])
    assert s["final"] == pytest.approx(100)
    assert s["wins"] == 1 and s["losses"] == 1 and s["pushes"] == 1


def test_daily_exposure_cap():
    n = 20
    bets = pd.DataFrame({
        "game_date": pd.to_datetime(["2026-10-03"] * n), "game_id": range(n), "player_id": range(n),
        "line": [200.5] * n, "over_odds": [100] * n, "under_odds": [-120] * n,
        "pred": [300.0] * n, "passing_yards": [250] * n,
    })
    cfg = BetConfig(min_ev=0.0, max_stake=0.05, max_daily_exposure=0.30)
    log, _ = simulate_bankroll(bets, "pred", sharp_model(), cfg, start=100)
    assert log["stake"].sum() == pytest.approx(30)


def test_simulate_bankroll_with_no_qualifying_bets():
    bets = pd.DataFrame({
        "game_date": pd.to_datetime(["2026-10-03"]), "game_id": [1], "player_id": [1],
        "line": [200.5], "over_odds": [-110], "under_odds": [-110],
        "pred": [201.0], "passing_yards": [250],
    })
    log, s = simulate_bankroll(bets, "pred", sharp_model(), BetConfig(min_ev=0.05))
    assert log.empty and s["bets"] == 0 and s["final"] == s["start"]


def test_cap_exposure():
    np.testing.assert_allclose(cap_exposure([0.05, 0.05], 0.30), [0.05, 0.05])   # under the cap
    capped = cap_exposure([0.05] * 10, 0.30)
    assert capped.sum() == pytest.approx(0.30) and np.allclose(capped, 0.03)
    assert cap_exposure([], 0.30).sum() == 0
