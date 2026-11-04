import pandas as pd
import pytest

from scripts.record_picks import picks_from
from scripts.snapshot_closing import window_end
from scripts.weekly_report import live_clv, live_clv_section


def ct(s):
    return pd.Timestamp(s, tz="America/Chicago")


# 2026-11-06 is a Friday, 2026-11-07 a Saturday
@pytest.mark.parametrize("now, end", [
    ("2026-11-06 16:00", "2026-11-07 10:30"),   # Friday afternoon run -> until Saturday's first run
    ("2026-11-07 10:00", "2026-11-07 13:30"),
    ("2026-11-07 13:02", "2026-11-07 16:30"),   # a slightly late run still points at the next one
    ("2026-11-07 16:00", "2026-11-07 19:30"),
    ("2026-11-07 19:00", "2026-11-08 03:00"),   # last run of the week -> next 8 hours
    ("2026-11-06 09:00", "2026-11-06 16:30"),   # a run on wake before Friday 4pm
])
def test_window_ends_at_next_run(now, end):
    assert window_end(ct(now).tz_convert("UTC")) == ct(end).tz_convert("UTC")


def test_confident_needs_big_edge_and_all_four_models_agreeing():
    qbs = pd.DataFrame({
        "game_id": [1, 2, 3, 4], "player_id": [10, 20, 30, 40],
        "pred_xgboost": [270, 270, 225, 250],
        "pred_linear_regression": [260, 240, 230, 250],
        "pred_random_forest": [265, 260, 228, 250],
        "pred_neural_network": [262, 255, 220, 250],
    })
    lines = pd.DataFrame({
        "game_id": [1, 1, 1, 2, 3, 4], "player_id": [10, 10, 10, 20, 30, 40],
        "sportsbook": ["a", "b", "c", "a", "a", "a"],
        "line": [249.5, 250.5, 252.5, 250.5, 250.5, 250.0],
    })
    p = picks_from(qbs, lines).set_index("game_id")
    assert p.loc[1, "line"] == 250.5 and p.loc[1, "books"] == 3       # median across books
    assert p.loc[1, "side"] == "OVER" and p.loc[1, "confident"]       # +19.5, all over
    assert not p.loc[2, "confident"]                                  # linear regression says under
    assert p.loc[3, "side"] == "UNDER" and p.loc[3, "confident"]      # -25.5, all under
    assert 4 not in p.index                                           # no edge, no pick


def test_clv_uses_each_books_last_snapshot_and_the_pick_side():
    picks = pd.DataFrame({"game_id": [1, 2, 3], "player_id": [10, 20, 30], "line": [250.5, 220.5, 200.5],
                          "side": ["OVER", "UNDER", "OVER"], "season": 2026, "week": 11})
    t0, t1 = pd.Timestamp("2026-11-07 15:00"), pd.Timestamp("2026-11-07 17:00")
    closing = pd.DataFrame({
        "game_id": [1, 1, 1, 2],
        "player_id": [10, 10, 10, 20],
        "sportsbook": ["a", "a", "b", "a"],
        "line": [240.5, 255.5, 253.5, 225.5],
        "fetched_at": [t0, t1, t1, t1],
    })
    c = live_clv(picks, closing).set_index("game_id")
    assert c.loc[1, "close_line"] == 254.5 and c.loc[1, "clv"] == 4.0   # over: line rose 4 -> +4
    assert c.loc[2, "clv"] == -5.0                                      # under: line rose 5 -> -5
    assert 3 not in c.index                                             # no snapshot, not counted

    section = "\n".join(live_clv_section({"picks": picks, "clv": c.reset_index()}))
    assert "| 2026 | 11 | 3 | 2 | -0.5 | 50% | 50% |" in section


def test_clv_section_before_any_picks():
    section = live_clv_section({"picks": pd.DataFrame(), "clv": pd.DataFrame()})
    assert "No recorded confident picks" in section[-1]


def test_graded_games_use_recorded_picks_and_skip_unrecorded_ones_before_the_cutoff():
    from datetime import date
    from scripts.weekly_report import ALL_MODELS, with_predictions

    class Constant:
        def __init__(self, v): self.v = v
        def predict(self, X): return [self.v] * len(X)

    bundle = {"models": {m: Constant(200.0) for m in ALL_MODELS if m != "ensemble"},
              "features": ["f"], "train_end": pd.Timestamp("2026-11-01")}
    games = pd.DataFrame({"game_id": [1, 2, 3], "player_id": [10, 20, 30], "f": [0, 0, 0],
                          "game_date": pd.to_datetime(["2026-10-31", "2026-10-31", "2026-11-07"])})
    recorded = pd.DataFrame({"game_id": [1], "player_id": [10], "model_trained_through": [date(2026, 10, 30)],
                             **{m: [250.0] for m in ALL_MODELS}})
    out = with_predictions(games, bundle, recorded).set_index("game_id")
    assert list(out.index) == [1, 3]                    # game 2: not recorded and before the cutoff
    assert out.loc[1, "p_xgboost"] == 250.0             # the pick actually made on Friday
    assert out.loc[1, "p_model_trained_through"] == date(2026, 10, 30)
    assert out.loc[3, "p_xgboost"] == 200.0             # current models, after their cutoff


@pytest.mark.parametrize("today, season", [("2026-11-04", 2026), ("2027-01-09", 2026), ("2027-08-29", 2027)])
def test_current_season_keeps_january_bowls_in_last_years_season(today, season):
    from datetime import date
    from src.ingestion.dates import current_season
    assert current_season(date.fromisoformat(today)) == season
