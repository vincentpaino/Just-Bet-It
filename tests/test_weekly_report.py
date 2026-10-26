"""Tests for the pre-registered confident-picks rule in scripts/weekly_report.py."""
import numpy as np
import pandas as pd
import pytest

from scripts.weekly_report import CONFIDENT_RULE, confident_section, is_confident

AFTER = "2026-10-31"     # a game played after the rule was registered
BEFORE = CONFIDENT_RULE["registered"]


def preds(lr, rf, xgb, nn):
    return {"linear_regression": lr, "random_forest": rf, "xgboost": xgb, "neural_network": nn}


@pytest.mark.parametrize("p, line, expected", [
    (preds(230, 225, 220, 240), 200.5, True),    # 19.5 yds over, all over
    (preds(170, 175, 180, 160), 200.5, True),    # 20.5 yds under, all under
    (preds(230, 225, 215.5, 240), 200.5, True),  # exactly 15 yds: counts
    (preds(230, 225, 215, 240), 200.5, False),   # 14.5 yds: not enough edge
    (preds(230, 195, 220, 240), 200.5, False),   # one model on the other side
    (preds(230, 200.5, 220, 240), 200.5, False), # one model exactly on the line
])
def test_rule(p, line, expected):
    assert is_confident(p, line, AFTER) is expected


def test_games_on_or_before_registration_are_not_eligible():
    assert is_confident(preds(300, 300, 300, 300), 200.5, BEFORE) is None
    assert is_confident(preds(300, 300, 300, 300), 200.5, "2026-10-24") is None


def picks_frame(rows):
    return pd.DataFrame(rows, columns=["season", "week", "confident_pick", "xgboost_result", "xgboost_units"])


def test_section_before_any_eligible_games():
    old = picks_frame([(2026, 8, np.nan, "win", 0.9), (2026, 8, np.nan, "loss", -1.0)])
    text = "\n".join(confident_section(old))
    assert "No eligible games graded yet" in text


def test_section_counts_only_confident_eligible_games():
    rows = picks_frame([
        (2026, 8, np.nan, "win", 0.9),     # week 8: before registration, ignored
        (2026, 9, True, "win", 0.9),
        (2026, 9, True, "loss", -1.0),
        (2026, 9, False, "win", 0.9),      # eligible but not confident: not bet
        (2026, 10, True, "win", 0.87),
    ])
    lines = confident_section(rows)
    week9 = next(l for l in lines if l.startswith("| 2026 | 9 |"))
    assert "| 3 | 2 | 1-1 | 50.0% | -0.10 |" in week9
    total = next(l for l in lines if "2026 total" in l)
    assert "| 4 | 3 | 2-1 | 66.7% | +0.77 |" in total
    assert not any(l.startswith("| 2026 | 8 |") for l in lines)
