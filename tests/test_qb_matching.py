"""Tests for matching sportsbook player names to CFBD QB IDs."""
from src.ingestion.run_ingestion import _QBMatcher

ROWS = [
    ("Thomas Castellanos", 1),
    ("Duce Robinson Jr.", 2),
    ("Jake Smith", 3),
    ("Tyler Smith", 4),
    ("Avery Johnson", 5),
]


def test_exact_name_and_case_punctuation():
    m = _QBMatcher(rows=ROWS)
    assert m("Avery Johnson") == 5
    assert m("avery johnson") == 5
    assert m("Duce Robinson") == 2          # "Jr." ignored


def test_nickname_falls_back_to_unique_last_name():
    m = _QBMatcher(rows=ROWS)
    assert m("Tommy Castellanos") == 1


def test_shared_last_name_is_not_guessed():
    m = _QBMatcher(rows=ROWS)
    assert m("J. Smith") is None             # two Smiths: ambiguous
    assert m("Bobby Smith") is None


def test_unknown_player():
    assert _QBMatcher(rows=ROWS)("Someone Else") is None
