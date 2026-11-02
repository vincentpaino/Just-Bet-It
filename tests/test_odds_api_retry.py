"""Retries in the Odds API client: temporary errors retried, permanent ones not."""
import pytest
import requests

import src.ingestion.odds_api as odds


class FakeResponse:
    def __init__(self, status, text="x"):
        self.status_code, self.text, self.headers = status, text, {}

    def json(self):
        return []


@pytest.fixture(autouse=True)
def no_sleep_and_key(monkeypatch):
    monkeypatch.setattr(odds.time, "sleep", lambda s: None)
    monkeypatch.setenv("ODDS_API_KEY", "test")


def sequence(monkeypatch, *outcomes):
    calls = []
    def fake_get(*a, **k):
        outcome = outcomes[len(calls)]
        calls.append(outcome)
        if isinstance(outcome, Exception):
            raise outcome
        return FakeResponse(outcome)
    monkeypatch.setattr(odds.requests, "get", fake_get)
    return calls


def test_recovers_after_temporary_errors(monkeypatch):
    calls = sequence(monkeypatch, 502, requests.ConnectionError(), 200)
    assert odds._api_get("events").status_code == 200
    assert len(calls) == 3


def test_gives_up_after_all_retries(monkeypatch):
    calls = sequence(monkeypatch, 502, 502, 502, 502)
    with pytest.raises(odds.OddsAPIError, match="502"):
        odds._api_get("events")
    assert len(calls) == 1 + len(odds.RETRY_WAITS)


def test_does_not_retry_permanent_errors(monkeypatch):
    calls = sequence(monkeypatch, 401)
    with pytest.raises(odds.OddsAPIError, match="401"):
        odds._api_get("events")
    assert len(calls) == 1
