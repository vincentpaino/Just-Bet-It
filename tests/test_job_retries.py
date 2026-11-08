import subprocess
from pathlib import Path

from sqlalchemy.exc import OperationalError

import src.ingestion.run_ingestion as ri


class FakeSession:
    def __init__(self):
        self.rollbacks, self.logged = 0, []
    def rollback(self): self.rollbacks += 1
    def add(self, row): self.logged.append(row)
    def commit(self): pass


def test_ingestion_step_retries_once_after_a_dropped_connection(monkeypatch):
    monkeypatch.setattr(ri.time, "sleep", lambda s: None)
    calls = []
    def step(session, year):
        calls.append(year)
        if len(calls) == 1:
            raise OperationalError("SELECT 1", {}, Exception("server closed the connection unexpectedly"))
        return 7
    session = FakeSession()
    ri._run_step(session, "test", "test", step, 2026)
    assert calls == [2026, 2026] and session.rollbacks == 1
    assert session.logged[-1].success and session.logged[-1].record_count == 7


def test_ingestion_step_gives_up_after_the_retry(monkeypatch):
    monkeypatch.setattr(ri.time, "sleep", lambda s: None)
    def step(session, year):
        raise OperationalError("SELECT 1", {}, Exception("timed out"))
    session = FakeSession()
    ri._run_step(session, "test", "test", step, 2026)   # logged as a failure, doesn't raise
    assert not session.logged[-1].success


def test_job_retry_reruns_a_failing_step_and_never_stops_the_job(tmp_path):
    script = Path("scripts/snapshot_odds.sh").read_text()
    retry_fn = script[script.index("retry() {"):script.index("\n}\n", script.index("retry() {")) + 3]
    counter = tmp_path / "n"
    harness = f"""
set -euo pipefail
sleep() {{ :; }}
wait_for_network() {{ :; }}
{retry_fn}
flaky() {{ n=$(cat {counter} 2>/dev/null || echo 0); echo $((n+1)) > {counter}; [ $((n+1)) -ge 2 ]; }}
retry flaky flaky
retry broken false
echo "job reached the end"
"""
    out = subprocess.run(["bash", "-c", harness], capture_output=True, text=True)
    assert out.returncode == 0
    assert counter.read_text().strip() == "2"                 # failed once, succeeded on retry
    assert "[broken] FAILED after 3 attempts" in out.stdout
    assert "job reached the end" in out.stdout
