import asyncio
import json
from datetime import datetime, timedelta

import pytest

import operator1
import operator2


def future_time(seconds: int = 30) -> str:
    return (datetime.now() + timedelta(seconds=seconds)).strftime("%H:%M:%S")


def test_operator1_schedule_validates_parameters_and_exposes_sync_fields() -> None:
    coord = operator1.Operator1Coordinator()

    with pytest.raises(ValueError, match="Frekuensi"):
        coord.schedule(future_time(), "S1", 100, 40, 26, {})
    with pytest.raises(ValueError, match="LNA"):
        coord.schedule(future_time(), "S1", 2437, 48, 26, {})

    coord.schedule(
        future_time(),
        "Subjek 1",
        2437,
        40,
        26,
        {"inhale": 4, "hold1": 4, "exhale": 4, "hold2": 0},
    )

    snapshot = coord.build_snapshot()
    assert coord.state_label == "SCHEDULED"
    assert snapshot["scheduled_start_iso"] is not None
    assert snapshot["breath"]["durations"]["hold2"] == 0.0
    assert coord.session_id.lower().startswith("subjek_1_")


def test_operator1_schedule_rejects_too_soon_start() -> None:
    coord = operator1.Operator1Coordinator()

    with pytest.raises(ValueError, match="mepet|lewat"):
        coord.schedule(future_time(0), "S1", 2437, 40, 26, {})


def test_operator2_schedule_validates_ip_and_interval() -> None:
    coord = operator2.Operator2Coordinator()
    start = future_time(40)

    with pytest.raises(ValueError, match="IP tujuan"):
        coord.schedule(start, "bukan-ip", 0.01)
    with pytest.raises(ValueError, match="Interval"):
        coord.schedule(start, "192.168.96.1", 5.0)

    coord.schedule(start, "192.168.96.1", 0.01)
    assert coord.state_label == "SCHEDULED"
    assert coord.ping_start_dt is not None
    assert (coord.agreed_start_dt - coord.ping_start_dt).total_seconds() == 5.0

    coord.state_label = "WARMUP"
    with pytest.raises(ValueError, match="sedang berjalan"):
        coord.schedule(future_time(60), "192.168.96.1", 0.01)


def test_operator2_fetch_op1_schedule(monkeypatch) -> None:
    payload = {
        "ok": True,
        "state": "SCHEDULED",
        "session_id": "S1_20260101_000000",
        "scheduled_start_str": "12:34:56",
        "scheduled_start_iso": "2026-01-01T12:34:56+07:00",
        "countdown_start_sec": 12.5,
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(operator2.urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())

    coord = operator2.Operator2Coordinator()
    result = asyncio.run(coord.fetch_op1_schedule())

    assert result["agreed_time"] == "12:34:56"
    assert result["session_id"] == payload["session_id"]
    assert result["countdown_start_sec"] == 12.5


def test_operator2_fetch_op1_schedule_reports_idle(monkeypatch) -> None:
    payload = {"ok": True, "state": "IDLE", "scheduled_start_str": "-"}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(operator2.urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())

    coord = operator2.Operator2Coordinator()
    with pytest.raises(ValueError, match="belum memiliki jadwal"):
        asyncio.run(coord.fetch_op1_schedule())
