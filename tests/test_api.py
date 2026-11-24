import csv
import json

from fastapi.testclient import TestClient

import operator1
import operator2


def write_csi(path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "rssi_dbfs", "cfo_hz", "rho", "amp_k-1", "label"])
        writer.writerow([0.0, -60.0, 10.0, 0.85, 0.5, "TARIK_NAFAS"])


def test_operator1_api_serves_session_archive_and_summary(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(operator1, "LOG_DIR", tmp_path)
    write_csi(tmp_path / "S1_csi.csv")
    (tmp_path / "S1_isyarat.csv").write_text("t_s,phase\n0,TARIK_NAFAS\n", encoding="utf-8")

    with TestClient(operator1.app) as client:
        assert client.get("/").status_code == 200

        session = client.get("/api/session").json()
        assert session["ok"] is True
        assert session["state"] == "IDLE"

        files = client.get("/api/files").json()["files"]
        assert {item["kind"] for item in files} == {"csi", "cue"}

        summary = client.get("/api/summary", params={"name": "S1_csi.csv"}).json()
        assert summary["packets"] == 1
        assert summary["phase_counts"] == {"TARIK_NAFAS": 1}

        download = client.get("/api/files/S1_csi.csv")
        assert download.status_code == 200
        assert "amp_k-1" in download.text

        missing = client.get("/api/files/tidak_ada.csv")
        assert missing.status_code == 404
        invalid = client.get("/api/summary", params={"name": "S1_isyarat.csv"})
        assert invalid.status_code == 400


def test_operator2_api_lists_audit_files_and_syncs(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(operator2, "LOG_DIR", tmp_path)
    (tmp_path / "op2_ping_20260101_000000.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    payload = {
        "ok": True,
        "state": "SCHEDULED",
        "session_id": "S1",
        "scheduled_start_str": "12:00:00",
        "countdown_start_sec": 5.0,
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(operator2.urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())

    with TestClient(operator2.app) as client:
        assert client.get("/").status_code == 200
        files = client.get("/api/files").json()["files"]
        assert len(files) == 1
        assert files[0]["kind"] == "audit"
        assert client.get("/api/files/op2_ping_20260101_000000.csv").status_code == 200

        synced = client.get("/api/op1-schedule")
        assert synced.status_code == 200
        assert synced.json()["agreed_time"] == "12:00:00"
