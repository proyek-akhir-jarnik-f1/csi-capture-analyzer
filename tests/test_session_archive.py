import csv

import pytest

from shared.session_archive import (
    classify_log_file,
    list_session_files,
    resolve_log_file,
    summarize_csi_csv,
)


def write_csi_csv(path, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "rssi_dbfs", "cfo_hz", "rho", "label"])
        writer.writerows(rows)


def test_summarize_csi_csv_returns_phase_and_signal_stats(tmp_path) -> None:
    path = tmp_path / "subjek_csi.csv"
    write_csi_csv(
        path,
        [
            [0.0, -60.0, 120.0, 0.8, "TARIK_NAFAS"],
            [1.0, -58.0, 130.0, 0.9, "TARIK_NAFAS"],
            [2.0, -62.0, 110.0, 0.7, "TAHAN_1"],
            ["rusak", "-", "-", "-", "IDLE"],
        ],
    )

    summary = summarize_csi_csv(path)

    assert summary is not None
    assert summary.packets == 3
    assert summary.phase_counts == {"TARIK_NAFAS": 2, "TAHAN_1": 1}
    assert summary.mean_rssi_dbfs == pytest.approx(-60.0)
    assert summary.mean_rho == pytest.approx(0.8)
    assert summary.mean_cfo_hz == pytest.approx(120.0)
    assert summary.as_dict()["duration_sec"] == 2.0


def test_summarize_csi_csv_rejects_unknown_format(tmp_path) -> None:
    path = tmp_path / "acak.csv"
    path.write_text("foo,bar\n1,2\n", encoding="utf-8")

    assert summarize_csi_csv(path) is None


def test_list_session_files_classifies_and_sorts(tmp_path) -> None:
    write_csi_csv(tmp_path / "s1_csi.csv", [[0.0, -60, 0, 0.8, "IDLE"]])
    (tmp_path / "s1_isyarat.csv").write_text("phase\nIDLE\n", encoding="utf-8")
    (tmp_path / "s1.cs8").write_bytes(b"1234")
    (tmp_path / "op2_ping_20260101.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "abaikan.txt").write_text("x", encoding="utf-8")

    files = list_session_files(tmp_path)
    kinds = {item["name"]: item["kind"] for item in files}

    assert kinds == {
        "s1_csi.csv": "csi",
        "s1_isyarat.csv": "cue",
        "s1.cs8": "raw",
        "op2_ping_20260101.csv": "audit",
    }
    csi_entry = next(item for item in files if item["kind"] == "csi")
    assert csi_entry["session_id"] == "s1"
    assert csi_entry["summary"]["packets"] == 1
    assert classify_log_file("x_csi.csv") == "csi"


def test_resolve_log_file_blocks_path_traversal(tmp_path) -> None:
    outside = tmp_path.parent / "rahasia.csv"
    outside.write_text("rahasia", encoding="utf-8")

    with pytest.raises(ValueError):
        resolve_log_file(tmp_path, "../rahasia.csv")
    with pytest.raises(ValueError):
        resolve_log_file(tmp_path, "sub/fail.csv")

    aman = tmp_path / "aman.csv"
    aman.write_text("ok", encoding="utf-8")
    assert resolve_log_file(tmp_path, "aman.csv") == aman.resolve()

    with pytest.raises(FileNotFoundError):
        resolve_log_file(tmp_path, "hilang.csv")
