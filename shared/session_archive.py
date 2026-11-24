from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CSI_SUFFIX = "_csi.csv"
CUE_SUFFIX = "_isyarat.csv"
ALLOWED_SUFFIXES = (".csv", ".cs8")


@dataclass
class SessionSummary:
    session_id: str
    packets: int = 0
    first_t_s: float | None = None
    last_t_s: float | None = None
    mean_rssi_dbfs: float | None = None
    mean_rho: float | None = None
    mean_cfo_hz: float | None = None
    phase_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("first_t_s", "last_t_s", "mean_rssi_dbfs", "mean_rho", "mean_cfo_hz"):
            if data[key] is not None:
                data[key] = round(float(data[key]), 4)
        if self.first_t_s is not None and self.last_t_s is not None:
            data["duration_sec"] = round(self.last_t_s - self.first_t_s, 4)
        else:
            data["duration_sec"] = None
        return data


def summarize_csi_csv(path: Path) -> SessionSummary | None:
    """Ringkas file CSI menjadi statistik ringan tanpa memuat seluruh baris ke memori."""
    summary = SessionSummary(session_id=path.name.removesuffix(CSI_SUFFIX))
    count = 0
    rssi_sum = rho_sum = cfo_sum = 0.0

    try:
        with open(path, "r", newline="", encoding="utf-8-sig") as f_csv:
            reader = csv.DictReader(f_csv)
            required = {"t_s", "rssi_dbfs", "cfo_hz", "rho", "label"}
            if not required.issubset(set(reader.fieldnames or [])):
                return None

            for row in reader:
                try:
                    t_s = float(row["t_s"])
                    rssi = float(row["rssi_dbfs"])
                    rho = float(row["rho"])
                    cfo = float(row["cfo_hz"])
                except (TypeError, ValueError):
                    continue

                count += 1
                rssi_sum += rssi
                rho_sum += rho
                cfo_sum += cfo
                summary.first_t_s = t_s if summary.first_t_s is None else min(summary.first_t_s, t_s)
                summary.last_t_s = t_s if summary.last_t_s is None else max(summary.last_t_s, t_s)
                label = (row.get("label") or "IDLE").strip() or "IDLE"
                summary.phase_counts[label] = summary.phase_counts.get(label, 0) + 1
    except OSError:
        return None

    summary.packets = count
    if count:
        summary.mean_rssi_dbfs = rssi_sum / count
        summary.mean_rho = rho_sum / count
        summary.mean_cfo_hz = cfo_sum / count
    return summary


def resolve_log_file(log_dir: Path, name: str) -> Path:
    """Validasi nama file agar tidak bisa keluar dari direktori logs."""
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Nama file tidak valid")
    base = log_dir.resolve()
    path = (base / name).resolve()
    if path.parent != base:
        raise ValueError("Nama file tidak valid")
    if not path.is_file():
        raise FileNotFoundError(f"File tidak ditemukan: {name}")
    return path


def classify_log_file(name: str) -> str:
    if name.endswith(CSI_SUFFIX):
        return "csi"
    if name.endswith(CUE_SUFFIX):
        return "cue"
    if name.startswith("op2_ping_") and name.endswith(".csv"):
        return "audit"
    if name.endswith(".cs8"):
        return "raw"
    return "csv"


def list_session_files(log_dir: Path) -> list[dict[str, Any]]:
    """Daftar file hasil sesi, terbaru lebih dulu, lengkap dengan ringkasan CSI."""
    if not log_dir.is_dir():
        return []

    files: list[dict[str, Any]] = []
    for path in log_dir.iterdir():
        if not path.is_file() or path.suffix not in ALLOWED_SUFFIXES:
            continue

        stat = path.stat()
        kind = classify_log_file(path.name)
        session_id = path.name
        for suffix in (CSI_SUFFIX, CUE_SUFFIX, ".cs8"):
            if session_id.endswith(suffix):
                session_id = session_id[: -len(suffix)]
                break
        if kind == "audit":
            session_id = path.name.removesuffix(".csv")

        entry: dict[str, Any] = {
            "name": path.name,
            "kind": kind,
            "session_id": session_id,
            "size_bytes": stat.st_size,
            "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            "summary": None,
        }
        if kind == "csi":
            summary = summarize_csi_csv(path)
            entry["summary"] = summary.as_dict() if summary else None
        files.append(entry)

    files.sort(key=lambda item: item["modified_utc"], reverse=True)
    return files
