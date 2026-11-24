import asyncio
import csv
import ipaddress
import json
import os
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from shared.ping_worker import SubprocessPingWorker
from shared.session_archive import list_session_files, resolve_log_file

HOST = "127.0.0.1"
PORT = 8002
WARMUP_LEAD_SEC = 5.0
OP1_BASE_URL = os.getenv("CSI_OP1_URL", "http://127.0.0.1:8001").rstrip("/")

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


class Operator2Coordinator:
    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.state_label: str = "IDLE"  # IDLE | SCHEDULED | WARMUP | ACTIVE
        self.agreed_start_dt: datetime | None = None
        self.ping_start_dt: datetime | None = None
        self.target_ip: str = "192.168.96.1"
        self.interval_sec: float = 0.01
        self.error_message: str | None = None
        self._ping_started_for_schedule: bool = False
        self.worker = SubprocessPingWorker(on_update=self.broadcast_state)

    def _parse_agreed_time(self, time_str: str) -> datetime:
        now = datetime.now()
        parts = [int(p) for p in time_str.strip().split(":")]
        if len(parts) == 2:
            hour, minute, second = parts[0], parts[1], 0
        elif len(parts) == 3:
            hour, minute, second = parts[0], parts[1], parts[2]
        else:
            raise ValueError("Format waktu harus HH:MM atau HH:MM:SS")

        candidate = now.replace(hour=hour, minute=minute, second=second, microsecond=0)
        if (now - candidate).total_seconds() > 43200:
            candidate += timedelta(days=1)
        return candidate

    def schedule(self, agreed_time_str: str, target_ip: str, interval_sec: float) -> None:
        if self.state_label in ("WARMUP", "ACTIVE"):
            raise ValueError("Ping sedang berjalan. Hentikan terlebih dahulu sebelum menjadwalkan ulang.")

        clean_ip = target_ip.strip() or "192.168.96.1"
        try:
            ipaddress.ip_address(clean_ip)
        except ValueError as exc:
            raise ValueError(f"IP tujuan tidak valid: {target_ip}") from exc

        interval_value = float(interval_sec)
        if not 0.001 <= interval_value <= 1.0:
            raise ValueError("Interval harus antara 0.001 dan 1.0 detik.")

        agreed_dt = self._parse_agreed_time(agreed_time_str)
        ping_start = agreed_dt - timedelta(seconds=WARMUP_LEAD_SEC)
        now = datetime.now()

        if ping_start <= now + timedelta(seconds=1):
            raise ValueError(
                f"Jadwal terlalu dekat/sudah lewat. Waktu mulai ping (T-5s) jatuh pada "
                f"{ping_start.strftime('%H:%M:%S')}, sedangkan jam sekarang {now.strftime('%H:%M:%S')}."
            )

        self.agreed_start_dt = agreed_dt
        self.ping_start_dt = ping_start
        self.target_ip = clean_ip
        self.interval_sec = interval_value
        self.state_label = "SCHEDULED"
        self.error_message = None
        self._ping_started_for_schedule = False

    async def cancel_or_stop(self) -> None:
        was_running = self.worker.stats.running
        await self.worker.stop()
        if was_running:
            self._write_audit_log()
        self.state_label = "IDLE"
        self.agreed_start_dt = None
        self.ping_start_dt = None
        self._ping_started_for_schedule = False
        await self.broadcast_state()

    def _write_audit_log(self) -> None:
        now = datetime.now()
        log_file = LOG_DIR / f"op2_ping_{now.strftime('%Y%m%d_%H%M%S')}.csv"
        with open(log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "agreed_op1_start",
                "ping_start_target",
                "stopped_at",
                "target_ip",
                "interval_sec",
                "transmitted",
                "received",
            ])
            writer.writerow([
                self.agreed_start_dt.isoformat() if self.agreed_start_dt else "-",
                self.ping_start_dt.isoformat() if self.ping_start_dt else "-",
                now.isoformat(),
                self.target_ip,
                self.interval_sec,
                self.worker.stats.transmitted,
                self.worker.stats.received,
            ])

    async def scheduler_loop(self) -> None:
        while True:
            now = datetime.now()
            if (
                self.state_label in ("SCHEDULED", "WARMUP", "ACTIVE")
                and self.agreed_start_dt is not None
                and self.ping_start_dt is not None
            ):
                if now >= self.ping_start_dt and not self._ping_started_for_schedule:
                    self._ping_started_for_schedule = True
                    self.state_label = "WARMUP"
                    try:
                        await self.worker.start(self.target_ip, self.interval_sec)
                    except Exception as exc:
                        self.state_label = "IDLE"
                        self.agreed_start_dt = None
                        self.ping_start_dt = None
                        self._ping_started_for_schedule = False
                        self.error_message = f"Gagal menjalankan ping: {exc}"

                if now >= self.agreed_start_dt and self.state_label == "WARMUP":
                    self.state_label = "ACTIVE"

            await self.broadcast_state()
            await asyncio.sleep(0.1)

    def build_snapshot(self) -> dict[str, Any]:
        now = datetime.now()
        countdown_ping_sec: float | None = None
        countdown_op1_sec: float | None = None
        elapsed_active_sec: float | None = None

        if self.ping_start_dt:
            countdown_ping_sec = max(0.0, (self.ping_start_dt - now).total_seconds())
        if self.agreed_start_dt:
            countdown_op1_sec = max(0.0, (self.agreed_start_dt - now).total_seconds())
            if now >= self.agreed_start_dt and self.state_label == "ACTIVE":
                elapsed_active_sec = (now - self.agreed_start_dt).total_seconds()

        return {
            "type": "STATE",
            "now_str": now.strftime("%H:%M:%S.%f")[:-4],
            "state": self.state_label,
            "agreed_start_str": self.agreed_start_dt.strftime("%H:%M:%S") if self.agreed_start_dt else "-",
            "ping_start_str": self.ping_start_dt.strftime("%H:%M:%S") if self.ping_start_dt else "-",
            "countdown_ping_sec": countdown_ping_sec,
            "countdown_op1_sec": countdown_op1_sec,
            "elapsed_active_sec": elapsed_active_sec,
            "error_message": self.error_message,
            "op1_base_url": OP1_BASE_URL,
            "ping": asdict(self.worker.stats),
        }

    async def fetch_op1_schedule(self) -> dict[str, Any]:
        """Ambil waktu mulai yang sudah dijadwalkan Operator 1 (opsional)."""

        def _request() -> dict[str, Any]:
            req = urllib.request.Request(
                f"{OP1_BASE_URL}/api/session", headers={"Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=3) as response:
                return json.loads(response.read().decode("utf-8"))

        try:
            data = await asyncio.to_thread(_request)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise ValueError(f"Gagal menghubungi Operator 1 di {OP1_BASE_URL}: {exc}") from exc

        if data.get("state") not in ("SCHEDULED", "RUNNING"):
            raise ValueError("Operator 1 belum memiliki jadwal aktif.")
        agreed_time = data.get("scheduled_start_str") or ""
        if not agreed_time or agreed_time == "-":
            raise ValueError("Waktu mulai Operator 1 tidak tersedia.")
        return {
            "ok": True,
            "agreed_time": agreed_time,
            "session_id": data.get("session_id", ""),
            "state": data["state"],
            "countdown_start_sec": data.get("countdown_start_sec"),
        }

    async def broadcast_state(self) -> None:
        if not self.clients:
            return
        payload = json.dumps(self.build_snapshot())
        dead: list[WebSocket] = []
        for ws in self.clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


coordinator = Operator2Coordinator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(coordinator.scheduler_loop())
    yield
    task.cancel()
    await coordinator.worker.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(BASE_DIR / "templates" / "operator2.html")


@app.get("/api/op1-schedule")
async def api_op1_schedule() -> dict[str, Any]:
    try:
        return await coordinator.fetch_op1_schedule()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/api/files")
async def api_files() -> dict[str, Any]:
    files = await asyncio.to_thread(list_session_files, LOG_DIR)
    return {"files": [item for item in files if item["kind"] == "audit"]}


@app.get("/api/files/{name}")
async def api_download(name: str) -> FileResponse:
    try:
        path = resolve_log_file(LOG_DIR, name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return FileResponse(path, filename=name, media_type="text/csv")


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    coordinator.clients.add(websocket)
    await websocket.send_text(json.dumps(coordinator.build_snapshot()))

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            action = msg.get("action")

            if action == "SCHEDULE":
                try:
                    coordinator.schedule(
                        agreed_time_str=msg["agreed_time"],
                        target_ip=msg["target_ip"],
                        interval_sec=float(msg.get("interval_sec", 0.01)),
                    )
                except Exception as exc:
                    coordinator.error_message = str(exc)
                await coordinator.broadcast_state()

            elif action == "STOP":
                await coordinator.cancel_or_stop()

    except WebSocketDisconnect:
        coordinator.clients.discard(websocket)


def _open_browser() -> None:
    time.sleep(0.8)
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")