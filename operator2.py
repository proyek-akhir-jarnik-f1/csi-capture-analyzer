import asyncio
import csv
import json
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from shared.ping_worker import SubprocessPingWorker

HOST = "127.0.0.1"
PORT = 8002
WARMUP_LEAD_SEC = 5.0

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
        self.target_ip = target_ip.strip() or "192.168.96.1"
        self.interval_sec = interval_sec
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
                    await self.worker.start(self.target_ip, self.interval_sec)

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
            "ping": asdict(self.worker.stats),
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