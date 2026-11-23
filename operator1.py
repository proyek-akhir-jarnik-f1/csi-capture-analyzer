import asyncio
import csv
import json
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from shared.hackrf_csi import HackRFRecorder, extract_cs8_to_csv

HOST = "127.0.0.1"
PORT = 8001

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


class Operator1Coordinator:
    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.state_label: str = "IDLE"  # IDLE | SCHEDULED | RUNNING | EXTRACTING
        self.scheduled_start_dt: datetime | None = None
        self.session_id: str = ""
        self.start_mono_ns: int = 0

        # Parameter HackRF (default Channel 6 = 2437 MHz)
        self.freq_hz: int = 2_437_000_000
        self.lna_gain: int = 40
        self.vga_gain: int = 26

        # Durasi siklus napas (detik)
        self.dur_inhale: float = 4.0
        self.dur_hold1: float = 4.0
        self.dur_exhale: float = 4.0
        self.dur_hold2: float = 4.0

        # State siklus aktif
        self.current_phase: str = "IDLE"
        self.cycle_index: int = 0
        self.phase_remaining_sec: float = 0.0
        self.phase_progress: float = 0.0
        self.knock_event_id: int = 0
        self._last_logged_phase_key: str = ""
        self._logged_knock_keys: set[str] = set()
        self._last_prestart_knock_idx: int = -1

        self._cue_file = None
        self._cue_writer = None
        self._cs8_path: Path | None = None
        self._cue_path: Path | None = None
        self._csi_csv_path: Path | None = None

        self.error_message: str | None = None
        self.status_note: str = ""
        self.recorder = HackRFRecorder(on_update=self.broadcast_state)

    def _parse_time(self, time_str: str) -> datetime:
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

    def schedule(
        self,
        start_time_str: str,
        subject_tag: str,
        freq_mhz: float,
        lna_gain: int,
        vga_gain: int,
        breath_cfg: dict[str, float],
    ) -> None:
        if self.state_label in ("RUNNING", "EXTRACTING"):
            raise ValueError("Sesi sedang berjalan. Hentikan sesi terlebih dahulu.")

        start_dt = self._parse_time(start_time_str)
        now = datetime.now()
        if start_dt <= now + timedelta(seconds=1.5):
            raise ValueError(
                f"Jadwal terlalu mepet/sudah lewat ({start_dt.strftime('%H:%M:%S')}). "
                f"Jam sekarang {now.strftime('%H:%M:%S')}."
            )

        clean_tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in subject_tag.strip()) or "subjek"
        self.session_id = f"{clean_tag}_{start_dt.strftime('%Y%m%d_%H%M%S')}"
        self.scheduled_start_dt = start_dt
        self.freq_hz = int(round(freq_mhz * 1_000_000))
        self.lna_gain = int(lna_gain)
        self.vga_gain = int(vga_gain)

        self.dur_inhale = max(1.0, float(breath_cfg.get("inhale", 4.0)))
        self.dur_hold1 = max(0.0, float(breath_cfg.get("hold1", 4.0)))
        self.dur_exhale = max(1.0, float(breath_cfg.get("exhale", 4.0)))
        self.dur_hold2 = max(0.0, float(breath_cfg.get("hold2", 4.0)))

        self.state_label = "SCHEDULED"
        self.error_message = None
        self.status_note = ""
        self._last_prestart_knock_idx = -1

    def _open_cue_log(self) -> None:
        self._cs8_path = LOG_DIR / f"{self.session_id}.cs8"
        self._cue_path = LOG_DIR / f"{self.session_id}_isyarat.csv"
        self._csi_csv_path = LOG_DIR / f"{self.session_id}_csi.csv"

        self._cue_file = open(self._cue_path, "w", newline="", encoding="utf-8")
        self._cue_writer = csv.writer(self._cue_file)
        self._cue_writer.writerow(["timestamp_utc", "t_s", "cycle_index", "phase"])
        self._cue_file.flush()

    def _write_phase_start(self, phase: str, cycle_idx: int, t_s: float) -> None:
        if not self._cue_writer or not self._cue_file:
            return
        utc_iso = datetime.now(timezone.utc).isoformat()
        self._cue_writer.writerow([utc_iso, f"{t_s:.6f}", cycle_idx, phase])
        self._cue_file.flush()

    def _close_cue_log(self) -> None:
        if self._cue_file:
            self._cue_file.close()
            self._cue_file = None
            self._cue_writer = None

    def _begin_running(self) -> None:
        self.start_mono_ns = time.monotonic_ns()
        self._last_logged_phase_key = ""
        self._logged_knock_keys.clear()
        self._open_cue_log()

        assert self._cs8_path is not None
        self.recorder.start(
            cs8_path=self._cs8_path,
            freq_hz=self.freq_hz,
            sample_rate=20_000_000,
            lna_gain=self.lna_gain,
            vga_gain=self.vga_gain,
        )
        self.state_label = "RUNNING"

    async def stop_or_cancel(self) -> None:
        if self.state_label == "SCHEDULED":
            self.state_label = "IDLE"
            self.scheduled_start_dt = None
            await self.broadcast_state()
            return

        if self.state_label == "RUNNING":
            self._close_cue_log()
            self.recorder.stop()

            self.state_label = "EXTRACTING"
            self.status_note = "Mengekstrak .cs8 ke .csv dan menyematkan label napas..."
            await self.broadcast_state()

            if self._cs8_path and self._cue_path and self._csi_csv_path and self._cs8_path.exists():
                try:
                    pkt_count = await asyncio.to_thread(
                        extract_cs8_to_csv,
                        self._cs8_path,
                        self._cue_path,
                        self._csi_csv_path,
                    )
                    self.recorder.stats.csi_csv_path = str(self._csi_csv_path)
                    self.status_note = (
                        f"Selesai! {pkt_count} paket CSI tersimpan di {self._csi_csv_path.name} "
                        f"(backup mentah: {self._cs8_path.name})."
                    )
                except Exception as exc:
                    self.status_note = ""
                    self.error_message = f"Gagal ekstraksi CSV: {exc}"

            self.state_label = "IDLE"
            self.current_phase = "IDLE"
            self.scheduled_start_dt = None
            await self.broadcast_state()

    def _evaluate_breathing_cycle(self, elapsed_s: float) -> None:
        steps: list[tuple[str, float, str]] = [
            ("TARIK_NAFAS", self.dur_inhale, "HEMBUS_OR_START"),
        ]
        if self.dur_hold1 > 0:
            steps.append(("TAHAN_1", self.dur_hold1, "HEMBUS_NAFAS"))
        steps.append(("HEMBUS_NAFAS", self.dur_exhale, "TAHAN_2_OR_TARIK"))
        if self.dur_hold2 > 0:
            steps.append(("TAHAN_2", self.dur_hold2, "TARIK_NAFAS"))

        cycle_total = sum(d for _, d, _ in steps)
        cycle_idx = int(elapsed_s // cycle_total) + 1
        pos_in_cycle = elapsed_s % cycle_total

        acc = 0.0
        for phase_name, dur, _ in steps:
            if pos_in_cycle < acc + dur:
                local_t = pos_in_cycle - acc
                self.cycle_index = cycle_idx
                self.current_phase = phase_name
                self.phase_remaining_sec = max(0.0, dur - local_t)
                self.phase_progress = min(1.0, max(0.0, local_t / dur))

                # Catat log hanya saat isyarat fase napas mulai diberikan
                phase_key = f"{cycle_idx}:{phase_name}"
                if phase_key != self._last_logged_phase_key:
                    self._last_logged_phase_key = phase_key
                    self._write_phase_start(
                        phase=phase_name,
                        cycle_idx=cycle_idx,
                        t_s=elapsed_s,
                    )

                # Tetap picu audio ketukan di UI tanpa dicatat ke CSV
                if phase_name in ("TAHAN_1", "TAHAN_2"):
                    max_knocks = min(3, int(dur // 0.5))
                    for k in range(max_knocks, 0, -1):
                        if local_t >= dur - (k * 0.5):
                            knock_key = f"{cycle_idx}:{phase_name}:K{max_knocks - k + 1}"
                            if knock_key not in self._logged_knock_keys:
                                self._logged_knock_keys.add(knock_key)
                                self.knock_event_id += 1
                break
            acc += dur

    async def scheduler_loop(self) -> None:
        while True:
            now = datetime.now()
            if self.state_label == "SCHEDULED" and self.scheduled_start_dt is not None:
                rem_to_start = (self.scheduled_start_dt - now).total_seconds()
                # Ketukan pra-mulai pada 1.5s, 1.0s, 0.5s sebelum T = 0
                for idx, threshold in enumerate((1.5, 1.0, 0.5)):
                    if rem_to_start <= threshold and self._last_prestart_knock_idx < idx:
                        self._last_prestart_knock_idx = idx
                        self.knock_event_id += 1

                if now >= self.scheduled_start_dt:
                    self._begin_running()

            if self.state_label == "RUNNING":
                elapsed_s = (time.monotonic_ns() - self.start_mono_ns) / 1e9
                self._evaluate_breathing_cycle(elapsed_s)

            await self.broadcast_state()
            await asyncio.sleep(0.05)  # 20 Hz untuk timer & animasi halus

    def build_snapshot(self) -> dict[str, Any]:
        now = datetime.now()
        cd_start: float | None = None
        elapsed_s: float | None = None

        if self.scheduled_start_dt and self.state_label == "SCHEDULED":
            cd_start = max(0.0, (self.scheduled_start_dt - now).total_seconds())
        if self.state_label == "RUNNING":
            elapsed_s = (time.monotonic_ns() - self.start_mono_ns) / 1e9

        return {
            "type": "STATE",
            "now_str": now.strftime("%H:%M:%S.%f")[:-4],
            "state": self.state_label,
            "session_id": self.session_id,
            "scheduled_start_str": self.scheduled_start_dt.strftime("%H:%M:%S") if self.scheduled_start_dt else "-",
            "countdown_start_sec": cd_start,
            "elapsed_sec": elapsed_s,
            "breath": {
                "phase": self.current_phase,
                "cycle": self.cycle_index,
                "remaining_sec": round(self.phase_remaining_sec, 2),
                "progress": round(self.phase_progress, 4),
                "knock_event_id": self.knock_event_id,
                "durations": {
                    "inhale": self.dur_inhale,
                    "hold1": self.dur_hold1,
                    "exhale": self.dur_exhale,
                    "hold2": self.dur_hold2,
                },
            },
            "error_message": self.error_message,
            "status_note": self.status_note,
            "hackrf": asdict(self.recorder.stats),
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


coordinator = Operator1Coordinator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(coordinator.scheduler_loop())
    yield
    task.cancel()
    coordinator.recorder.stop()
    coordinator._close_cue_log()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(BASE_DIR / "templates" / "operator1.html")


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
                        start_time_str=msg["start_time"],
                        subject_tag=msg.get("subject_tag", "subjek"),
                        freq_mhz=float(msg.get("freq_mhz", 2437)),
                        lna_gain=int(msg.get("lna_gain", 40)),
                        vga_gain=int(msg.get("vga_gain", 26)),
                        breath_cfg=msg.get("breath_cfg", {}),
                    )
                except Exception as exc:
                    coordinator.error_message = str(exc)
                await coordinator.broadcast_state()

            elif action == "STOP":
                await coordinator.stop_or_cancel()

    except WebSocketDisconnect:
        coordinator.clients.discard(websocket)


def _open_browser() -> None:
    time.sleep(0.8)
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")