import asyncio
import re
import signal
from dataclasses import dataclass, field
from typing import Callable, Awaitable


@dataclass
class PingStats:
    running: bool = False
    target: str = "192.168.96.1"
    interval_sec: float = 0.01
    transmitted: int = 0
    received: int = 0
    last_rtt_ms: float | None = None
    last_error: str | None = None
    recent_lines: list[str] = field(default_factory=list)


class SubprocessPingWorker:
    def __init__(self, on_update: Callable[[], Awaitable[None]] | None = None) -> None:
        self.stats = PingStats()
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._on_update = on_update

    async def start(self, target: str, interval_sec: float = 0.01) -> None:
        await self.stop()

        self.stats = PingStats(
            running=True,
            target=target,
            interval_sec=interval_sec,
        )

        # Menjalankan perintah persis: ping -i 0.01 <ip_hotspot>
        self._process = await asyncio.create_subprocess_exec(
            "ping",
            "-i",
            f"{interval_sec:g}",
            target,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self._reader_task = asyncio.create_task(self._read_output())

    async def _read_output(self) -> None:
        assert self._process is not None
        assert self._process.stdout is not None

        seq_pattern = re.compile(r"icmp_seq=(\d+).*time=([\d.]+)\s*ms")
        last_notify = 0.0
        loop = asyncio.get_running_loop()

        try:
            while True:
                line_bytes = await self._process.stdout.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                match = seq_pattern.search(line)
                if match:
                    seq = int(match.group(1))
                    rtt = float(match.group(2))
                    self.stats.received += 1
                    self.stats.transmitted = max(self.stats.transmitted, seq)
                    self.stats.last_rtt_ms = rtt
                elif "Operation not permitted" in line or "cannot flood" in line:
                    self.stats.last_error = line

                self.stats.recent_lines.append(line)
                if len(self.stats.recent_lines) > 20:
                    self.stats.recent_lines.pop(0)

                # Throttle update UI maksimal 10x/detik agar WebSocket tetap ringan
                now = loop.time()
                if self._on_update and (now - last_notify) >= 0.1:
                    last_notify = now
                    await self._on_update()
        finally:
            rc = await self._process.wait()
            self.stats.running = False
            if rc not in (0, -signal.SIGINT, -signal.SIGTERM) and not self.stats.last_error:
                self.stats.last_error = f"Proses ping berhenti dengan exit code {rc}"
            if self._on_update:
                await self._on_update()

    async def stop(self) -> None:
        if self._process and self._process.returncode is None:
            # Kirim SIGINT agar ping mencetak ringkasan statistik akhir sebelum keluar
            self._process.send_signal(signal.SIGINT)
            try:
                await asyncio.wait_for(self._process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()

        if self._reader_task and not self._reader_task.done():
            await self._reader_task

        self.stats.running = False