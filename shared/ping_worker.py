import asyncio
import ctypes
import re
import signal
import socket
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Awaitable, Callable

_WIN_ICMP_PAYLOAD = b"csi-ping-payload-0123456789abcde"
_WIN_INVALID_HANDLE = ctypes.c_void_p(-1).value


if sys.platform == "win32":

    class _IpOptionInformation(ctypes.Structure):
        _fields_ = [
            ("Ttl", ctypes.c_ubyte),
            ("Tos", ctypes.c_ubyte),
            ("Flags", ctypes.c_ubyte),
            ("OptionsSize", ctypes.c_ubyte),
            ("OptionsData", ctypes.c_void_p),
        ]

    class _IcmpEchoReply(ctypes.Structure):
        _fields_ = [
            ("Address", ctypes.c_uint32),
            ("Status", ctypes.c_uint32),
            ("RoundTripTime", ctypes.c_uint32),
            ("DataSize", ctypes.c_uint16),
            ("Reserved", ctypes.c_uint16),
            ("Data", ctypes.c_void_p),
            ("Options", _IpOptionInformation),
        ]

    _IPHLPAPI = ctypes.WinDLL("iphlpapi", use_last_error=True)
    _IPHLPAPI.IcmpCreateFile.argtypes = []
    _IPHLPAPI.IcmpCreateFile.restype = ctypes.c_void_p
    _IPHLPAPI.IcmpCloseHandle.argtypes = [ctypes.c_void_p]
    _IPHLPAPI.IcmpCloseHandle.restype = ctypes.c_int
    _IPHLPAPI.IcmpSendEcho.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint16,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    _IPHLPAPI.IcmpSendEcho.restype = ctypes.c_uint32

    _WINMM = ctypes.WinDLL("winmm")
    _WINMM.timeBeginPeriod.argtypes = [ctypes.c_uint]
    _WINMM.timeBeginPeriod.restype = ctypes.c_uint
    _WINMM.timeEndPeriod.argtypes = [ctypes.c_uint]
    _WINMM.timeEndPeriod.restype = ctypes.c_uint


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

        # State backend ICMP Win32 (digunakan saat sys.platform == "win32")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._win_stop_event: threading.Event | None = None
        self._win_executor: ThreadPoolExecutor | None = None
        self._win_pacer_thread: threading.Thread | None = None
        self._win_handles: list[int] = []
        self._win_lock = threading.Lock()
        self._win_in_flight: int = 0
        self._win_tls = threading.local()
        self._last_notify: float = 0.0

    async def start(self, target: str, interval_sec: float = 0.01) -> None:
        await self.stop()

        clean_target = target.strip()
        self.stats = PingStats(
            running=True,
            target=clean_target,
            interval_sec=interval_sec,
        )

        if sys.platform == "win32":
            await self._start_windows(clean_target, interval_sec)
            return

        # Menjalankan perintah persis: ping -i 0.01 <ip_hotspot>
        self._process = await asyncio.create_subprocess_exec(
            "ping",
            "-i",
            f"{interval_sec:g}",
            clean_target,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self._reader_task = asyncio.create_task(self._read_output())

    async def _start_windows(self, target: str, interval_sec: float) -> None:
        try:
            target_ip = await asyncio.to_thread(socket.gethostbyname, target)
            dst_addr = struct.unpack("<I", socket.inet_aton(target_ip))[0]
        except Exception as exc:
            err_msg = f"Target tidak valid / gagal resolve IP: {exc}"
            self.stats.last_error = err_msg
            self.stats.recent_lines.append(err_msg)
            self.stats.running = False
            if self._on_update:
                await self._on_update()
            return

        self._loop = asyncio.get_running_loop()
        self._last_notify = 0.0
        self._win_stop_event = threading.Event()
        self._win_tls = threading.local()
        with self._win_lock:
            self._win_in_flight = 0
            self._win_handles.clear()
        self._win_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="win_icmp")
        self._win_pacer_thread = threading.Thread(
            target=self._win_pacer_loop,
            args=(dst_addr, interval_sec),
            daemon=True,
        )
        self._win_pacer_thread.start()

    def _win_pacer_loop(self, dst_addr: int, interval_sec: float) -> None:
        assert self._win_stop_event is not None
        stop_event = self._win_stop_event
        executor = self._win_executor
        interval = max(0.001, float(interval_sec))

        _WINMM.timeBeginPeriod(1)
        try:
            next_tick = time.perf_counter()
            seq = 0
            while not stop_event.is_set():
                if executor is None:
                    break

                should_submit = False
                with self._win_lock:
                    if self._win_in_flight < 32:
                        seq += 1
                        self._win_in_flight += 1
                        cur_seq = seq
                        should_submit = True

                if should_submit:
                    try:
                        executor.submit(self._win_send_one, dst_addr, cur_seq)
                    except RuntimeError:
                        with self._win_lock:
                            self._win_in_flight = max(0, self._win_in_flight - 1)
                        break

                next_tick += interval
                now = time.perf_counter()
                if now - next_tick > interval * 5:
                    next_tick = now
                rem = next_tick - now
                if rem > 0:
                    stop_event.wait(rem)
        finally:
            _WINMM.timeEndPeriod(1)

    def _win_send_one(self, dst_addr: int, seq: int) -> None:
        stop_event = self._win_stop_event
        try:
            if stop_event is not None and stop_event.is_set():
                return

            handle = getattr(self._win_tls, "handle", None)
            if not handle:
                handle = _IPHLPAPI.IcmpCreateFile()
                if not handle or handle == _WIN_INVALID_HANDLE:
                    err_code = ctypes.get_last_error()
                    line = f"IcmpCreateFile gagal untuk icmp_seq={seq} (error={err_code})"
                    self._schedule_win_record(seq, False, 0.0, line)
                    return
                self._win_tls.handle = handle
                with self._win_lock:
                    self._win_handles.append(handle)

            reply_buf = ctypes.create_string_buffer(1024)
            t0 = time.perf_counter()
            replies = _IPHLPAPI.IcmpSendEcho(
                handle,
                dst_addr,
                _WIN_ICMP_PAYLOAD,
                len(_WIN_ICMP_PAYLOAD),
                None,
                reply_buf,
                len(reply_buf),
                500,
            )
            rtt_ms = (time.perf_counter() - t0) * 1000.0
            reply = ctypes.cast(reply_buf, ctypes.POINTER(_IcmpEchoReply)).contents

            if replies > 0 and reply.Status == 0:
                reply_ip = socket.inet_ntoa(struct.pack("<I", reply.Address))
                line = (
                    f"{reply.DataSize} bytes from {reply_ip}: "
                    f"icmp_seq={seq} ttl={reply.Options.Ttl} time={rtt_ms:.2f} ms"
                )
                ok = True
            else:
                code = reply.Status if replies > 0 else ctypes.get_last_error()
                line = f"Request timeout/error for icmp_seq={seq} (status={code})"
                ok = False

            self._schedule_win_record(seq, ok, rtt_ms, line)
        finally:
            with self._win_lock:
                self._win_in_flight = max(0, self._win_in_flight - 1)

    def _schedule_win_record(self, seq: int, ok: bool, rtt_ms: float, line: str) -> None:
        loop = self._loop
        stop_event = self._win_stop_event
        if not loop or loop.is_closed() or (stop_event is not None and stop_event.is_set()):
            return
        try:
            loop.call_soon_threadsafe(self._record_win_result, seq, ok, rtt_ms, line)
        except RuntimeError:
            pass

    def _record_win_result(self, seq: int, ok: bool, rtt_ms: float, line: str) -> None:
        if not self.stats.running:
            return

        self.stats.transmitted = max(self.stats.transmitted, seq)
        if ok:
            self.stats.received += 1
            self.stats.last_rtt_ms = round(rtt_ms, 2)

        self.stats.recent_lines.append(line)
        if len(self.stats.recent_lines) > 20:
            self.stats.recent_lines.pop(0)

        now = time.monotonic()
        if self._on_update and (now - self._last_notify) >= 0.1:
            self._last_notify = now
            asyncio.create_task(self._on_update())

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
        if sys.platform == "win32":
            await self._stop_windows()
            return

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

    async def _stop_windows(self) -> None:
        if self._win_stop_event is not None:
            self._win_stop_event.set()

        pacer = self._win_pacer_thread
        executor = self._win_executor
        self._win_pacer_thread = None
        self._win_executor = None

        def _cleanup_threads() -> None:
            if pacer and pacer.is_alive():
                pacer.join(timeout=2.0)
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            with self._win_lock:
                handles = list(self._win_handles)
                self._win_handles.clear()
            for handle in handles:
                try:
                    _IPHLPAPI.IcmpCloseHandle(handle)
                except Exception:
                    pass

        if pacer is not None or executor is not None or self._win_handles:
            await asyncio.to_thread(_cleanup_threads)

        was_running = self.stats.running
        self.stats.running = False
        if was_running and self._on_update:
            await self._on_update()
