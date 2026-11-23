import asyncio
import csv
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

import numpy as np

# L-LTF domain frekuensi, subcarrier -26..+26 (IEEE 802.11)
L_SEQ = np.array(
    [
        1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1,
        0,
        1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1,
    ],
    dtype=float,
)
K_ALL = np.arange(-26, 27)
X_FREQ = np.zeros(64, dtype=complex)
X_FREQ[K_ALL % 64] = L_SEQ
LTF_TIME = np.fft.ifft(X_FREQ)
SUBCARRIER_INDICES = K_ALL[K_ALL != 0]  # -26..-1, 1..26 (52 subcarrier)
IDX_FFT = SUBCARRIER_INDICES % 64
LTF_NORM = float(np.linalg.norm(LTF_TIME))


def sanitize_phase(h_vec: np.ndarray) -> np.ndarray:
    """Unwrap fase terhadap indeks subcarrier k dan kurangkan garis regresi linear a*k + b."""
    raw_phase = np.unwrap(np.angle(h_vec))
    slope, intercept = np.polyfit(SUBCARRIER_INDICES, raw_phase, 1)
    return raw_phase - (slope * SUBCARRIER_INDICES + intercept)


def cari_ltf(seg: np.ndarray) -> tuple[int, float]:
    """Cari indeks awal simbol LTF pertama pada potongan sinyal I/Q."""
    if len(seg) < 128:
        return 0, 0.0
    e = np.sqrt(np.convolve(np.abs(seg) ** 2, np.ones(64), mode="valid"))
    corr = np.abs(np.correlate(seg, LTF_TIME, mode="valid"))
    n = min(len(e), len(corr))
    rho = corr[:n] / (LTF_NORM * e[:n] + 1e-12)
    i = int(np.argmax(rho))
    if i >= 64 and rho[i - 64] > 0.7 * rho[i]:
        i -= 64
    return i, float(rho[i])


def csi_dari_ltf(seg: np.ndarray, i: int, fs: float = 20e6) -> tuple[np.ndarray, float, float]:
    """Estimasi H[k] (52 subcarrier), CFO (Hz), dan RSSI (dBFS) dari simbol L-LTF."""
    s1, s2 = seg[i : i + 64], seg[i + 64 : i + 128]
    cfo_rad = float(np.angle(np.vdot(s1, s2)) / 64.0)
    s = seg[i : i + 128] * np.exp(-1j * cfo_rad * np.arange(128))
    y_fft = (np.fft.fft(s[:64]) + np.fft.fft(s[64:])) / 2.0
    h_vec = y_fft[IDX_FFT] / X_FREQ[IDX_FFT]
    cfo_hz = cfo_rad * fs / (2.0 * np.pi)
    mean_pwr = float(np.mean(np.abs(seg[i : i + 128]) ** 2)) + 1e-12
    rssi_dbfs = 10.0 * np.log10(mean_pwr)
    return h_vec, cfo_hz, rssi_dbfs


def detect_bursts_and_extract(
    iq: np.ndarray,
    t_offset_s: float = 0.0,
    fs: float = 20e6,
    rho_thresh: float = 0.6,
    snr_db_thresh: float = 10.0,
    max_packets: int | None = None,
) -> list[dict]:
    """Deteksi burst energi 1 us dan ekstrak paket yang memiliki L-LTF valid."""
    win = max(1, int(round(fs * 1e-6)))  # 20 sampel pada 20 Msps
    n_blocks = len(iq) // win
    if n_blocks < 20:
        return []

    pwr_blocks = np.mean(np.abs(iq[: n_blocks * win].reshape(n_blocks, win)) ** 2, axis=1)
    noise_floor = float(np.percentile(pwr_blocks, 20)) + 1e-12
    thresh = noise_floor * (10.0 ** (snr_db_thresh / 10.0))

    above = pwr_blocks > thresh
    # Transisi False -> True menandai awal burst
    edges = np.where(np.diff(above.astype(np.int8), prepend=0) == 1)[0]

    results: list[dict] = []
    last_end_sample = -1

    for blk_idx in edges:
        start_sample = blk_idx * win
        if start_sample < last_end_sample:
            continue
        seg_start = max(0, start_sample - 20)
        seg_end = min(len(iq), start_sample + 400)
        if seg_end - seg_start < 160:
            continue

        seg = iq[seg_start:seg_end]
        idx_ltf, rho = cari_ltf(seg)
        if rho >= rho_thresh and (idx_ltf + 128) <= len(seg):
            h_vec, cfo_hz, rssi_dbfs = csi_dari_ltf(seg, idx_ltf, fs=fs)
            amp = np.abs(h_vec)
            phase_san = sanitize_phase(h_vec)
            phase_raw = np.angle(h_vec)
            pkt_sample = seg_start + idx_ltf
            results.append(
                {
                    "t_s": t_offset_s + (pkt_sample / fs),
                    "rssi_dbfs": rssi_dbfs,
                    "cfo_hz": cfo_hz,
                    "rho": rho,
                    "amp": amp,
                    "phase_san": phase_san,
                    "phase_raw": phase_raw,
                }
            )
            last_end_sample = seg_start + idx_ltf + 160
            if max_packets is not None and len(results) >= max_packets:
                break

    return results


@dataclass
class HackRFLiveStats:
    recording: bool = False
    extracting: bool = False
    cs8_path: str = ""
    csi_csv_path: str = ""
    bytes_written: int = 0
    packets_detected: int = 0
    last_rho: float = 0.0
    last_cfo_hz: float = 0.0
    last_rssi_dbfs: float = -100.0
    clipping_warning: bool = False
    last_error: str | None = None
    subcarriers: list[int] = field(default_factory=lambda: SUBCARRIER_INDICES.tolist())
    latest_amp: list[float] = field(default_factory=list)
    latest_phase: list[float] = field(default_factory=list)


class HackRFRecorder:
    def __init__(self, on_update: Callable[[], Awaitable[None]] | None = None) -> None:
        self.stats = HackRFLiveStats()
        self._proc: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._on_update = on_update
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(
        self,
        cs8_path: Path,
        freq_hz: int = 2_437_000_000,
        sample_rate: int = 20_000_000,
        lna_gain: int = 40,
        vga_gain: int = 26,
    ) -> None:
        self.stop()
        self._loop = asyncio.get_running_loop()
        self._stop_event.clear()

        # Pastikan LNA kelipatan 8 (0..40) dan VGA kelipatan 2 (0..62) sesuai aturan hackrf_transfer
        lna_valid = max(0, min(40, (int(lna_gain) // 8) * 8))
        vga_valid = max(0, min(62, (int(vga_gain) // 2) * 2))

        self.stats = HackRFLiveStats(
            recording=True,
            cs8_path=str(cs8_path),
        )

        cmd = [
            "hackrf_transfer",
            "-r",
            "-",
            "-f",
            str(int(freq_hz)),
            "-s",
            str(int(sample_rate)),
            "-b",
            str(int(sample_rate)),
            "-l",
            str(lna_valid),
            "-g",
            str(vga_valid),
            "-a",
            "0",
        ]
        print("[HackRF CMD]", " ".join(cmd))

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1024 * 1024,
        )
        self._thread = threading.Thread(
            target=self._io_worker,
            args=(cs8_path, float(sample_rate)),
            daemon=True,
        )
        self._thread.start()

    def _notify_async(self) -> None:
        if self._on_update and self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(self._on_update(), self._loop)

    def _io_worker(self, cs8_path: Path, fs: float) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None

        chunk_size = 262_144  # 256 KB per baca
        # Cuplik 2 ms (40.000 sampel = 80.000 byte) setiap ~150 ms untuk plot live
        preview_bytes_needed = int(fs * 0.002) * 2
        last_preview_time = 0.0
        preview_buf = bytearray()

        try:
            with open(cs8_path, "wb") as f_out:
                while not self._stop_event.is_set():
                    data = self._proc.stdout.read(chunk_size)
                    if not data:
                        break
                    f_out.write(data)
                    self.stats.bytes_written += len(data)

                    now = time.monotonic()
                    if (now - last_preview_time) >= 0.15:
                        preview_buf.extend(data)
                        if len(preview_buf) >= preview_bytes_needed:
                            last_preview_time = now
                            raw = np.frombuffer(
                                bytes(preview_buf[:preview_bytes_needed]), dtype=np.int8
                            )
                            preview_buf.clear()

                            # Cek clipping ADC 8-bit (+/-127)
                            clip_ratio = float(np.mean(np.abs(raw) >= 126))
                            self.stats.clipping_warning = clip_ratio > 0.01

                            iq = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64) / 128.0
                            pkts = detect_bursts_and_extract(iq, fs=fs, max_packets=1)
                            if pkts:
                                latest = pkts[-1]
                                self.stats.packets_detected += 1
                                self.stats.last_rho = round(latest["rho"], 3)
                                self.stats.last_cfo_hz = round(latest["cfo_hz"], 1)
                                self.stats.last_rssi_dbfs = round(latest["rssi_dbfs"], 2)
                                self.stats.latest_amp = np.round(latest["amp"], 4).tolist()
                                self.stats.latest_phase = np.round(latest["phase_san"], 4).tolist()
                            self._notify_async()
        except Exception as exc:
            self.stats.last_error = f"Error I/O HackRF: {exc}"
        finally:
            if self._proc:
                self._proc.wait(timeout=1.0)
                err_txt = ""
                if self._proc.stderr:
                    err_txt = self._proc.stderr.read().decode("utf-8", errors="replace").strip()
                if err_txt and not self.stats.last_error:
                    self.stats.last_error = " | ".join(err_txt.splitlines())
            self.stats.recording = False
            self._notify_async()

    def stop(self) -> None:
        self._stop_event.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.stats.recording = False


def extract_cs8_to_csv(
    cs8_path: Path,
    cue_log_path: Path,
    output_csv_path: Path,
    fs: float = 20e6,
    chunk_sec: float = 1.0,
) -> int:
    """
    Ekstraksi seluruh file .cs8 menjadi .csv sesuai format kolom Bagian 3.4 PDF,
    lalu menyematkan label fase napas dari file log isyarat berdasarkan waktu t_s.
    """
    # Muat timeline fase dari log isyarat
    phase_timeline: list[tuple[float, str]] = []
    if cue_log_path.exists():
        with open(cue_log_path, "r", encoding="utf-8") as f_cue:
            reader = csv.DictReader(f_cue)
            for row in reader:
                if "t_s" in row and "phase" in row:
                    phase_timeline.append((float(row["t_s"]), row["phase"]))
    phase_timeline.sort(key=lambda x: x[0])

    def lookup_label(t_sec: float) -> str:
        active = "IDLE"
        for t_start, phase_name in phase_timeline:
            if t_sec >= t_start:
                active = phase_name
            else:
                break
        return active

    header = ["t_s", "rssi_dbfs", "cfo_hz", "rho"]
    header.extend([f"amp_k{k}" for k in SUBCARRIER_INDICES])
    header.extend([f"fasa_k{k}" for k in SUBCARRIER_INDICES])
    header.append("label")

    total_bytes = cs8_path.stat().st_size
    bytes_per_sec = int(fs) * 2
    chunk_bytes = int(chunk_sec * bytes_per_sec)
    overlap_bytes = 512 * 2  # Overlap kecil agar paket di batas blok tidak terpotong

    packet_count = 0
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(header)

        offset = 0
        while offset < total_bytes:
            read_len = min(chunk_bytes + overlap_bytes, total_bytes - offset)
            raw = np.fromfile(cs8_path, dtype=np.int8, count=read_len, offset=offset)
            if len(raw) < 400:
                break
            iq = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64) / 128.0
            t_offset = offset / bytes_per_sec

            pkts = detect_bursts_and_extract(iq, t_offset_s=t_offset, fs=fs)
            for p in pkts:
                # Hindari duplikasi paket di zona overlap
                if offset + chunk_bytes < total_bytes and p["t_s"] >= (offset + chunk_bytes) / bytes_per_sec:
                    continue
                row = [
                    f"{p['t_s']:.6f}",
                    f"{p['rssi_dbfs']:.3f}",
                    f"{p['cfo_hz']:.2f}",
                    f"{p['rho']:.4f}",
                ]
                row.extend([f"{v:.5f}" for v in p["amp"]])
                row.extend([f"{v:.5f}" for v in p["phase_raw"]])
                row.append(lookup_label(p["t_s"]))
                writer.writerow(row)
                packet_count += 1

            offset += chunk_bytes

    return packet_count


if __name__ == "__main__":
    # Bisa dijalankan ulang secara manual jika parameter ekstraksi diubah:
    # python -m shared.hackrf_csi logs/sesi.cs8 logs/sesi_isyarat.csv logs/sesi_csi.csv
    if len(sys.argv) < 4:
        print("Pemakaian: python -m shared.hackrf_csi <file.cs8> <file_isyarat.csv> <output_csi.csv>")
        sys.exit(1)
    count = extract_cs8_to_csv(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    print(f"Ekstraksi selesai: {count} paket L-LTF tersimpan ke {sys.argv[3]}")