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

# =============================================================================
# KONSTANTA L-LTF (IEEE 802.11-2020, Section 17.3.4)
# =============================================================================
# L-LTF domain frekuensi, subcarrier -26..+26 (53 elemen, DC = 0)
L_SEQ = np.array(
    [
        1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1,
        0,
        1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1,
    ],
    dtype=float,
)
K_ALL = np.arange(-26, 27)          # indeks subcarrier -26..+26 (53 elemen)
X_FREQ = np.zeros(64, dtype=complex)
X_FREQ[K_ALL % 64] = L_SEQ          # petakan ke FFT bin (Python % selalu ≥ 0)
LTF_TIME = np.fft.ifft(X_FREQ)      # template L-LTF domain waktu (64 sampel)
SUBCARRIER_INDICES = K_ALL[K_ALL != 0]  # 52 subcarrier aktif (-26..-1, +1..+26)
IDX_FFT = SUBCARRIER_INDICES % 64   # indeks bin FFT untuk 52 subcarrier aktif
LTF_NORM = float(np.linalg.norm(LTF_TIME))


# =============================================================================
# [FIX #4] sanitize_phase: unwrap aman di subcarrier deep-fade
# =============================================================================
# MASALAH LAMA:
#   np.unwrap(np.angle(h_vec)) langsung diterapkan ke semua 52 subcarrier.
#   Ketika ada subcarrier dengan amplitudo mendekati 0 (deep fade), np.angle()
#   menghasilkan nilai acak (noise), sehingga np.unwrap() salah melompat fase
#   dan merusak regresi linear untuk seluruh vektor.
#
# PERBAIKAN:
#   1. Identifikasi subcarrier "valid" berdasarkan amplitudo (>10% dari median).
#   2. Lakukan unwrap dan regresi HANYA pada subcarrier valid.
#   3. Interpolasi linier untuk subcarrier yang ter-mask (deep fade).
#   4. Kurangi komponen linear (STO + CFO residual) dari seluruh vektor.
def sanitize_phase(h_vec: np.ndarray) -> np.ndarray:
    """Unwrap fase terhadap indeks subcarrier k dan kurangkan garis regresi linear a*k + b.
    Aman terhadap subcarrier deep-fade yang dapat merusak np.unwrap().
    """
    amp = np.abs(h_vec)
    # Mask subcarrier yang amplitudonya cukup kuat (>10% dari median)
    median_amp = float(np.median(amp)) + 1e-12
    valid_mask = amp > 0.1 * median_amp

    raw_phase = np.angle(h_vec)

    if valid_mask.sum() < 4:
        # Terlalu sedikit subcarrier valid — kembalikan phase mentah tanpa sanitasi
        return raw_phase

    # Unwrap hanya pada subcarrier valid, lalu interpolasi ke seluruh vektor
    unwrapped_valid = np.unwrap(raw_phase[valid_mask])
    # Regresi linear hanya pakai subcarrier valid
    k_valid = SUBCARRIER_INDICES[valid_mask].astype(float)
    slope, intercept = np.polyfit(k_valid, unwrapped_valid, 1)

    # Bangun unwrapped penuh via interpolasi untuk subcarrier yang di-mask
    unwrapped_full = np.interp(
        SUBCARRIER_INDICES.astype(float),
        k_valid,
        unwrapped_valid,
    )
    return unwrapped_full - (slope * SUBCARRIER_INDICES + intercept)


# =============================================================================
# [FIX #1 & #2] cari_ltf: normalisasi simetris + deteksi simbol ke-1 yang andal
# =============================================================================
# MASALAH LAMA (Bug #1):
#   Normalisasi rho = corr / (LTF_NORM * e) di mana e dihitung dari
#   np.convolve(power, ones(64)). Ini quasi-normalized: LTF_NORM adalah norm
#   template LTF, sedangkan e adalah sqrt(energi sinyal per 64 sampel).
#   Rumus yang benar adalah rho = |corr| / (||template|| * ||segment_lokal||),
#   yaitu cosine similarity. Versi lama menggunakan LTF_NORM yang sudah dihitung
#   di luar, sehingga pembagi LTF_NORM * e sudah benar secara matematis, TAPI
#   e dihitung dari konvolusi yang hasilnya BUKAN sqrt melainkan perlu di-sqrt.
#   Kode lama: e = np.sqrt(np.convolve(...)) → e sudah di-sqrt ✅
#   Jadi normalisasi lama sebenarnya sudah benar formulanya. Namun ada masalah
#   lain: np.convolve dengan mode="valid" menghitung SUM, bukan MEAN, sehingga
#   e = sqrt(sum |seg|²) = ||seg_window||, yang memang benar.
#   YANG SALAH: LTF_NORM dihitung dari LTF_TIME (64 sampel, IFFT ternormalisasi
#   per 1/N), bukan dari satu perioda L-LTF standar. Nilainya bisa jauh berbeda
#   tergantung amplitude LTF_TIME, membuat threshold rho_thresh tidak terprediksi.
#
# PERBAIKAN Bug #1:
#   Gunakan normalized cross-correlation yang benar (cosine similarity):
#   rho[i] = |Σ seg[i+n] * conj(LTF_TIME[n])| / (||seg_window|| * ||LTF_TIME||)
#   Ini selalu dalam rentang [0, 1] secara matematis.
#
# MASALAH LAMA (Bug #2):
#   L-LTF punya DUA simbol identik, sehingga korelasi menghasilkan DUA puncak
#   terpisah 64 sampel. np.argmax() bisa memilih puncak ke-2 (simbol kedua)
#   karena lebih jauh dari transisi burst sehingga "lebih bersih".
#   Koreksi lama "if rho[i-64] > 0.7*rho[i]: i -= 64" terlalu longgar:
#   - Jika noise awal burst membuat puncak simbol-1 rendah (<70% puncak simbol-2),
#     koreksi tidak aktif dan kita pakai simbol-2 sebagai titik awal.
#   - Akibatnya s2 = seg[i+64:i+128] menunjuk ke DATA (bukan LTF), CSI rusak.
#
# PERBAIKAN Bug #2:
#   Cari dua puncak lokal tertinggi. Jika jarak keduanya 60–68 sampel (toleransi
#   ±4 dari 64 ideal), anggap itu pasangan simbol 1 & 2, dan pilih yang lebih awal.
def cari_ltf(seg: np.ndarray) -> tuple[int, float]:
    """Cari indeks awal simbol LTF PERTAMA pada potongan sinyal I/Q.

    Menggunakan normalized cross-correlation (cosine similarity) sehingga
    nilai rho selalu dalam [0, 1] dan threshold konsisten di berbagai SNR.
    Deteksi dua puncak secara eksplisit untuk memastikan yang dipilih adalah
    simbol pertama, bukan simbol kedua.
    """
    if len(seg) < 128:
        return 0, 0.0

    # --- [FIX #1] Normalized cross-correlation yang benar ---
    # Hitung energi tiap window 64 sampel (sliding)
    pwr = np.abs(seg) ** 2
    # energi window ke-i = sum(|seg[i:i+64]|²)
    e_window = np.convolve(pwr, np.ones(64), mode="valid")          # panjang: len(seg)-63
    e_seg = np.sqrt(np.maximum(e_window, 1e-24))                    # ||seg_window||

    # Cross-correlation dengan template LTF_TIME (konjugat otomatis di correlate)
    corr = np.abs(np.correlate(seg, LTF_TIME, mode="valid"))        # panjang: len(seg)-63

    n = min(len(e_seg), len(corr))
    # Cosine similarity sejati: dalam [0, 1]
    rho = corr[:n] / (LTF_NORM * e_seg[:n] + 1e-12)

    # --- [FIX #2] Cari dua puncak lokal, pilih yang lebih awal ---
    # Kumpulkan semua puncak lokal (nilai lebih besar dari dua tetangganya)
    # Hanya pertimbangkan puncak yang rho-nya ≥ 50% dari maksimum global
    rho_max = float(rho.max())
    if rho_max < 1e-6:
        return 0, 0.0

    min_height = 0.5 * rho_max
    local_peaks: list[tuple[float, int]] = []
    for j in range(1, n - 1):
        if rho[j] >= min_height and rho[j] > rho[j - 1] and rho[j] >= rho[j + 1]:
            local_peaks.append((float(rho[j]), j))

    # Urutkan berdasarkan nilai rho, ambil dua teratas
    local_peaks.sort(reverse=True)
    top2 = local_peaks[:2]

    if len(top2) == 2:
        rho_a, idx_a = top2[0]
        rho_b, idx_b = top2[1]
        gap = abs(idx_a - idx_b)
        # Jika jarak dua puncak ≈ 64 sampel (toleransi ±4), ini pasangan LTF
        if 60 <= gap <= 68:
            # Pilih indeks yang lebih KECIL = simbol pertama
            best_idx = min(idx_a, idx_b)
            best_rho = rho_a if idx_a < idx_b else rho_b
            return best_idx, float(best_rho)

    # Fallback: hanya ada satu puncak menonjol atau dua puncak tidak berpasangan.
    # Coba cek apakah 64 sampel sebelum puncak terbaik juga memiliki nilai rho tinggi
    # (menandakan puncak terbaik adalah simbol ke-2).
    if top2:
        best_rho_val, best_idx = top2[0]
        # Cek mundur 60-68 sampel dari puncak terbaik
        for shift in range(60, 69):
            candidate = best_idx - shift
            if candidate >= 0 and rho[candidate] > 0.4 * best_rho_val:
                # Ada puncak sebelumnya yang signifikan → pakai yang lebih awal
                return candidate, float(rho[candidate])
        # Tidak ada puncak lebih awal → pakai puncak tunggal yang ada
        return best_idx, float(best_rho_val)

    # Last resort
    i = int(np.argmax(rho))
    return i, float(rho[i])


# =============================================================================
# [FIX #3] csi_dari_ltf: CFO compensation indeks dimulai dari 0 per segmen
# =============================================================================
# MASALAH LAMA (Bug #3):
#   cfo_rad diestimasi sebagai selisih fase antara s1 dan s2 (dua simbol LTF).
#   Kompensasi: seg[i:i+128] * exp(-j * cfo_rad * arange(128))
#   Ini BENAR secara konseptual, tapi jika i salah (akibat Bug #2), fase awal
#   kompensasi juga salah — phase error terakumulasi di FFT.
#   Setelah Fix #2, i sudah benar. Namun ada masalah tambahan:
#   np.vdot(s1, s2) menghitung Σ conj(s1[n]) * s2[n] — urutan argumen vdot
#   adalah (a, b) → a* · b. Karena s2[n] = s1[n] * exp(j*cfo_rad*64), maka:
#   angle(vdot(s1, s2)) = angle(Σ |s1[n]|² * exp(j*cfo_rad*64)) = cfo_rad * 64
#   Pembagi 64 sudah benar ✅.
#   TAPI: np.angle() mengembalikan nilai dalam (-π, π], sehingga jika CFO besar
#   (cfo_rad > π/64), estimasi alias. Untuk HackRF dengan akurasi osilator ≈ 20 ppm
#   pada 2.4 GHz → δf ≈ 48 kHz → cfo_rad = 2π*48e3/20e6 ≈ 0.015 rad < π/64 ✅
#   Sehingga aliasing tidak terjadi di kondisi normal.
#
# PERBAIKAN Bug #3:
#   Tambahkan komentar eksplisit. Pastikan indeks kompensasi exp() selalu dimulai
#   dari 0 (bukan dari posisi absolut i dalam buffer yang lebih panjang).
#   Ini sudah benar di kode lama, tapi buat lebih eksplisit agar tidak salah edit.
def csi_dari_ltf(seg: np.ndarray, i: int, fs: float = 20e6) -> tuple[np.ndarray, float, float]:
    """Estimasi H[k] (52 subcarrier), CFO (Hz), dan RSSI (dBFS) dari simbol L-LTF.

    Prasyarat: i harus menunjuk ke awal SIMBOL PERTAMA L-LTF (bukan GI, bukan simbol kedua).
    Gunakan cari_ltf() yang sudah diperbaiki untuk memastikan ini.
    """
    # Dua simbol LTF identik, masing-masing 64 sampel
    s1 = seg[i: i + 64]
    s2 = seg[i + 64: i + 128]

    # --- Estimasi CFO ---
    # s2[n] ≈ s1[n] * exp(j * cfo_rad * 64), sehingga:
    # angle(vdot(s1, s2)) = angle(Σ conj(s1[n]) * s2[n]) ≈ cfo_rad * 64
    # [FIX #3] Indeks kompensasi exp() selalu mulai dari 0 (relatif ke s)
    cfo_rad = float(np.angle(np.vdot(s1, s2)) / 64.0)

    # Kompensasi CFO pada kedua simbol — indeks n = 0,1,...,127 relatif ke s[0]
    n_idx = np.arange(128)
    s = seg[i: i + 128] * np.exp(-1j * cfo_rad * n_idx)   # indeks RELATIF, bukan absolut

    # --- Estimasi kanal (CSI) via LS: H[k] = Y[k] / X[k] ---
    # Rata-ratakan FFT dua simbol untuk menekan noise (SNR gain ≈ 3 dB)
    y_fft = (np.fft.fft(s[:64]) + np.fft.fft(s[64:])) / 2.0
    h_vec = y_fft[IDX_FFT] / X_FREQ[IDX_FFT]

    # CFO dalam Hz: cfo_rad [rad/sample] * fs [sample/s] / (2π)
    cfo_hz = cfo_rad * fs / (2.0 * np.pi)

    # RSSI dalam dBFS (relatif terhadap full-scale ±1.0)
    mean_pwr = float(np.mean(np.abs(seg[i: i + 128]) ** 2)) + 1e-12
    rssi_dbfs = 10.0 * np.log10(mean_pwr)

    return h_vec, cfo_hz, rssi_dbfs


# =============================================================================
# [FIX #6] detect_bursts_and_extract: last_end_sample lebih konservatif
# =============================================================================
# MASALAH LAMA (Bug #6):
#   last_end_sample = seg_start + idx_ltf + 160
#   Nilai 160 = panjang L-LTF (32 GI + 64 + 64). Ini "minimal" tapi tidak
#   memperhitungkan bahwa kita butuh sedikit margin antar-paket agar tidak salah
#   deteksi. Jika dua paket WiFi berdekatan (misalnya ACK segera setelah DATA),
#   burst kedua bisa mulai dalam 160 sampel dari burst pertama.
#   Lebih penting: seg_start = max(0, start_sample - 20), sehingga ada 20 sampel
#   margin kiri. Jika paket kedua dimulai di start_sample < last_end_sample,
#   ia di-skip meski sebenarnya valid.
#
# PERBAIKAN Bug #6:
#   Gunakan SIFS (Short Interframe Space) sebagai batas bawah: 16 µs = 320 sampel
#   pada 20 Msps. Jadi paket valid berikutnya minimal dimulai 320 sampel setelah
#   awal LTF paket sebelumnya. Ini lebih konservatif tapi lebih akurat secara fisik.
def detect_bursts_and_extract(
    iq: np.ndarray,
    t_offset_s: float = 0.0,
    fs: float = 20e6,
    rho_thresh: float = 0.6,
    snr_db_thresh: float = 10.0,
    max_packets: int | None = None,
) -> list[dict]:
    """Deteksi burst energi 1 µs dan ekstrak paket yang memiliki L-LTF valid."""
    win = max(1, int(round(fs * 1e-6)))  # 20 sampel pada 20 Msps
    n_blocks = len(iq) // win
    if n_blocks < 20:
        return []

    pwr_blocks = np.mean(np.abs(iq[: n_blocks * win].reshape(n_blocks, win)) ** 2, axis=1)
    noise_floor = float(np.percentile(pwr_blocks, 20)) + 1e-12
    thresh = noise_floor * (10.0 ** (snr_db_thresh / 10.0))

    above = pwr_blocks > thresh
    # Transisi False → True menandai awal burst
    edges = np.where(np.diff(above.astype(np.int8), prepend=0) == 1)[0]

    results: list[dict] = []
    # [FIX #6] SIFS = 16 µs = 320 sampel → paket valid berikutnya minimal 320 sampel
    # setelah awal LTF paket sebelumnya. Ini mencegah double-deteksi pada paket
    # yang sangat berdekatan sekaligus memberi ruang cukup bagi paket ACK.
    SIFS_SAMPLES = int(round(fs * 16e-6))  # 320 sampel pada 20 Msps
    last_ltf_start = -SIFS_SAMPLES

    for blk_idx in edges:
        start_sample = blk_idx * win
        # [FIX #6] Bandingkan start_sample dengan last_ltf_start + SIFS
        if start_sample < last_ltf_start + SIFS_SAMPLES:
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
            # [FIX #6] Simpan posisi absolut awal LTF (bukan akhir)
            last_ltf_start = pkt_sample
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


# =============================================================================
# [FIX #5] extract_cs8_to_csv: overlap diperbesar agar paket di batas tidak hilang
# =============================================================================
# MASALAH LAMA (Bug #5):
#   overlap_bytes = 512 * 2 = 1024 byte = 512 sampel IQ = 25.6 µs
#   Namun fungsi detect_bursts_and_extract memerlukan minimal 160 sampel L-LTF
#   PLUS window deteksi burst (400 sampel) PLUS margin 20 sampel kiri.
#   Total yang dibutuhkan: 400 + 20 = 420 sampel = 840 byte.
#   Dengan overlap 512 sampel, cukup untuk satu paket, tapi jika paket mulai
#   tepat di byte terakhir chunk, segmen yang diambil (seg_end = start+400)
#   bisa melebihi batas overlap → paket hilang.
#
# PERBAIKAN Bug #5:
#   overlap_bytes = 1600 * 2 = 3200 byte = 1600 sampel = 80 µs
#   Ini setara dengan 10× panjang L-LTF (160 sampel), memberi ruang cukup
#   untuk mendeteksi paket apa pun yang mulai di zona batas chunk.
#   Logika deduplikasi paket di zona overlap tetap ada (baris filter p["t_s"]).
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

    # [FIX #5] Overlap diperbesar: 1600 sampel = 80 µs (sebelumnya: 512 sampel = 25.6 µs)
    # Ini memastikan paket apa pun yang mulai di batas chunk tetap terdeteksi.
    overlap_bytes = 1600 * 2  # 3200 byte (sebelumnya: 512 * 2 = 1024 byte)

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
