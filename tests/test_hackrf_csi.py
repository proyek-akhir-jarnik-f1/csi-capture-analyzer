import csv

import numpy as np
import pytest

from shared.hackrf_csi import (
    IDX_FFT,
    LTF_NORM,
    LTF_TIME,
    SUBCARRIER_INDICES,
    X_FREQ,
    cari_ltf,
    csi_dari_ltf,
    detect_bursts_and_extract,
    extract_cs8_to_csv,
    sanitize_phase,
)

FS = 20e6


def make_ltf_signal(
    offset: int,
    length: int = 2000,
    amp: float = 0.5,
    cfo_rad: float = 0.0,
    noise: float = 0.001,
    seed: int = 7,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sig = noise * (rng.standard_normal(length) + 1j * rng.standard_normal(length))
    n = np.arange(128)
    sig[offset : offset + 128] += amp * np.tile(LTF_TIME, 2) * np.exp(1j * cfo_rad * n)
    return sig.astype(np.complex64)


def test_ltf_template_is_unit_energy_reference() -> None:
    assert LTF_TIME.shape == (64,)
    assert LTF_NORM > 0
    assert len(SUBCARRIER_INDICES) == 52
    assert X_FREQ[0] == 0
    assert np.allclose(np.abs(X_FREQ[IDX_FFT]), 1.0)


def test_cari_ltf_detects_first_of_two_symbols() -> None:
    idx, rho = cari_ltf(make_ltf_signal(200))

    assert rho > 0.6
    assert abs(idx - 200) <= 2
    assert idx < 264


def test_cari_ltf_rejects_pure_noise() -> None:
    rng = np.random.default_rng(3)
    noise = (rng.standard_normal(2000) + 1j * rng.standard_normal(2000)).astype(np.complex64)
    idx, rho = cari_ltf(noise)

    assert 0 <= rho < 0.6


def test_sanitize_phase_removes_linear_trend() -> None:
    k = SUBCARRIER_INDICES.astype(float)
    h = (1.0 + 0.1 * np.cos(k / 10.0)) * np.exp(1j * (0.02 * k + 0.3))
    h[12] = 1e-6 * np.exp(1j * 2.5)

    cleaned = sanitize_phase(h)
    valid = np.abs(h) > 0.1 * np.median(np.abs(h))

    assert np.max(np.abs(cleaned[valid])) < 0.2


def test_csi_dari_ltf_recovers_channel_and_cfo() -> None:
    cfo_rad = 0.01
    sig = make_ltf_signal(150, length=900, amp=0.4, cfo_rad=cfo_rad, noise=0.0)
    idx, rho = cari_ltf(sig)
    h_vec, cfo_hz, rssi = csi_dari_ltf(sig, idx, fs=FS)

    assert rho > 0.6
    assert np.allclose(np.abs(h_vec), 0.4, atol=0.02)
    assert abs(cfo_hz - cfo_rad * FS / (2 * np.pi)) < 200
    assert rssi < 0


def test_detect_bursts_extracts_only_valid_packets() -> None:
    sig = make_ltf_signal(300, length=3000)
    packets = detect_bursts_and_extract(sig, fs=FS)

    assert len(packets) == 1
    assert abs(packets[0]["t_s"] - 300 / FS) < 1e-5
    assert len(packets[0]["amp"]) == 52
    assert len(packets[0]["phase_san"]) == 52


def test_extract_cs8_to_csv_writes_labeled_csi_rows(tmp_path) -> None:
    sig = make_ltf_signal(1000, length=40000)
    iq = np.stack((sig.real, sig.imag), axis=1)
    raw = np.clip(np.round(iq * 127), -127, 127).astype(np.int8).reshape(-1)
    cs8_path = tmp_path / "subjek.cs8"
    raw.tofile(cs8_path)

    cue_path = tmp_path / "subjek_isyarat.csv"
    with open(cue_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_utc", "t_s", "cycle_index", "phase"])
        writer.writerow(["2026-01-01T00:00:00+00:00", "0.000000", 1, "TARIK_NAFAS"])
        writer.writerow(["2026-01-01T00:00:01+00:00", "0.001000", 1, "TAHAN_1"])

    out_path = tmp_path / "subjek_csi.csv"
    count = extract_cs8_to_csv(cs8_path, cue_path, out_path, fs=FS)

    assert count >= 1
    with open(out_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == count
    assert rows[0]["label"] == "TARIK_NAFAS"
    assert len(rows[0]) == 109


def test_extract_cs8_to_csv_handles_missing_cue_log(tmp_path) -> None:
    sig = make_ltf_signal(500, length=20000)
    raw = np.clip(np.round(np.stack((sig.real, sig.imag), axis=1) * 127), -127, 127)
    cs8_path = tmp_path / "tanpa_cue.cs8"
    raw.astype(np.int8).reshape(-1).tofile(cs8_path)

    out_path = tmp_path / "tanpa_cue_csi.csv"
    count = extract_cs8_to_csv(cs8_path, tmp_path / "tidak_ada.csv", out_path, fs=FS)

    assert count >= 1
    with open(out_path, newline="", encoding="utf-8") as f:
        first = next(csv.DictReader(f))
    assert first["label"] == "IDLE"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
