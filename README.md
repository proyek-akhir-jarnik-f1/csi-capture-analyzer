# CSI Capture Analyzer

Aplikasi lokal dua operator untuk menangkap CSI Wi-Fi memakai HackRF sambil memberi isyarat napas,
dengan generator traffic ping sebagai sumber paket. Fungsi utama tidak berubah: Operator 1 menjadwalkan
rekaman + pemandu napas, Operator 2 mengirim ping otomatis mulai 5 detik sebelum waktu yang disepakati.

## Instalasi

```bash
pip install -r requirements.txt
```

Butuh `hackrf_transfer` di `PATH` untuk Operator 1, dan perintah `ping` untuk Operator 2.
Izin administratif/root mungkin diperlukan jika interval ping < 0.2 detik ditolak sistem.

## Pemakaian

Jalankan dua terminal terpisah:

```bash
python operator1.py   # http://127.0.0.1:8001
python operator2.py   # http://127.0.0.1:8002
```

Alur:

1. Di Operator 1, isi kode subjek, waktu mulai, frekuensi, gain, dan setting napas lalu klik **Jadwalkan Start**.
2. Di Operator 2, klik **Ambil Jadwal O1** agar waktu yang disepakati terisi otomatis, lalu **Jadwalkan Ping**.
   Tombol ini opsional; waktu juga bisa diisi manual.
3. Ping mulai otomatis pada T-5 detik, perekaman + isyarat napas mulai pada T=0. Klik **Stop Sesi** /
   **Stop Ping** untuk menghentikan lebih awal.
4. Setelah berhenti, Operator 1 mengekstrak `.cs8` menjadi `_csi.csv` berisi label fase napas.

Jika Operator 2 berjalan di komputer lain, arahkan sinkron ke alamat Operator 1:

```bash
set CSI_OP1_URL=http://192.168.1.10:8001   # Windows
# atau: export CSI_OP1_URL=http://192.168.1.10:8001 (Linux/macOS)
```

## Arsip & analisis

Semua hasil ada di folder `logs/`:

| File | Isi |
| --- | --- |
| `<sesi>.cs8` | Backup sampel I/Q mentah dari HackRF |
| `<sesi>_isyarat.csv` | Waktu mulai setiap fase napas (t_s relatif terhadap awal rekaman) |
| `<sesi>_csi.csv` | 52 amplitudo, 52 fase, RSSI, CFO, rho, dan `label` per paket |
| `op2_ping_<waktu>.csv` | Audit jadwal dan statistik ping Operator 2 |

Kedua UI punya kartu **Arsip** yang menampilkan ringkasan (jumlah paket, durasi, RSSI rata-rata,
fase dominan) dan menyediakan tombol unduh. Ringkasan dihitung langsung dari `_csi.csv`.
Koneksi WebSocket otomatis mencoba reconnect dengan backoff saat server restart.

## API

### Operator 1 (port 8001)

| Endpoint | Kegunaan |
| --- | --- |
| `GET /api/session` | Status sesi & waktu mulai untuk sinkronisasi Operator 2 |
| `GET /api/files` | Daftar file di `logs/` beserta ringkasan CSI |
| `GET /api/summary?name=<sesi>_csi.csv` | Ringkasan statistik satu file CSI |
| `GET /api/files/<nama>` | Unduh file dari `logs/` (path divalidasi) |

### Operator 2 (port 8002)

| Endpoint | Kegunaan |
| --- | --- |
| `GET /api/op1-schedule` | Ambil jadwal aktif dari Operator 1 |
| `GET /api/files` | Daftar log audit ping |
| `GET /api/files/<nama>` | Unduh log audit |

Ekstraksi juga bisa dijalankan ulang manual:

```bash
python -m shared.hackrf_csi logs/sesi.cs8 logs/sesi_isyarat.csv logs/sesi_csi.csv
```

## Test

```bash
python -m pytest -q
```

Test mencakup deteksi L-LTF, estimasi CSI/CFO, sanitasi fase, ekstraksi `.cs8` berlabel,
validasi penjadwalan kedua operator, sinkronisasi jadwal, dan keamanan path arsip.
