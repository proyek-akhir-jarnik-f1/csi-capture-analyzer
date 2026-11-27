from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ===== ADJUST THIS =====
cs8_path_str = "/run/media/system/CSI/P2_sesi1_20261124_115725.cs8"

output_path_str = "output/1-grafik-gelombang-radio.png"
script_dir = Path(__file__).resolve().parent
output_path = Path(output_path_str)
if not output_path.is_absolute():
    output_path = script_dir / output_path
output_path.parent.mkdir(parents=True, exist_ok=True)

sample_rate = 20_000_000
end_data_sec: float = 0.0035

cs8_path = Path(cs8_path_str)
total_bytes = cs8_path.stat().st_size

expected_samples = int(round(sample_rate * end_data_sec))
if expected_samples <= 0:
    raise ValueError("Durasi end_data_sec harus menghasilkan minimal 1 sampel.")

end_data_bytes = expected_samples * 2
if total_bytes < end_data_bytes:
    raise ValueError(
        f"Ukuran file ({total_bytes} byte) lebih kecil dari kebutuhan ({end_data_bytes} byte)."
    )

raw_data = np.fromfile(cs8_path, dtype=np.int8, count=end_data_bytes)
even_len = (len(raw_data) // 2) * 2
raw_data = raw_data[:even_len]

i = raw_data[0::2].astype(np.float32) / 128.0
q = raw_data[1::2].astype(np.float32) / 128.0
amp = np.hypot(i, q)

t = np.arange(len(amp), dtype=np.float64) / sample_rate

fig, ax = plt.subplots(figsize=(12, 4.5), dpi=150)

ax.plot(t, amp, color="blue", linewidth=0.5)
ax.set_title("Selubung Amplitudo Sinyal")
ax.set_xlabel("Waktu (s)", fontsize=10)
ax.set_ylabel("Amplitudo Selubung", fontsize=10)
ax.set_xlim(0, t[-1])
y_max = float(amp.max())
ax.set_ylim(0, y_max * 1.18 if y_max > 0 else 1.0)

plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.show()