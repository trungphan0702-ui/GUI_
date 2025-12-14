# -*- coding: utf-8 -*- Chương trình này đo THD và compression threshold và ratio có save file.csv, setting sound card kiểu 1
"""
ĐO THD VÀ COMPRESSION (Threshold, Ratio) QUA UMC22 LOOPBACK
Tác giả: ChatGPT + Luật Trần
"""

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import csv, os, platform, time

# ==================== CẤU HÌNH ====================
fs = 48000
freq = 1000.0
duration = 2.0
amp = 0.7
filename_csv = "ket_qua_do.csv"

# ==================== HÀM PHỤ ====================
def safe_rms(x):
    return np.sqrt(np.mean(np.square(x))) if len(x) > 0 else 0.0

def thd_analysis(signal, fs, freq):
    N = len(signal)
    fft = np.fft.rfft(signal * np.hanning(N))
    mag = np.abs(fft)
    freqs = np.fft.rfftfreq(N, 1/fs)
    fund_idx = np.argmin(np.abs(freqs - freq))
    fund_mag = mag[fund_idx]
    harmonics = {}
    for n in range(2, 6):
        idx = np.argmin(np.abs(freqs - n*freq))
        harmonics[n] = 20*np.log10(mag[idx]/fund_mag + 1e-12)
    thd_ratio = np.sqrt(np.sum([10**(v/10) for v in harmonics.values()]))
    thd_db = 20*np.log10(thd_ratio+1e-12)
    return harmonics, thd_ratio*100, thd_db

# ==================== CHỌN THIẾT BỊ ====================
input_index = output_index = None
for i, dev in enumerate(sd.query_devices()):
    if "USB Audio CODEC" in dev['name']:
        if dev['max_input_channels'] > 0:
            input_index = i
        if dev['max_output_channels'] > 0:
            output_index = i
print(f"✅ Input index: {input_index}, Output index: {output_index}")
sd.default.device = (input_index, output_index)
sd.default.samplerate = fs

# ==================== CHỌN CHẾ ĐỘ ====================
print("\n1️⃣ - THD / Harmonic Distortion")
print("2️⃣ - Compression (Threshold & Ratio)")
mode = input("Nhập lựa chọn (1 hoặc 2): ").strip()

# ========================================================
# 1️⃣ THD MODE
# ========================================================
if mode == "1":
    print(f"\n🎙️ Phát & ghi {freq:.0f} Hz trong {duration:.1f} s ...")
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    sine = amp * np.sin(2*np.pi*freq*t)
    sine_stereo = np.column_stack((sine, np.zeros_like(sine)))
    rec = sd.playrec(sine_stereo, samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    sig = np.squeeze(rec)
    print("✅ Ghi xong.")
    print(f"Min: {np.min(sig):.6f} Max: {np.max(sig):.6f}")

    # Bỏ 50 ms đầu để tránh phần im lặng
    sig = sig[int(0.05*fs):]
    sig /= np.max(np.abs(sig))

    # === VẼ ===
    plt.figure(figsize=(12,6))
    # --- 10 chu kỳ ở giữa tín hiệu ---
    cycles = int(fs/freq*10)
    mid = len(sig)//2
    plt.subplot(2,1,1)
    plt.plot(np.arange(cycles)/fs*1000, sig[mid:mid+cycles], 'b')
    plt.title(f"Recorded Waveform (10 cycles @ {freq:.0f} Hz)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.magnitude_spectrum(sig, Fs=fs, scale='dB', color='r')
    plt.title(f"FFT Spectrum ({freq:.0f} Hz test tone)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dBFS)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === PHÂN TÍCH THD ===
    harmonics, thd_percent, thd_db = thd_analysis(sig, fs, freq)
    print("\n🎵 THD / Harmonic Analysis:")
    for n, v in harmonics.items():
        print(f"H{n}: {n*freq:.1f} Hz, {v:.2f} dBc")
    print(f"🔸 THD = {thd_percent:.4f}% ({thd_db:.2f} dB)")

    # === Ghi CSV ===
    with open(filename_csv, "a", newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), "THD",
                    f"{thd_percent:.4f}%", f"{thd_db:.2f} dB"])
    print(f"💾 Đã lưu kết quả vào '{filename_csv}'.")

# ========================================================
# 2️⃣ COMPRESSION MODE
# ========================================================
elif mode == "2":
    print(f"\n🎛️ Đo compressor @ {freq:.0f} Hz ...")
    seg_dur, gap_dur = 0.25, 0.05
    amps = np.linspace(0.05, 1.36, 36)
    protect = 1.36
    t_seg = np.linspace(0, seg_dur, int(fs*seg_dur), endpoint=False)
    gap = np.zeros(int(fs*gap_dur))
    tx = np.concatenate([np.concatenate((min(A,protect)*np.sin(2*np.pi*freq*t_seg), gap)) for A in amps])
    print(f"🔊 Tổng thời gian test ~{len(tx)/fs:.1f}s ({len(amps)} mức)")
    rx = sd.playrec(tx, samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    rx = np.squeeze(rx)

    segN, gapN = int(seg_dur*fs), int(gap_dur*fs)
    trim_lead, trim_tail = int(0.03*fs), int(0.01*fs)
    rms_in_db, rms_out_db = [], []

    for A, i in zip(amps, range(len(amps))):
        s0, s1 = i*(segN+gapN), i*(segN+gapN)+segN
        seg = rx[s0:s1]
        seg = seg[trim_lead:max(trim_lead, len(seg)-trim_tail)]
        rin = max(A/np.sqrt(2),1e-12)
        rout = max(safe_rms(seg),1e-12)
        rms_in_db.append(20*np.log10(rin))
        rms_out_db.append(20*np.log10(rout))
    rms_in_db, rms_out_db = np.array(rms_in_db), np.array(rms_out_db)
    diff = rms_out_db - rms_in_db

    # ---- Kiểm tra "Không nén" ----
    a_all, b_all = np.polyfit(rms_in_db, rms_out_db, 1)
    gain_offset_db = np.mean(diff)
    slope_tol, spread_tol = 0.05, 1.0
    no_compression = (abs(a_all - 1.0) < slope_tol) and ((diff.max()-diff.min()) < spread_tol)

    if no_compression:
        print("\n📊 KẾT QUẢ: Không phát hiện compression.")
        print(f"   Path gain (trung bình) ≈ {gain_offset_db:+.2f} dB")
        thr, ratio = np.nan, 1.0
    else:
        mask = diff < -0.5
        x, y = rms_in_db[mask], rms_out_db[mask]
        a, b = np.polyfit(x, y, 1)
        ratio = 1.0 / max(a,1e-12)
        thr = b / (1 - a) if abs(1 - a) > 1e-6 else np.nan
        print("\n📊 KẾT QUẢ ƯỚC LƯỢNG COMPRESSION:")
        print(f"   Threshold ≈ {thr:.2f} dBFS")
        print(f"   Ratio     ≈ {ratio:.2f}:1")
        print(f"   Path gain (dưới ngưỡng) ≈ {gain_offset_db:+.2f} dB")

    # --- Vẽ ---
    plt.figure(figsize=(9,6))
    plt.plot(rms_in_db, rms_out_db, 'b.-', label="Compressor curve")
    plt.plot(rms_in_db, rms_in_db, 'k--', label="Line 1:1")
    if not no_compression and np.isfinite(thr):
        plt.axvline(thr, color='g', ls='--', label="Threshold")
    else:
        xgrid = np.linspace(rms_in_db.min(), rms_in_db.max(), 100)
        plt.plot(xgrid, xgrid + gain_offset_db, 'g:', label=f"1:1 + offset ({gain_offset_db:+.2f} dB)")
    plt.xlim(-40,0); plt.ylim(-40,0)
    plt.xlabel("Input level (dBFS)")
    plt.ylabel("Output level (dBFS)")
    plt.title(f"Compression curve @ {freq:.0f} Hz")
    plt.grid(True, ls='--', alpha=0.6)
    plt.legend(); plt.tight_layout(); plt.show()

    # --- Ghi CSV ---
    # thêm để test thư mục
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename_csv = os.path.join(script_dir, "ket_qua_do.csv")
    # end test
    with open(filename_csv, "a", newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if no_compression:
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), "Compression",
                        "No compression", f"Gain {gain_offset_db:+.2f} dB"])
        else:
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), "Compression",
                        f"Thr {thr:.2f} dBFS", f"Ratio {ratio:.2f}:1"])
    print(f"💾 Đã lưu kết quả vào '{filename_csv}'.")

else:
    print("❌ Lựa chọn không hợp lệ.")
