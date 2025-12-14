import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import numpy as np
import threading
import matplotlib.pyplot as plt
import sounddevice as sd
# Import utilities from the shared file
from audio_utilities import (
    OUT_DIR, read_wav_and_normalize, write_wav_from_float, 
    list_output_wavs, convert_to_rms_db, rms_envelope, generate_timebase, EPSILON
)

# --- Compressor & AR Constants ---
VC_THAT4305_mV_per_dB = -6.2
NOISE_FLOOR_LIMIT_DB = -40.0
GR_THRESHOLD_DB = 0.2
WINDOW_MS = 50 

# --- Compressor Analysis Functions (from Tinh_M...) ---
def calculate_makeup_gain(V_in_db, V_out_db_original):
    """Ước lượng Makeup Gain M bằng cách lấy median difference trong vùng low-amplitude."""
    low_idx = np.where(V_in_db < NOISE_FLOOR_LIMIT_DB)[0]
    if len(low_idx) >= 10:
        diffs = V_out_db_original[low_idx] - V_in_db[low_idx]
        return float(np.median(diffs))
    else:
        if len(V_in_db) == 0: return 0.0
        k = max(1, int(0.1 * len(V_in_db)))
        sorted_idx = np.argsort(V_in_db)[:k]
        diffs = V_out_db_original[sorted_idx] - V_in_db[sorted_idx]
        return float(np.median(diffs))

def analyze_compressor_params(V_in, V_out, fs, window_ms=WINDOW_MS):
    """Phân tích Threshold, Ratio, Makeup Gain từ hai tín hiệu WAV."""
    min_len = min(len(V_in), len(V_out))
    if min_len == 0: return None, None, 0.0, np.array([]), np.array([])

    V_in = V_in[:min_len]
    V_out = V_out[:min_len]

    t, V_in_db = convert_to_rms_db(V_in, fs, window_ms)
    _, V_out_db_original = convert_to_rms_db(V_out, fs, window_ms)

    makeup_gain = calculate_makeup_gain(V_in_db, V_out_db_original)
    V_out_db_normalized = V_out_db_original - makeup_gain
    GR_db_normalized = V_in_db - V_out_db_normalized

    compressed_idx = np.where(GR_db_normalized > GR_THRESHOLD_DB)[0]
    if len(compressed_idx) < 5:
        return None, None, makeup_gain, V_in_db, V_out_db_normalized

    threshold_estimate = float(np.min(V_in_db[compressed_idx]))
    V_in_over = V_in_db[compressed_idx]
    V_out_over = V_out_db_normalized[compressed_idx]

    input_excess = V_in_over - threshold_estimate
    output_excess = V_out_over - threshold_estimate

    valid_idx = np.where(output_excess > 0.05)[0]
    if len(valid_idx) < 5:
        return threshold_estimate, None, makeup_gain, V_in_db, V_out_db_normalized

    x = output_excess[valid_idx]
    y = input_excess[valid_idx]
    
    # Fit tuyến tính y = k * x (k ~ ratio)
    denom = np.dot(x, x)
    if denom <= 0: return threshold_estimate, None, makeup_gain, V_in_db, V_out_db_normalized
    k = float(np.dot(x, y) / (denom + EPSILON))

    ratio_estimate = float(np.clip(k, 1.0, 100.0))
    return threshold_estimate, ratio_estimate, makeup_gain, V_in_db, V_out_db_normalized

# --- AR Step Tone Generation & Analysis ---
def generate_step_tone(fs, freq=1000.0, amp=0.8, pre_sil_ms=200, tone_ms=200, post_sil_ms=500, kind='attack'):
    """Tạo tín hiệu bước để kiểm tra AR."""
    pre = int(round(fs * (pre_sil_ms / 1000.0)))
    tone_len = int(round(fs * (tone_ms / 1000.0)))
    post = int(round(fs * (post_sil_ms / 1000.0)))
    t_tone = generate_timebase(fs, tone_ms / 1000.0) if tone_len > 0 else np.array([])
    tone = amp * np.sin(2 * np.pi * freq * t_tone) if tone_len > 0 else np.array([])
    
    if kind == 'attack': signal = np.concatenate([np.zeros(pre), tone, np.zeros(post)])
    elif kind == 'release': signal = np.concatenate([tone, np.zeros(post)])
    elif kind == 'both': signal = np.concatenate([np.zeros(pre), tone, np.zeros(post)])
    else: signal = np.concatenate([np.zeros(pre), tone, np.zeros(post)])
    return signal if signal.size > 0 else np.zeros(1)

def play_and_record(play_sig, fs, record_channels=1):
    """Play stereo and record mono using sounddevice."""
    try:
        if play_sig.ndim == 1:
            play_out = np.column_stack((play_sig, play_sig))
        else:
            play_out = play_sig
        rec = sd.playrec(play_out, samplerate=fs, channels=record_channels)
        sd.wait()
        rec_mono = rec[:, 0] if rec.ndim > 1 else rec
        return rec_mono.astype(np.float64)
    except Exception as e:
        messagebox.showerror("Play/Record error", str(e))
        return None

def detect_attack_release_from_envelopes(env_ref_db, env_rec_db, fs, max_time_ms=500):
    """Phát hiện thời gian Attack và Release từ bao hình RMS."""
    N = len(env_rec_db)
    max_samples = int(round(fs * (max_time_ms / 1000.0)))
    
    # Tìm điểm rise và fall (sử dụng gradient để tìm thay đổi lớn)
    grad_rec = np.gradient(env_rec_db)
    rise_idx, fall_idx = None, None
    try:
        thr_rise_rec = np.percentile(grad_rec, 95)
        if thr_rise_rec > 0.01:
            idxs = np.where(grad_rec >= thr_rise_rec)[0]
            if idxs.size > 0: rise_idx = int(idxs[0])
    except Exception: pass
    try:
        thr_fall_rec = np.percentile(grad_rec, 5)
        if thr_fall_rec < -0.01:
            idxs2 = np.where(grad_rec <= thr_fall_rec)[0]
            if idxs2.size > 0: fall_idx = int(idxs2[0])
    except Exception: pass

    results = {'rise_idx': rise_idx, 'fall_idx': fall_idx}
    attack_ms, release_ms = None, None

    if rise_idx is not None:
        start = rise_idx
        end_search = min(N-1, start + max_samples)
        window = env_rec_db[start:end_search+1]
        if window.size >= 5:
            peak_val = np.max(window)
            start_val = env_rec_db[start]
            target90 = start_val + 0.9 * (peak_val - start_val) # 90% rise
            rel_idxs = np.where(env_rec_db[start:end_search+1] >= target90)[0]
            if rel_idxs.size > 0:
                idx90 = start + int(rel_idxs[0])
                attack_ms = (idx90 - start) / fs * 1000.0
                results['attack_start'] = start
                results['attack_90'] = idx90

    if fall_idx is not None:
        start = fall_idx
        end_search = min(N-1, start + max_samples)
        window = env_rec_db[start:end_search+1]
        if window.size >= 5:
            start_val = env_rec_db[start]
            min_val = np.min(window)
            target10 = min_val + 0.1 * (start_val - min_val) # 10% fall
            rel_idxs = np.where(env_rec_db[start:end_search+1] <= target10)[0]
            if rel_idxs.size > 0:
                idx10 = start + int(rel_idxs[0])
                release_ms = (idx10 - start) / fs * 1000.0
                results['release_start'] = start
                results['release_10'] = idx10

    return attack_ms, release_ms, results

def analyze_attack_release_from_files(fs, ref_sig, rec_sig, rms_window_ms=5, max_time_ms=500):
    if rec_sig is None or len(rec_sig) == 0:
        return None, None, None, None, None
    env_rec = rms_envelope(rec_sig, fs, window_ms=rms_window_ms)
    env_rec_db = 20.0 * np.log10(env_rec + EPSILON)
    if ref_sig is not None and len(ref_sig) > 0 and len(ref_sig) == len(rec_sig):
        env_ref = rms_envelope(ref_sig, fs, window_ms=rms_window_ms)
        env_ref_db = 20.0 * np.log10(env_ref + EPSILON)
    else:
        env_ref_db = np.zeros_like(env_rec_db)
    attack_ms, release_ms, idxs = detect_attack_release_from_envelopes(env_ref_db, env_rec_db, fs, max_time_ms=max_time_ms)
    return attack_ms, release_ms, env_ref_db, env_rec_db, idxs


# --- GUI Class ---
class CompressorARApp:
    def __init__(self, master):
        self.master = master
        master.title("Compressor & AR Analyzer")

        self.comp_orig_file = tk.StringVar(value="")
        self.comp_proc_file = tk.StringVar(value="")
        self.comp_thresh = tk.StringVar(value="-20")
        self.comp_ratio = tk.StringVar(value="4")
        
        self.ar_fs = tk.StringVar(value="44100")
        self.ar_freq = tk.StringVar(value="1000")
        self.ar_amp = tk.StringVar(value="0.8")
        self.ar_pre_ms = tk.StringVar(value="200")
        self.ar_tone_ms = tk.StringVar(value="200")
        self.ar_post_ms = tk.StringVar(value="500")
        self.ar_kind = tk.StringVar(value="attack")
        self.ar_save_name = tk.StringVar(value="step_test.wav")
        self.ar_ref_file = tk.StringVar(value="")
        self.ar_rec_file = tk.StringVar(value="")
        self.ar_rms_win = tk.StringVar(value="5")

        self._build_ui()
        self.refresh_generated_file_lists()

    def _build_ui(self):
        # ... (Compressor Tab UI implementation)
        tab2 = ttk.Frame(self.master, padding=8)
        tab2.pack(fill='both', expand=True)

        top = ttk.Frame(tab2)
        top.pack(fill='x', padx=6, pady=6)
        
        # File Selectors
        ttk.Label(top, text="Original (input) WAV:").grid(row=0, column=0, sticky='w')
        self.orig_menu = self._create_file_menu(top, 0, self.comp_orig_file)
        ttk.Label(top, text="Processed (output) WAV:").grid(row=1, column=0, sticky='w')
        self.proc_menu = self._create_file_menu(top, 1, self.comp_proc_file)

        params = ttk.Frame(tab2)
        params.pack(fill='x', padx=6, pady=(6,8))
        
        ttk.Button(tab2, text="Analyze Compressor (Threshold, Ratio, A/R)", command=self.analyze_compressor).pack(pady=6)
        self.comp_result_text = tk.Text(tab2, height=8)
        self.comp_result_text.pack(fill='both', expand=False, padx=6, pady=6)

        # AR Frame
        ar_frame = ttk.LabelFrame(tab2, text="Attack / Release Measurement (Step Test)")
        ar_frame.pack(fill='x', padx=6, pady=6)
        left_ar = ttk.Frame(ar_frame); left_ar.pack(side='left', fill='y', padx=8, pady=8)
        # AR controls (same as original code)
        ttk.Label(left_ar, text="FS (Hz):").grid(row=0, column=0, sticky='w')
        ttk.Entry(left_ar, textvariable=self.ar_fs, width=10).grid(row=0, column=1, sticky='w')
        ttk.Label(left_ar, text="Freq (Hz):").grid(row=1, column=0, sticky='w')
        ttk.Entry(left_ar, textvariable=self.ar_freq, width=10).grid(row=1, column=1, sticky='w')
        ttk.Label(left_ar, text="Amp (0-1):").grid(row=2, column=0, sticky='w')
        ttk.Entry(left_ar, textvariable=self.ar_amp, width=10).grid(row=2, column=1, sticky='w')
        ttk.Label(left_ar, text="Pre-sil (ms):").grid(row=3, column=0, sticky='w')
        ttk.Entry(left_ar, textvariable=self.ar_pre_ms, width=10).grid(row=3, column=1, sticky='w')
        ttk.Label(left_ar, text="Tone len (ms):").grid(row=4, column=0, sticky='w')
        ttk.Entry(left_ar, textvariable=self.ar_tone_ms, width=10).grid(row=4, column=1, sticky='w')
        ttk.Label(left_ar, text="Post-sil (ms):").grid(row=5, column=0, sticky='w')
        ttk.Entry(left_ar, textvariable=self.ar_post_ms, width=10).grid(row=5, column=1, sticky='w')
        ttk.Label(left_ar, text="Kind:").grid(row=6, column=0, sticky='w')
        ttk.OptionMenu(left_ar, self.ar_kind, "attack", "attack", "release", "both").grid(row=6, column=1, sticky='w')
        ttk.Label(left_ar, text="Save name:").grid(row=7, column=0, sticky='w')
        ttk.Entry(left_ar, textvariable=self.ar_save_name, width=20).grid(row=7, column=1, sticky='w')
        ttk.Button(left_ar, text="Create & Save Step WAV", command=self.create_step_wav).grid(row=8, column=0, columnspan=2, pady=(6,2))
        ttk.Button(left_ar, text="Play&Record Step Test (auto)", command=self.run_step_test_thread).grid(row=9, column=0, columnspan=2, pady=(6,2))

        right_ar = ttk.Frame(ar_frame); right_ar.pack(side='left', fill='both', expand=True, padx=8, pady=8)
        ttk.Label(right_ar, text="(Optional) Reference WAV (original):").grid(row=0, column=0, sticky='w')
        self.ar_ref_menu = self._create_file_menu(right_ar, 0, self.ar_ref_file, column_offset=1)
        ttk.Label(right_ar, text="Recorded/Test WAV:").grid(row=1, column=0, sticky='w', pady=6)
        self.ar_rec_menu = self._create_file_menu(right_ar, 1, self.ar_rec_file, column_offset=1)
        ttk.Label(right_ar, text="RMS Window (ms):").grid(row=2, column=0, sticky='w')
        ttk.Entry(right_ar, textvariable=self.ar_rms_win, width=8).grid(row=2, column=1, sticky='w')
        ttk.Button(right_ar, text="Analyze AR from Selected Files", command=self.analyze_ar_from_files).grid(row=3, column=0, columnspan=3, pady=(8,6))
        self.ar_result_text = tk.Text(right_ar, height=6)
        self.ar_result_text.grid(row=4, column=0, columnspan=3, sticky='nsew', padx=2, pady=2)
        right_ar.rowconfigure(4, weight=1)

    def _create_file_menu(self, parent, row, var, column_offset=0):
        menu = ttk.OptionMenu(parent, var, "", *[])
        menu.grid(row=row, column=1+column_offset, sticky='ew', padx=6)
        ttk.Button(parent, text="Browse...", command=lambda: self._manual_set_file(var)).grid(row=row, column=2+column_offset)
        parent.grid_columnconfigure(1+column_offset, weight=1)
        return menu

    def _manual_set_file(self, var):
        p = filedialog.askopenfilename(filetypes=[('WAV files', '*.wav')], initialdir=OUT_DIR)
        if p: var.set(p)

    def refresh_generated_file_lists(self):
        files = list_output_wavs()
        menus = [self.orig_menu, self.proc_menu, self.ar_ref_menu, self.ar_rec_menu]
        vars = [self.comp_orig_file, self.comp_proc_file, self.ar_ref_file, self.ar_rec_file]
        for menu, var in zip(menus, vars):
            m = menu['menu']
            m.delete(0, 'end')
            for p in files:
                # Use full path for internal variable, display basename
                m.add_command(label=os.path.basename(p), command=lambda v=p: var.set(v))

    def analyze_compressor(self):
        # ... (Implementation of analyze_compressor including plotting)
        orig = self.comp_orig_file.get()
        proc = self.comp_proc_file.get()
        if not orig or not proc:
            messagebox.showwarning("Select files", "Select both original and processed WAV files.")
            return
        fs, sig1 = read_wav_and_normalize(orig)
        _, sig2 = read_wav_and_normalize(proc)
        if fs is None: return

        threshold, ratio, makeup_gain, V_in_db, V_out_db_normalized = analyze_compressor_params(sig1, sig2, fs)
        
        self.comp_result_text.delete(1.0, tk.END)
        self.comp_result_text.insert(tk.END, f"Original: {os.path.basename(orig)}\nProcessed: {os.path.basename(proc)}\n\n")

        if makeup_gain is not None: self.comp_result_text.insert(tk.END, f"Estimated Makeup Gain: {makeup_gain:.2f} dB\n")
        if threshold is not None: self.comp_result_text.insert(tk.END, f"Estimated Threshold (input dB): {threshold:.2f} dB\n")
        else: self.comp_result_text.insert(tk.END, "Estimated Threshold: Not determined\n")
        if ratio is not None: self.comp_result_text.insert(tk.END, f"Estimated Ratio: {ratio:.2f} : 1\n")
        else: self.comp_result_text.insert(tk.END, "Estimated Ratio: Not determined\n")

        if len(V_in_db) > 0:
            GR_db = V_in_db - V_out_db_normalized
            comp_idx = np.where(GR_db > 0.5)[0]
            GdB_repr = float(np.median(GR_db[comp_idx])) if comp_idx.size > 0 else float(np.median(GR_db))
            Vc = GdB_repr * (VC_THAT4305_mV_per_dB / 1000.0)
            self.comp_result_text.insert(tk.END, f"Representative Gain Reduction (G_dB): {GdB_repr:.2f} dB\n")
            self.comp_result_text.insert(tk.END, f"Estimated Vc (THAT4305 model): {Vc:.6f} V\n")

        # Plotting IO-Curve
        try:
            plt.figure(figsize=(10, 7))
            if len(V_in_db) > 0:
                plt.scatter(V_in_db, V_out_db_normalized, alpha=0.35, s=6, label='Dữ liệu RMS')
                min_db, max_db = np.min(V_in_db), np.max(V_in_db)
                x_no_comp = np.linspace(min_db, threshold if threshold is not None else max_db, 80)
                plt.plot(x_no_comp, x_no_comp, linestyle='--', label='Ratio 1:1 (không nén)')
                if threshold is not None and ratio is not None:
                    x_comp = np.linspace(threshold, max_db, 80)
                    y_comp = threshold + (x_comp - threshold) / ratio
                    plt.plot(x_comp, y_comp, linewidth=2.5, label=f'Line ước tính: Ratio {ratio:.2f}:1')
                    plt.axvline(threshold, color='orange', linestyle=':', label=f'Threshold ({threshold:.2f} dB)')
                plt.title('Đường cong Input -> Output (dBFS) [Output đã loại bỏ Makeup Gain]')
                plt.xlabel('V_in (dBFS)'); plt.ylabel('V_out (dBFS) [đã loại bỏ Makeup Gain]'); plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
            else:
                messagebox.showwarning("Không có dữ liệu", "Không có dữ liệu RMS để vẽ IO Curve.")
        except Exception as e:
            messagebox.showerror("Lỗi vẽ", f"Lỗi khi vẽ IO Curve:\n{e}")

    def create_step_wav(self):
        # ... (Implementation of create_step_wav)
        try:
            fs = int(self.ar_fs.get()); freq = float(self.ar_freq.get()); amp = float(self.ar_amp.get())
            pre = int(self.ar_pre_ms.get()); tone = int(self.ar_tone_ms.get()); post = int(self.ar_post_ms.get())
            kind = self.ar_kind.get()
        except Exception as e:
            messagebox.showerror("Invalid params", f"Check AR parameters: {e}")
            return
        sig = generate_step_tone(fs, freq=freq, amp=amp, pre_sil_ms=pre, tone_ms=tone, post_sil_ms=post, kind=kind)
        fname = self.ar_save_name.get().strip();
        if not fname.lower().endswith('.wav'): fname += '.wav'
        outpath = os.path.join(OUT_DIR, fname)
        write_wav_from_float(outpath, fs, sig)
        messagebox.showinfo("Saved", f"Saved step WAV: {outpath}")
        self.refresh_generated_file_lists()

    def run_step_test_thread(self):
        t = threading.Thread(target=self.run_step_test, daemon=True)
        t.start()

    def run_step_test(self):
        # ... (Implementation of run_step_test)
        try:
            fs = int(self.ar_fs.get()); freq = float(self.ar_freq.get()); amp = float(self.ar_amp.get())
            pre = int(self.ar_pre_ms.get()); tone = int(self.ar_tone_ms.get()); post = int(self.ar_post_ms.get())
            kind = self.ar_kind.get()
        except Exception as e:
            messagebox.showerror("Invalid params", f"Check AR parameters: {e}")
            return
        sig = generate_step_tone(fs, freq=freq, amp=amp, pre_sil_ms=pre, tone_ms=tone, post_sil_ms=post, kind=kind)
        rec = play_and_record(sig, fs, record_channels=1)
        if rec is None: return

        out_rec = os.path.join(OUT_DIR, "ar_recorded.wav")
        write_wav_from_float(out_rec, fs, rec)
        self.ar_rec_file.set(out_rec)
        self.refresh_generated_file_lists()
        
        attack_ms, release_ms, env_ref_db, env_rec_db, idxs = analyze_attack_release_from_files(
            fs, sig, rec, rms_window_ms=int(self.ar_rms_win.get()), max_time_ms=500
        )
        self.ar_result_text.delete(1.0, tk.END)
        if attack_ms is not None: self.ar_result_text.insert(tk.END, f"Attack (auto): {attack_ms:.2f} ms\n")
        else: self.ar_result_text.insert(tk.END, "Attack (auto): Not detected\n")
        if release_ms is not None: self.ar_result_text.insert(tk.END, f"Release (auto): {release_ms:.2f} ms\n")
        else: self.ar_result_text.insert(tk.END, "Release (auto): Not detected\n")
        self.plot_ar_envelopes(env_ref_db, env_rec_db, fs, idxs)
    
    def analyze_ar_from_files(self):
        # ... (Implementation of analyze_ar_from_files)
        refp = self.ar_ref_file.get().strip(); recp = self.ar_rec_file.get().strip()
        if not recp:
            messagebox.showwarning("Select file", "Select a recorded/test WAV file to analyze.")
            return
        ref_sig, fs = None, None
        if refp: fs, ref_sig = read_wav_and_normalize(refp)
        fs_rec, rec_sig = read_wav_and_normalize(recp)
        if fs_rec is None: return
        if fs is None: fs = fs_rec
        if fs != fs_rec:
            messagebox.showerror("Fs mismatch", "Sampling rates must match.")
            return
        try: rms_win = int(self.ar_rms_win.get())
        except Exception: rms_win = 5
        
        attack_ms, release_ms, env_ref_db, env_rec_db, idxs = analyze_attack_release_from_files(
            fs, ref_sig, rec_sig, rms_window_ms=rms_win, max_time_ms=500
        )
        self.ar_result_text.delete(1.0, tk.END)
        if attack_ms is not None: self.ar_result_text.insert(tk.END, f"Attack (auto): {attack_ms:.2f} ms\n")
        else: self.ar_result_text.insert(tk.END, "Attack (auto): Not detected\n")
        if release_ms is not None: self.ar_result_text.insert(tk.END, f"Release (auto): {release_ms:.2f} ms\n")
        else: self.ar_result_text.insert(tk.END, "Release (auto): Not detected\n")
        self.plot_ar_envelopes(env_ref_db, env_rec_db, fs, idxs)

    def plot_ar_envelopes(self, env_ref_db, env_rec_db, fs, idxs):
        # ... (Implementation of plot_ar_envelopes)
        try:
            t = np.arange(len(env_rec_db)) / float(fs)
            plt.figure(figsize=(10,5))
            plt.plot(t, env_rec_db, label='Recorded (dB)')
            if env_ref_db is not None and env_ref_db.size == env_rec_db.size:
                plt.plot(t, env_ref_db, label='Reference (dB)', alpha=0.7)
            if isinstance(idxs, dict):
                if idxs.get('attack_start') is not None: plt.axvline(idxs['attack_start']/fs, color='orange', linestyle='--', label='attack start')
                if idxs.get('attack_90') is not None: plt.axvline(idxs['attack_90']/fs, color='red', linestyle='--', label='attack 90%')
                if idxs.get('release_start') is not None: plt.axvline(idxs['release_start']/fs, color='purple', linestyle='--', label='release start')
                if idxs.get('release_10') is not None: plt.axvline(idxs['release_10']/fs, color='green', linestyle='--', label='release 10%')
            plt.xlabel('Time (s)'); plt.ylabel('RMS (dB)'); plt.title('Attack / Release envelopes'); plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
        except Exception as e:
            messagebox.showerror("Plot AR error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = CompressorARApp(root)
    root.mainloop()
