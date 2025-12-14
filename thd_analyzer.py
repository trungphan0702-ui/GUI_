import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
# Import utilities from the shared file
from audio_utilities import OUT_DIR, read_wav_and_normalize, list_output_wavs

# --- THD Analysis Function ---
def analyze_distortion(signal, fs, analysis_duration_s=1.0, harmonic_count=10):
    """Phân tích méo hài (THD) bằng FFT."""
    chunk_size = int(fs * analysis_duration_s)
    if len(signal) >= chunk_size:
        sig = signal[:chunk_size]
    else:
        sig = signal
    N = len(sig)
    if N < 32: return None
    
    window = windows.hann(N, sym=False)
    sig_w = sig * window
    yf = np.fft.fft(sig_w)
    yf_half = yf[:N // 2]
    lin_mag = np.abs(yf_half) / (N / 2)
    mag_db = 20 * np.log10(lin_mag + 1e-12)
    freqs = np.fft.fftfreq(N, 1.0 / fs)[:N // 2]
    
    # Find Fundamental (F0)
    idx0 = np.argmin(np.abs(freqs - 20.0))
    idx_fund = idx0 + np.argmax(mag_db[idx0:]) if idx0 < len(mag_db) else 0
    f0 = freqs[idx_fund] if idx_fund < len(freqs) else 0.0
    V1_lin = lin_mag[idx_fund] if idx_fund < len(lin_mag) else 0.0
    V1_db = mag_db[idx_fund] if idx_fund < len(mag_db) else -999.0
    
    # Calculate Harmonics
    harmonic_power = 0.0
    indiv = {}
    for h in range(2, harmonic_count + 1):
        fh = f0 * h
        if fh >= fs / 2: break
        idxh = np.argmin(np.abs(freqs - fh))
        Vh_db = mag_db[idxh]
        Vh_lin = lin_mag[idxh]
        if Vh_db > -120:
            harmonic_power += Vh_lin ** 2
            indiv[h] = {'percent': (Vh_lin / (V1_lin + 1e-12)) * 100.0, 'dB': Vh_db, 'freq': fh}
            
    THD_percent = np.sqrt(harmonic_power) / (V1_lin + 1e-12) * 100.0 if V1_lin > 0 else 0.0
    
    return {
        'frequencies': freqs, 'magnitude_dB': mag_db, 'fundamental_freq': f0,
        'V1_db': V1_db, 'individual': indiv, 'THD_percent': THD_percent,
        'time_domain': sig, 'fs': fs
    }


# --- GUI Class ---
class THDAnalyzerApp:
    def __init__(self, master):
        self.master = master
        master.title("THD / FFT Analyzer")

        self.thd_file = tk.StringVar(value="")
        self.thd_duration = tk.StringVar(value="1.0")
        self.thd_hmax = tk.StringVar(value="10")

        self._build_ui()
        self.refresh_generated_file_lists()

    def _build_ui(self):
        tab3 = ttk.Frame(self.master, padding=8)
        tab3.pack(fill='both', expand=True)

        top3 = ttk.Frame(tab3)
        top3.pack(fill='x', padx=6, pady=6)
        
        ttk.Label(top3, text="Choose file:").grid(row=0, column=0, sticky='w')
        self.thd_menu = self._create_file_menu(top3, 0, self.thd_file)
        
        ttk.Label(top3, text="Analysis duration (s):").grid(row=1, column=0, sticky='w', pady=6)
        ttk.Entry(top3, textvariable=self.thd_duration, width=8).grid(row=1, column=1, sticky='w')
        
        ttk.Label(top3, text="Harmonics max (2..10):").grid(row=2, column=0, sticky='w')
        ttk.Entry(top3, textvariable=self.thd_hmax, width=8).grid(row=2, column=1, sticky='w')
        
        ttk.Button(tab3, text="Analyze THD & Plot", command=self.analyze_thd_button).pack(pady=6)
        self.thd_text = tk.Text(tab3, height=10)
        self.thd_text.pack(fill='both', padx=6, pady=6)

    def _create_file_menu(self, parent, row, var):
        menu = ttk.OptionMenu(parent, var, "", *[])
        menu.grid(row=row, column=1, sticky='ew', padx=6)
        ttk.Button(parent, text="Browse...", command=lambda: self._manual_set_file(var)).grid(row=row, column=2)
        parent.grid_columnconfigure(1, weight=1)
        return menu

    def _manual_set_file(self, var):
        p = filedialog.askopenfilename(filetypes=[('WAV files', '*.wav')], initialdir=OUT_DIR)
        if p: var.set(p)

    def refresh_generated_file_lists(self):
        files = list_output_wavs()
        menu = self.thd_menu['menu']
        menu.delete(0, 'end')
        for p in files:
            menu.add_command(label=os.path.basename(p), command=lambda v=p: self.thd_file.set(v))

    def analyze_thd_button(self):
        f = self.thd_file.get()
        if not f:
            messagebox.showwarning("Select file", "Choose a WAV file to analyze.")
            return
            
        fs, data = read_wav_and_normalize(f)
        if fs is None: return
        
        try: dur = float(self.thd_duration.get())
        except Exception: dur = 1.0
        
        try: hmax = int(self.thd_hmax.get())
        except Exception: hmax = 10
        if hmax < 2: hmax = 10
            
        res = analyze_distortion(data, fs, analysis_duration_s=dur, harmonic_count=hmax)
        if res is None:
            messagebox.showwarning("Too short", "Signal too short for FFT analysis.")
            return
            
        self.thd_text.delete(1.0, tk.END)
        self.thd_text.insert(tk.END, f"File: {os.path.basename(f)}\nFs: {fs} Hz\n")
        self.thd_text.insert(tk.END, f"F0: {res['fundamental_freq']:.2f} Hz\nV1: {res['V1_db']:.2f} dB\nTHD (to H{hmax}): {res['THD_percent']:.4f}%\n\n")
        self.thd_text.insert(tk.END, "Harmonics (dB, % of fundamental):\n")
        for h, v in sorted(res['individual'].items()):
            self.thd_text.insert(tk.END, f"H{h}: {v['dB']:.2f} dB ({v['percent']:.2f}%) @ {v['freq']:.2f} Hz\n")

        # Plotting
        try:
            sig = res['time_domain']
            t = np.arange(len(sig)) / fs
            freqs = res['frequencies']
            mag = res['magnitude_dB']
            
            plt.figure(figsize=(12, 8))
            
            plt.subplot(3, 1, 1); plt.plot(t, sig); plt.title("Time domain (first analyzed chunk)")
            plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
            
            plt.subplot(3, 1, 2); plt.plot(freqs, mag); plt.xscale('log')
            plt.xlim(20, min(fs/2, 20000)); plt.title("FFT (dB)"); plt.xlabel("Frequency (Hz)"); plt.ylabel("Magnitude (dB)")
            
            plt.subplot(3, 1, 3)
            hs = sorted(res['individual'].items())
            if hs:
                orders = [f"H{h}" for h, _ in hs]
                vals = [v['dB'] for _, v in hs]
                plt.bar(orders, vals); plt.title(f"Harmonic levels (H2..H{hmax})"); plt.ylabel("dB")
            else:
                plt.text(0.2, 0.5, "No significant harmonics detected", fontsize=12)
            plt.tight_layout(); plt.show()
        except Exception as e:
            messagebox.showerror("Plotting error", f"Error plotting results:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = THDAnalyzerApp(root)
    root.mainloop()
