import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv, time
import threading # Cần thiết để chạy các test HW mà không làm treo GUI

# ============================================================
# HẰNG SỐ CHUNG (CONSTANTS)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
OUT_DIR = os.path.join(BASE_DIR, "_sóng_sin")
ACCENT = '#0078d4'
FILENAME_CSV = os.path.join(BASE_DIR, "ket_qua_do.csv")
FS = 48000 # Tần số lấy mẫu mặc định
THD_DURATION = 2.0
COMP_SEG_DUR, COMP_GAP_DUR = 0.25, 0.05
COMP_AMPS_COUNT = 36

# ============================================================
# HÀM PHỤ TRỢ (TỪ MÃ LOGIC THỨ HAI)
# ============================================================
def safe_rms(x):
    """Tính RMS an toàn, tránh lỗi chia cho zero."""
    return np.sqrt(np.mean(np.square(x))) if len(x) > 0 else 0.0

def thd_analysis(signal, fs, freq, h_max=5):
    """Phân tích THD cho tín hiệu đã ghi."""
    N = len(signal)
    # Áp dụng cửa sổ Hanning
    fft = np.fft.rfft(signal * np.hanning(N))
    mag = np.abs(fft)
    freqs = np.fft.rfftfreq(N, 1/fs)
    
    # Tìm Tần số cơ bản (Fundamental)
    fund_idx = np.argmin(np.abs(freqs - freq))
    fund_mag = mag[fund_idx]
    
    harmonics = {}
    total_harmonic_power = 0
    
    # Tìm các hài (Harmonics)
    for n in range(2, h_max + 1):
        idx = np.argmin(np.abs(freqs - n*freq))
        # Chuyển đổi sang dB so với tần số cơ bản (dBc)
        v_db = 20 * np.log10(mag[idx] / fund_mag + 1e-12)
        harmonics[n] = v_db
        # Tính lũy thừa cho tổng công suất hài
        total_harmonic_power += (mag[idx] / fund_mag)**2

    thd_ratio = np.sqrt(total_harmonic_power)
    thd_percent = thd_ratio * 100
    thd_db = 20 * np.log10(thd_ratio + 1e-12)
    
    return harmonics, thd_percent, thd_db, fund_mag, freqs, mag


# ============================================================
# CLASS CHÍNH: AudioAnalysisToolkitApp (CHỈ PHẦN GUI + LOGIC)
# ============================================================
class AudioAnalysisToolkitApp:
    def __init__(self, master):
        self.master = master
        master.title("Audio Analysis Suite v3.3 (Loopback File Analysis Added)")
        master.geometry("1400x900")
        
        # --- Khởi tạo Styles ---
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Accent.TButton', foreground='white', background=ACCENT)

        # --- Khởi tạo Variables ---
        self.hw_freq = tk.StringVar(value="1000")
        self.hw_amp = tk.StringVar(value="0.7")
        self.hw_input_dev = tk.StringVar()
        self.hw_output_dev = tk.StringVar()
        self.hw_loop_file = tk.StringVar(value="")
        self.hw_ar_rms_win = tk.StringVar(value="5")
        self.hw_thd_hmax = tk.StringVar(value="5")
        
        # Các thành phần GUI sẽ được gán sau
        self.hw_log_text = None
        self.cb_in = None
        self.cb_out = None
        self.fig = None
        self.canvas = None
        
        # --- Xây dựng UI ---
        self._build_ui()
        self._refresh_hw_devices()

    # ============================================================
    # PHƯƠNG THỨC HỖ TRỢ UI/DATA (HELPERS)
    # ============================================================
    
    def hw_log(self, message):
        """Ghi log ra màn hình Tab 4"""
        if self.hw_log_text:
            self.hw_log_text.insert(tk.END, message + "\n")
            self.hw_log_text.see(tk.END)

    def _refresh_hw_devices(self):
        """Làm mới danh sách thiết bị âm thanh"""
        try:
            devs = sd.query_devices()
            inputs = []
            outputs = []
            for i, d in enumerate(devs):
                name = f"{i}: {d['name']}"
                if d['max_input_channels'] > 0: inputs.append(name)
                if d['max_output_channels'] > 0: outputs.append(name)
            
            if self.cb_in: self.cb_in['values'] = inputs
            if self.cb_out: self.cb_out['values'] = outputs
            
            # Chọn thiết bị mặc định hoặc cái đầu tiên
            default_in = sd.default.device[0]
            default_out = sd.default.device[1]
            if self.cb_in and inputs and default_in < len(inputs):
                self.cb_in.current(default_in)
            elif self.cb_in and inputs:
                self.cb_in.current(0)
                
            if self.cb_out and outputs and default_out < len(outputs):
                self.cb_out.current(default_out)
            elif self.cb_out and outputs:
                self.cb_out.current(0)
                
            self.hw_log("Đã làm mới danh sách thiết bị âm thanh.")
        except Exception as e:
            self.hw_log(f"Lỗi refresh thiết bị: {e}. Vui lòng kiểm tra sounddevice.")
            
    def _get_hw_indices(self, input_var, output_var):
        """Lấy chỉ số Input/Output đã chọn từ StringVar"""
        try:
            in_str = input_var.get()
            out_str = output_var.get()
            if not in_str or not out_str:
                self.hw_log("❌ Lỗi: Chưa chọn thiết bị Input/Output.")
                return None, None
            return int(in_str.split(':')[0]), int(out_str.split(':')[0])
        except Exception as e:
            self.hw_log(f"❌ Lỗi đọc chỉ số thiết bị: {e}")
            return None, None

    def select_hw_loop_file(self):
        """Mở cửa sổ chọn file thủ công cho HW Loopback"""
        p = filedialog.askopenfilename(filetypes=[('WAV files', '*.wav')])
        if p: self.hw_loop_file.set(p)

    def _update_plot(self, fig):
        """Cập nhật biểu đồ lên canvas"""
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        
        self.fig = fig
        self.canvas = FigureCanvasTkAgg(fig, master=self.right_panel)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill='both', expand=True)
        self.canvas.draw()
        
    def _clear_plot(self):
        """Xóa biểu đồ cũ"""
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.fig = None
        self.canvas = None
    
    # ============================================================
    # PHƯƠNG THỨC XÂY DỰNG GIAO DIỆN (UI BUILDING METHODS)
    # ============================================================

    def _build_ui(self):
        """Xây dựng toàn bộ giao diện"""
        nb = ttk.Notebook(self.master)
        nb.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Chỉ xây dựng Tab 4 theo yêu cầu
        self._build_tab4(nb)

    def _build_tab4(self, parent_notebook):
        """ Hardware Loopback (Real-time tests)"""
        self.tab_hw = ttk.Frame(parent_notebook)
        parent_notebook.add(self.tab_hw, text="4. Hardware Loopback (Real-time)")

        # 1. Khu vực chọn Soundcard 
        dev_frame = ttk.LabelFrame(self.tab_hw, text="⚙️ Cấu hình Soundcard (Input/Output)")
        dev_frame.pack(fill='x', padx=10, pady=5)
        ttk.Label(dev_frame, text="Input Device:").grid(row=0, column=0, sticky='e', padx=5)
        self.cb_in = ttk.Combobox(dev_frame, textvariable=self.hw_input_dev, width=45, state='readonly')
        self.cb_in.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(dev_frame, text="Output Device:").grid(row=0, column=2, sticky='e', padx=5)
        self.cb_out = ttk.Combobox(dev_frame, textvariable=self.hw_output_dev, width=45, state='readonly')
        self.cb_out.grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(dev_frame, text="Làm mới (Refresh)", command=self._refresh_hw_devices).grid(row=0, column=4, padx=10)

        # 2. Khu vực điều khiển và Log
        paned = ttk.PanedWindow(self.tab_hw, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=10, pady=5)

        left_panel = ttk.Frame(paned)
        paned.add(left_panel, weight=1)

        # A. ĐO COMPRESSOR 
        grp_comp = ttk.LabelFrame(left_panel, text="A. Đo Compressor (Stepped Sweep)")
        grp_comp.pack(fill='x', pady=5, padx=5)
        lbl_info = ttk.Label(grp_comp, text=f"Quét {COMP_AMPS_COUNT} mức ({COMP_SEG_DUR}s/mức) - Phát hiện Thr, Ratio, Makeup Gain", foreground="blue")
        lbl_info.pack(padx=5, pady=5)
        f_param = ttk.Frame(grp_comp)
        f_param.pack(fill='x', padx=5)
        ttk.Label(f_param, text="Freq (Hz):").pack(side='left')
        ttk.Entry(f_param, textvariable=self.hw_freq, width=8).pack(side='left', padx=5)
        # Tích hợp hàm chạy
        btn_comp = ttk.Button(grp_comp, text="▶ CHẠY TEST COMPRESSOR (HW)", command=self._start_compressor_test, style='Accent.TButton')
        btn_comp.pack(fill='x', padx=10, pady=10)

        # B. ĐO THD 
        grp_thd = ttk.LabelFrame(left_panel, text="B. Đo THD (Harmonic Distortion)")
        grp_thd.pack(fill='x', pady=5, padx=5)
        f_thd = ttk.Frame(grp_thd)
        f_thd.pack(fill='x', padx=5)
        ttk.Label(f_thd, text="Amp (0-1):").pack(side='left')
        ttk.Entry(f_thd, textvariable=self.hw_amp, width=8).pack(side='left', padx=5)
        ttk.Label(f_thd, text="Max H:").pack(side='left', padx=5)
        ttk.Entry(f_thd, textvariable=self.hw_thd_hmax, width=4).pack(side='left')
        # Tích hợp hàm chạy
        ttk.Button(grp_thd, text="▶ CHẠY TEST THD (HW)", command=self._start_thd_test).pack(fill='x', padx=10, pady=10)

        # C. ĐO ATTACK/RELEASE (Giữ nguyên Placeholder)
        grp_ar = ttk.LabelFrame(left_panel, text="C. Đo Attack / Release (Step Tone)")
        grp_ar.pack(fill='x', pady=5, padx=5)
        ttk.Button(grp_ar, text="▶ CHẠY TEST A/R (Step Tone - HW)", command=lambda: self.hw_log("Chức năng C: Đo Attack/Release chưa được định nghĩa logic.")).pack(fill='x', padx=10, pady=10)
        
        # D. LOOPBACK BẰNG FILE TÙY CHỌN (Giữ nguyên Placeholder)
        grp_file = ttk.LabelFrame(left_panel, text="D. Loopback & Phân tích File tùy chọn")
        grp_file.pack(fill='x', pady=5, padx=5)
        
        f_file_sel = ttk.Frame(grp_file)
        f_file_sel.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(f_file_sel, text="File WAV Input:").grid(row=0, column=0, sticky='w')
        self.hw_file_entry = ttk.Entry(f_file_sel, textvariable=self.hw_loop_file, width=30)
        self.hw_file_entry.grid(row=0, column=1, sticky='we', padx=5)
        ttk.Button(f_file_sel, text="Browse...", command=self.select_hw_loop_file).grid(row=0, column=2, padx=5)
        f_file_sel.grid_columnconfigure(1, weight=1)
        
        btn_file_loop = ttk.Button(grp_file, text="1. ▶ CHẠY LOOPBACK & SAVE (Mọi File)", command=lambda: self.hw_log("Chức năng D1: Loopback File đã được kích hoạt (cần logic Threading)"), style='Accent.TButton')
        btn_file_loop.pack(fill='x', padx=10, pady=10)
        
        f_analysis = ttk.LabelFrame(grp_file, text="Phân tích File Ghi âm (Cần File Input tương ứng)")
        f_analysis.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(f_analysis, text="A. Phân tích Compressor (Cần Stepped Sweep)", command=lambda: self.hw_log("Chức năng D-A: Phân tích Compressor (File) chưa được định nghĩa logic.")).pack(fill='x', padx=5, pady=2)
        
        f_thd_opts = ttk.Frame(f_analysis)
        f_thd_opts.pack(fill='x', padx=5)
        ttk.Button(f_thd_opts, text="B. Phân tích THD (Cần Sine Tone)", command=lambda: self.hw_log("Chức năng D-B: Phân tích THD (File) chưa được định nghĩa logic.")).pack(side='left', fill='x', expand=True, pady=2)
        ttk.Label(f_thd_opts, text="Max H:").pack(side='left', padx=5)
        ttk.Entry(f_thd_opts, textvariable=self.hw_thd_hmax, width=4, state='readonly').pack(side='left') # Dùng chung Hmax nhưng chỉ hiển thị
        
        f_ar_opts = ttk.Frame(f_analysis)
        f_ar_opts.pack(fill='x', padx=5)
        ttk.Button(f_ar_opts, text="C. Phân tích A/R (Cần Step Tone)", command=lambda: self.hw_log("Chức năng D-C: Phân tích A/R (File) chưa được định nghĩa logic.")).pack(side='left', fill='x', expand=True, pady=2)
        ttk.Label(f_ar_opts, text="RMS Win (ms):").pack(side='left', padx=5)
        ttk.Entry(f_ar_opts, textvariable=self.hw_ar_rms_win, width=4, state='readonly').pack(side='left') # Dùng chung RMS Win

        # Log Text Area và Plot Area
        self.right_panel = ttk.Frame(paned) # Gán frame vào self để _update_plot có thể dùng
        paned.add(self.right_panel, weight=2)
        
        # Log Text Area
        log_frame = ttk.LabelFrame(self.right_panel, text="Nhật ký (Logs):")
        log_frame.pack(fill='x', padx=5, pady=5)
        self.hw_log_text = tk.Text(log_frame, height=10, bg="#f4f4f4", font=("Consolas", 10))
        self.hw_log_text.pack(fill='both', expand=True)
        self.hw_log("🛠️ Sẵn sàng cho các bài kiểm tra phần cứng.")
        
        # Plot Area (Ban đầu trống)
        self.plot_label = ttk.Label(self.right_panel, text="📈 Kết quả (Vẽ đồ thị):")
        self.plot_label.pack(anchor='w', padx=5, pady=5)


    # ============================================================
    # PHƯƠNG THỨC LOGIC CHÍNH (MAIN LOGIC METHODS - TỪ MÃ THỨ HAI)
    # ============================================================

    def _get_hw_params(self):
        """Lấy và xác thực các tham số HW cơ bản."""
        in_idx, out_idx = self._get_hw_indices(self.hw_input_dev, self.hw_output_dev)
        if in_idx is None or out_idx is None: return None
        
        try:
            freq = float(self.hw_freq.get())
            amp = float(self.hw_amp.get())
            h_max = int(self.hw_thd_hmax.get())
            if not (10 <= freq <= 20000 and 0 < amp <= 1.0 and h_max >= 2):
                raise ValueError("Tham số không hợp lệ.")
            return {'in_idx': in_idx, 'out_idx': out_idx, 'freq': freq, 'amp': amp, 'h_max': h_max}
        except ValueError as e:
            self.hw_log(f"❌ Lỗi tham số đầu vào: {e}. Vui lòng kiểm tra Freq, Amp, Hmax.")
            return None

    # --- THD Test ---
    def _run_thd_test_thread(self, params):
        """Logic THD chạy trong Thread."""
        self.hw_log(f"🎙️ Bắt đầu test THD: {params['freq']:.0f} Hz, Amp {params['amp']:.2f}, Max H {params['h_max']} ...")
        
        try:
            sd.default.device = (params['in_idx'], params['out_idx'])
            sd.default.samplerate = FS
            
            t = np.linspace(0, THD_DURATION, int(FS*THD_DURATION), endpoint=False)
            sine = params['amp'] * np.sin(2*np.pi*params['freq']*t)
            # Phát Stereo, ghi Mono (channels=1)
            sine_stereo = np.column_stack((sine, np.zeros_like(sine))) 
            
            rec = sd.playrec(sine_stereo, samplerate=FS, channels=1, dtype='float32')
            sd.wait()
            sig = np.squeeze(rec)
            
            self.hw_log("✅ Ghi xong tín hiệu.")
            
            # Xử lý tín hiệu
            sig = sig[int(0.05*FS):] # Bỏ 50 ms đầu
            sig /= np.max(np.abs(sig)) # Chuẩn hóa về max 1.0
            
            # === PHÂN TÍCH THD ===
            harmonics, thd_percent, thd_db, _, freqs, mag = thd_analysis(sig, FS, params['freq'], params['h_max'])
            
            # === VẼ ===
            fig, ax = plt.subplots(2, 1, figsize=(10, 8))
            
            # --- 10 chu kỳ ở giữa tín hiệu ---
            cycles = int(FS/params['freq']*10)
            mid = len(sig)//2
            ax[0].plot(np.arange(cycles)/FS*1000, sig[mid:mid+cycles], 'b')
            ax[0].set_title(f"Recorded Waveform (10 cycles @ {params['freq']:.0f} Hz)")
            ax[0].set_xlabel("Time (ms)")
            ax[0].set_ylabel("Amplitude")
            ax[0].grid(True)

            # --- Phổ FFT ---
            db_mag = 20 * np.log10(mag / np.max(mag))
            ax[1].plot(freqs, db_mag, 'r')
            ax[1].set_xlim(0, params['freq'] * params['h_max'] * 1.5)
            ax[1].set_ylim(-100, 0)
            ax[1].set_title(f"FFT Spectrum ({params['freq']:.0f} Hz test tone)")
            ax[1].set_xlabel("Frequency (Hz)")
            ax[1].set_ylabel("Magnitude (dBFS)")
            ax[1].grid(True)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            self.master.after(0, lambda: self._update_plot(fig)) # Cập nhật plot trên luồng chính
            
            # === Ghi Log và CSV ===
            log_msg = "\n🎵 THD / Harmonic Analysis:"
            for n, v in harmonics.items():
                 log_msg += f"\n  H{n}: {n*params['freq']:.1f} Hz, {v:.2f} dBc"
            log_msg += f"\n🔸 THD = {thd_percent:.4f}% ({thd_db:.2f} dB)"
            self.master.after(0, lambda: self.hw_log(log_msg))
            
            with open(FILENAME_CSV, "a", newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), "THD",
                            f"{params['freq']:.0f}Hz, {params['amp']:.2f}Amp",
                            f"THD {thd_percent:.4f}%", f"{thd_db:.2f} dB"])
            self.master.after(0, lambda: self.hw_log(f"💾 Đã lưu kết quả vào '{FILENAME_CSV}'."))

        except Exception as e:
            self.master.after(0, lambda: self.hw_log(f"❌ Lỗi trong quá trình THD test: {e}"))


    def _start_thd_test(self):
        """Bắt đầu Thread cho THD test."""
        params = self._get_hw_params()
        if params is None: return
        self._clear_plot()
        threading.Thread(target=self._run_thd_test_thread, args=(params,)).start()

    # --- Compression Test ---
    def _run_compressor_test_thread(self, params):
        """Logic Compression chạy trong Thread."""
        self.hw_log(f"\n🎛️ Bắt đầu test Compressor @ {params['freq']:.0f} Hz ({COMP_AMPS_COUNT} mức)...")

        try:
            sd.default.device = (params['in_idx'], params['out_idx'])
            sd.default.samplerate = FS
            
            seg_dur, gap_dur = COMP_SEG_DUR, COMP_GAP_DUR
            amps = np.linspace(0.05, 0.98, COMP_AMPS_COUNT)
            protect = 0.98
            t_seg = np.linspace(0, seg_dur, int(FS*seg_dur), endpoint=False)
            gap = np.zeros(int(FS*gap_dur))
            
            # Tạo tín hiệu Step Sweep
            tx = np.concatenate([np.concatenate((min(A,protect)*np.sin(2*np.pi*params['freq']*t_seg), gap)) for A in amps])
            
            self.hw_log(f"🔊 Tổng thời gian test ~{len(tx)/FS:.1f}s.")
            rx = sd.playrec(tx, samplerate=FS, channels=1, dtype='float32')
            sd.wait()
            rx = np.squeeze(rx)
            
            self.hw_log("✅ Ghi xong tín hiệu.")
            
            # --- Phân tích ---
            segN, gapN = int(seg_dur*FS), int(gap_dur*FS)
            trim_lead, trim_tail = int(0.03*FS), int(0.01*FS) # Bỏ phần Attack/Release nhanh ở đầu/cuối segment
            rms_in_db, rms_out_db = [], []

            for A, i in zip(amps, range(len(amps))):
                s0, s1 = i*(segN+gapN), i*(segN+gapN)+segN
                seg = rx[s0:s1]
                # Trim segment
                seg = seg[trim_lead:max(trim_lead, len(seg)-trim_tail)]
                
                # Tính RMS Input/Output (dBFS)
                rin = max(A/np.sqrt(2), 1e-12)
                rout = max(safe_rms(seg), 1e-12)
                rms_in_db.append(20*np.log10(rin))
                rms_out_db.append(20*np.log10(rout))
            
            rms_in_db, rms_out_db = np.array(rms_in_db), np.array(rms_out_db)
            diff = rms_out_db - rms_in_db

            # Kiểm tra "Không nén"
            a_all, b_all = np.polyfit(rms_in_db, rms_out_db, 1)
            gain_offset_db = np.mean(diff)
            slope_tol, spread_tol = 0.05, 1.0
            no_compression = (abs(a_all - 1.0) < slope_tol) and ((diff.max()-diff.min()) < spread_tol)

            if no_compression:
                thr, ratio = np.nan, 1.0
                log_msg = f"\n📊 KẾT QUẢ: Không phát hiện compression."
                log_msg += f"\n  Path gain (trung bình) ≈ {gain_offset_db:+.2f} dB"
            else:
                # Ước lượng Threshold & Ratio
                mask = diff < -0.5
                if np.sum(mask) >= 2: # Cần ít nhất 2 điểm để fit đường thẳng
                    x, y = rms_in_db[mask], rms_out_db[mask]
                    a, b = np.polyfit(x, y, 1)
                    ratio = 1.0 / max(a, 1e-12)
                    thr = b / (1 - a) if abs(1 - a) > 1e-6 else np.nan
                else:
                    thr, ratio = np.nan, np.nan
                    
                log_msg = "\n📊 KẾT QUẢ ƯỚC LƯỢNG COMPRESSION:"
                log_msg += f"\n  Threshold ≈ {thr:.2f} dBFS"
                log_msg += f"\n  Ratio     ≈ {ratio:.2f}:1"
                log_msg += f"\n  Path gain (dưới ngưỡng) ≈ {gain_offset_db:+.2f} dB"

            self.master.after(0, lambda: self.hw_log(log_msg))

            # --- Vẽ ---
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.plot(rms_in_db, rms_out_db, 'b.-', label="Compressor curve")
            ax.plot(rms_in_db, rms_in_db, 'k--', label="Line 1:1")
            
            if not no_compression and np.isfinite(thr):
                # Vẽ mô hình nén
                x_pre = np.linspace(rms_in_db.min(), thr, 10)
                x_post = np.linspace(thr, rms_in_db.max(), 10)
                y_pre = x_pre + gain_offset_db # Giả định gain offset là gain dưới ngưỡng
                y_post = thr + (x_post - thr)/ratio # Mô hình nén
                
                ax.plot(x_pre, y_pre, 'g-', label="Pre-Threshold Model")
                ax.plot(x_post, y_post, 'r-', label="Post-Threshold Model")
                ax.axvline(thr, color='g', ls='--', label=f"Threshold ({thr:.2f} dBFS)")
                
            else:
                xgrid = np.linspace(rms_in_db.min(), rms_in_db.max(), 100)
                ax.plot(xgrid, xgrid + gain_offset_db, 'g:', label=f"1:1 + offset ({gain_offset_db:+.2f} dB)")
                
            ax.set_xlim(-40, 0)
            ax.set_ylim(-40, 0)
            ax.set_xlabel("Input level (dBFS)")
            ax.set_ylabel("Output level (dBFS)")
            ax.set_title(f"Compression curve @ {params['freq']:.0f} Hz")
            ax.grid(True, ls='--', alpha=0.6)
            ax.legend(); 
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            self.master.after(0, lambda: self._update_plot(fig))

            # --- Ghi CSV ---
            with open(FILENAME_CSV, "a", newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                if no_compression:
                    w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), "Compression",
                                f"{params['freq']:.0f}Hz",
                                "No compression", f"Gain {gain_offset_db:+.2f} dB"])
                else:
                    w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), "Compression",
                                f"{params['freq']:.0f}Hz",
                                f"Thr {thr:.2f} dBFS", f"Ratio {ratio:.2f}:1"])
            self.master.after(0, lambda: self.hw_log(f"💾 Đã lưu kết quả vào '{FILENAME_CSV}'."))

        except Exception as e:
            self.master.after(0, lambda: self.hw_log(f"❌ Lỗi trong quá trình Compressor test: {e}"))


    def _start_compressor_test(self):
        """Bắt đầu Thread cho Compression test."""
        params = self._get_hw_params()
        if params is None: return
        self._clear_plot()
        # Loại bỏ Amp vì nó không được dùng trong bài test nén, nhưng cần Freq
        threading.Thread(target=self._run_compressor_test_thread, args=(params,)).start()

# ============================================================
# KHỐI THỰC THI (RUN BLOCK)
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAnalysisToolkitApp(root)
    root.mainloop()
