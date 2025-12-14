import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading

# Try to import sounddevice for device listing
try:
    import sounddevice as sd
except Exception:
    sd = None

# ============================================================
# CONSTANTS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
ACCENT = '#0b66c3'
BTN_FONT = (None, 10, 'bold')
LOG_FONT = ('Consolas', 10)

# ============================================================
# Scrollable Frame Class (LEFT PANEL)
# ============================================================
class ScrollableFrame(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)

        canvas = tk.Canvas(self, borderwidth=0, background="#fafafa")
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.canvas = canvas


# ============================================================
# MAIN APP
# ============================================================
class AudioAnalysisToolkitApp:
    def __init__(self, master):
        self.master = master
        master.title("Audio Analysis Suite v3.4 – UI Upgraded")
        master.geometry("1400x900")

        # Variables
        self.hw_freq = tk.StringVar(value='1000')
        self.hw_amp = tk.StringVar(value='0.7')
        self.hw_input_dev = tk.StringVar()
        self.hw_output_dev = tk.StringVar()
        self.hw_loop_file = tk.StringVar(value='')
        self.hw_ar_rms_win = tk.StringVar(value='5')
        self.hw_thd_hmax = tk.StringVar(value='5')
        self.thd_max_h = tk.StringVar(value='5')

        self._configure_style()
        self._build_ui()
        self._refresh_hw_devices()

    # ---------------------------------------------------------
    # STYLE
    # ---------------------------------------------------------
    def _configure_style(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TFrame", background="#fafafa")
        style.configure("TLabelframe", background="#fafafa")
        style.configure("TLabelframe.Label", background="#fafafa", foreground=ACCENT, font=('Segoe UI', 11, 'bold'))
        style.configure("TLabel", background="#fafafa", font=('Segoe UI', 10))
        style.configure("TEntry", padding=4)
        style.configure("Accent.TButton", foreground="white", background=ACCENT, font=BTN_FONT)
        style.map("Accent.TButton", background=[("active", "#094f99")])

    # ---------------------------------------------------------
    # LOG
    # ---------------------------------------------------------
    def hw_log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    # ---------------------------------------------------------
    # DEVICE REFRESH
    # ---------------------------------------------------------
    def _refresh_hw_devices(self):
        if sd is None:
            self.hw_log("Sounddevice không khả dụng.")
            return
        try:
            devs = sd.query_devices()
            inputs, outputs = [], []
            for i, d in enumerate(devs):
                name = f"{i}: {d['name']}"
                if d.get('max_input_channels', 0) > 0:
                    inputs.append(name)
                if d.get('max_output_channels', 0) > 0:
                    outputs.append(name)

            self.cb_in['values'] = inputs
            self.cb_out['values'] = outputs

            if inputs:
                self.cb_in.current(0)
                self.hw_input_dev.set(inputs[0])
            if outputs:
                self.cb_out.current(0)
                self.hw_output_dev.set(outputs[0])

            self.hw_log("Đã làm mới danh sách thiết bị âm thanh.")
        except Exception as e:
            self.hw_log(f"Lỗi khi lấy thiết bị: {e}")

    # ---------------------------------------------------------
    def select_hw_loop_file(self):
        p = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if p:
            self.hw_loop_file.set(p)
            self.hw_log(f"Đã chọn file: {p}")

    # ---------------------------------------------------------
    # BUILD UI
    # ---------------------------------------------------------
    def _build_ui(self):

        # Notebook
        nb = ttk.Notebook(self.master)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        tab_hw = ttk.Frame(nb)
        nb.add(tab_hw, text="4. Hardware Loopback (Real-time)")

        # Top device frame
        dev_frame = ttk.LabelFrame(tab_hw, text="Cấu hình Soundcard (Input / Output)")
        dev_frame.pack(fill="x", padx=8, pady=6)

        ttk.Label(dev_frame, text="Input Device:").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        self.cb_in = ttk.Combobox(dev_frame, textvariable=self.hw_input_dev, width=60, state="readonly")
        self.cb_in.grid(row=0, column=1, sticky="w", padx=4)

        ttk.Label(dev_frame, text="Output Device:").grid(row=0, column=2, sticky="e", padx=4)
        self.cb_out = ttk.Combobox(dev_frame, textvariable=self.hw_output_dev, width=60, state="readonly")
        self.cb_out.grid(row=0, column=3, sticky="w", padx=4)

        ttk.Button(dev_frame, text="Làm mới", command=self._refresh_hw_devices).grid(row=0, column=4, padx=6)

        # PanedWindow
        paned = ttk.PanedWindow(tab_hw, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=6)

        # LEFT (scrollable)
        left_container = ttk.Frame(paned)
        paned.add(left_container, weight=1)

        scroll_left = ScrollableFrame(left_container)
        scroll_left.pack(fill="both", expand=True)
        left = scroll_left.scrollable_frame

        # -------------------------------------------------
        # SECTION A
        grp_a = ttk.LabelFrame(left, text="A. Đo Compressor (Stepped Sweep)")
        grp_a.pack(fill="x", padx=6, pady=8)

        ttk.Label(grp_a, text="Quét 36 mức (0.25s/mức) – Tìm Thr, Ratio, Makeup Gain", foreground="blue").pack(anchor="w", padx=6, pady=4)

        f = ttk.Frame(grp_a)
        f.pack(fill="x", padx=6, pady=4)
        ttk.Label(f, text="Freq (Hz):").pack(side="left")
        ttk.Entry(f, textvariable=self.hw_freq, width=8).pack(side="left", padx=6)

        ttk.Button(grp_a, text="▶ CHẠY TEST COMPRESSOR (HW)",
                   style="Accent.TButton",
                   command=lambda: self.hw_log("Test Compressor (chưa cài)")
                   ).pack(fill="x", padx=6, pady=8)

        # -------------------------------------------------
        # SECTION B
        grp_b = ttk.LabelFrame(left, text="B. Đo THD (Harmonic Distortion)")
        grp_b.pack(fill="x", padx=6, pady=8)

        fb = ttk.Frame(grp_b)
        fb.pack(fill="x", padx=6, pady=4)
        ttk.Label(fb, text="Amp (0-1):").pack(side="left")
        ttk.Entry(fb, textvariable=self.hw_amp, width=8).pack(side="left", padx=6)
        ttk.Label(fb, text="Max H:").pack(side="left", padx=(10, 2))
        ttk.Entry(fb, textvariable=self.thd_max_h, width=4).pack(side="left")

        ttk.Button(grp_b, text="▶ CHẠY TEST THD (HW)",
                   command=lambda: self.hw_log("Test THD (chưa cài)")
                   ).pack(fill="x", padx=6, pady=8)

        # -------------------------------------------------
        # SECTION C
        grp_c = ttk.LabelFrame(left, text="C. Đo Attack / Release (Step Tone)")
        grp_c.pack(fill="x", padx=6, pady=8)

        ttk.Button(grp_c, text="▶ CHẠY TEST A/R (HW)",
                   command=lambda: self.hw_log("Test AR (chưa cài)")
                   ).pack(fill="x", padx=6, pady=8)

        far = ttk.Frame(grp_c)
        far.pack(fill="x", padx=6, pady=4)
        ttk.Label(far, text="RMS Win (ms):").pack(side="left")
        ttk.Entry(far, textvariable=self.hw_ar_rms_win, width=6).pack(side="left", padx=6)

        # -------------------------------------------------
        # SECTION D
        grp_d = ttk.LabelFrame(left, text="D. Loopback & Phân tích File")
        grp_d.pack(fill="x", padx=6, pady=8)

        ffile = ttk.Frame(grp_d)
        ffile.pack(fill="x", padx=6, pady=6)
        ttk.Label(ffile, text="File WAV Input:").grid(row=0, column=0, sticky="w")
        ttk.Entry(ffile, textvariable=self.hw_loop_file, width=40).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(ffile, text="Browse...", command=self.select_hw_loop_file).grid(row=0, column=2, padx=6)
        ffile.grid_columnconfigure(1, weight=1)

        ttk.Button(grp_d, text="1. ▶ CHẠY LOOPBACK & SAVE (All Files)",
                   style="Accent.TButton",
                   command=lambda: self.hw_log("Loopback placeholder")
                   ).pack(fill="x", padx=6, pady=8)

        # Sub-analysis
        ana = ttk.LabelFrame(grp_d, text="Phân tích File Ghi âm")
        ana.pack(fill="x", padx=6, pady=4)

        ttk.Button(ana, text="A. Phân tích Compressor",
                   command=lambda: self.hw_log("Phân tích A")).pack(fill="x", padx=6, pady=4)

        f_thd = ttk.Frame(ana)
        f_thd.pack(fill="x", padx=6, pady=4)
        ttk.Button(f_thd, text="B. Phân tích THD",
                   command=lambda: self.hw_log("Phân tích THD")
                   ).pack(side="left", expand=True, fill="x")
        ttk.Label(f_thd, text="Max H:").pack(side="left", padx=6)
        ttk.Entry(f_thd, textvariable=self.hw_thd_hmax, width=4).pack(side="left")

        f_ar2 = ttk.Frame(ana)
        f_ar2.pack(fill="x", padx=6, pady=4)
        ttk.Button(f_ar2, text="C. Phân tích A/R",
                   command=lambda: self.hw_log("Phân tích AR")).pack(side="left", expand=True, fill="x")
        ttk.Label(f_ar2, text="RMS Win (ms):").pack(side="left", padx=6)
        ttk.Entry(f_ar2, textvariable=self.hw_ar_rms_win, width=4).pack(side="left")

        # -------------------------------------------------
        # SECTION E
        grp_e = ttk.LabelFrame(left, text="E. Phân tích 2 File Offline")
        grp_e.pack(fill="x", padx=6, pady=8)

        fe = ttk.Frame(grp_e)
        fe.pack(fill="x", padx=6, pady=6)

        ttk.Label(fe, text="File Input:").grid(row=0, column=0)
        ttk.Entry(fe, width=30).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(fe, text="Browse...").grid(row=0, column=2, padx=6)

        ttk.Label(fe, text="File Output:").grid(row=1, column=0)
        ttk.Entry(fe, width=30).grid(row=1, column=1, sticky="we", padx=6)
        ttk.Button(fe, text="Browse...").grid(row=1, column=2, padx=6)
        fe.grid_columnconfigure(1, weight=1)

        small_ana = ttk.Frame(grp_e)
        small_ana.pack(fill="x", padx=6, pady=4)
        ttk.Button(small_ana, text="A. Phân tích Compressor").pack(fill="x", pady=2)
        ttk.Button(small_ana, text="B. Phân tích THD").pack(fill="x", pady=2)
        ttk.Button(small_ana, text="C. Phân tích A/R").pack(fill="x", pady=2)

        # -------------------------------------------------
        # RIGHT PANEL: LOGS
        right = ttk.Frame(paned)
        paned.add(right, weight=2)

        ttk.Label(right, text="Nhật ký (Logs):", background="#fafafa").pack(anchor="w", padx=6)

        log_frame = ttk.Frame(right)
        log_frame.pack(fill="both", expand=True, padx=6, pady=6)

        self.log_text = tk.Text(log_frame, font=LOG_FONT, bg="#f4f4f4", wrap="none")
        self.log_text.pack(side="left", fill="both", expand=True)

        scroll_log = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scroll_log.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scroll_log.set)

        # Startup log
        self.hw_log("[Khởi động] Làm mới danh sách thiết bị âm thanh.")

    # ---------------------------------------------------------
    def _now_str(self):
        import datetime
        return datetime.datetime.now().strftime("%H:%M:%S")


# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAnalysisToolkitApp(root)
    root.mainloop()
