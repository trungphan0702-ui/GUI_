import os
import json
import numpy as np
from scipy.io import wavfile
from scipy.signal import windows
import matplotlib.pyplot as plt
import sounddevice as sd

# Config / Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "_sóng_sin")
os.makedirs(OUT_DIR, exist_ok=True)
HARM_FILE = os.path.join(OUT_DIR, "harmonics.json")
EPSILON = 1e-12
WINDOW_MS = 50 

# --- File Utilities ---
def list_output_wavs():
    files = []
    for f in sorted(os.listdir(OUT_DIR)):
        if f.lower().endswith('.wav'):
            files.append(os.path.join(OUT_DIR, f))
    return files

def read_wav_and_normalize(path):
    """Đọc WAV, lấy kênh mono (nếu stereo lấy kênh đầu), trả về fs, float signal trong [-1,1]."""
    try:
        fs, data = wavfile.read(path)
    except Exception:
        return None, None
    if getattr(data, "ndim", 1) > 1:
        data = data[:, 0]
    dtype = data.dtype
    if np.issubdtype(dtype, np.integer):
        max_val = float(np.iinfo(dtype).max)
        if dtype == np.uint8:
            data = (data.astype(np.float64) - 128.0) / 128.0
        else:
            data = data.astype(np.float64) / (max_val + 1.0)
    elif np.issubdtype(dtype, np.floating):
        data = data.astype(np.float64)
        maxv = np.max(np.abs(data)) + EPSILON
        if maxv > 1.0:
            data = data / maxv
    else:
        return None, None
    return fs, data

def write_wav_from_float(path, fs, data):
    data = np.clip(data, -1.0, 1.0)
    int16 = (data * 32767.0).astype(np.int16)
    wavfile.write(path, fs, int16)

def save_harmonics_dict(hdict):
    try:
        with open(HARM_FILE, 'w', encoding='utf-8') as f:
            json.dump(hdict, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Failed to save harmonics:", e)

def load_harmonics_dict():
    if os.path.exists(HARM_FILE):
        try:
            with open(HARM_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

# --- Signal Processing Utilities ---
def rms_envelope(x, fs, window_ms=WINDOW_MS):
    win = int(fs * (window_ms / 1000.0))
    if win < 1: win = 1
    kernel = np.ones(win) / win
    # Sử dụng np.convolve cho tính toán bao hình RMS
    env = np.sqrt(np.convolve(x**2, kernel, mode='same'))
    return env

def convert_to_rms_db(signal, fs, window_ms=WINDOW_MS):
    """Tính RMS theo khối cố định, trả về time axis và giá trị dBFS."""
    window_samples = max(1, int(round(fs * window_ms / 1000.0)))
    n = len(signal)
    n_blocks = n // window_samples
    if n_blocks == 0:
        rms = np.sqrt(np.mean(signal**2)) + EPSILON
        rms_db = np.array([20.0 * np.log10(rms)])
        time_axis = np.array([0.0])
        return time_axis, rms_db
    trimmed = signal[: n_blocks * window_samples]
    blocks = trimmed.reshape(n_blocks, window_samples)
    rms_per_block = np.sqrt(np.mean(blocks**2, axis=1)) + EPSILON
    rms_db = 20.0 * np.log10(rms_per_block)
    time_axis = (np.arange(n_blocks) * window_samples + window_samples/2.0) / float(fs)
    return time_axis, rms_db

def generate_timebase(fs, duration):
    return np.linspace(0, duration, int(fs * duration), endpoint=False)

def play_last_created(files):
    if not files:
        print("No file available.")
        return
    p = files[-1]
    fs, data = read_wav_and_normalize(p)
    if fs is None:
        return
    try:
        sd.play(data, samplerate=fs)
        sd.wait()
    except Exception as e:
        print("Playback error:", e)
