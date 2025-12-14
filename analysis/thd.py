import numpy as np
from typing import Dict, Any


def compute_thd(signal: np.ndarray, fs: int, freq: float, max_h: int = 5) -> Dict[str, Any]:
    """Compute THD and THD+N for a (nearly) single-tone signal."""

    sig = np.asarray(signal, dtype=np.float32)
    sig = sig - np.mean(sig)
    if sig.ndim > 1:
        sig = sig[:, 0]

    window = np.hanning(len(sig))
    spec = np.fft.rfft(sig * window)
    freqs = np.fft.rfftfreq(len(sig), 1 / fs)
    mag = np.abs(spec)
    power_spectrum = mag ** 2

    fund_idx = np.argmin(np.abs(freqs - freq))
    fund = mag[fund_idx] + 1e-12
    fundamental_power = power_spectrum[fund_idx]

    harmonics = {}
    power_sum = 0.0
    for h in range(2, max_h + 1):
        idx = np.argmin(np.abs(freqs - h * freq))
        val = mag[idx]
        harmonics[h] = 20 * np.log10(val / fund + 1e-12)
        power_sum += (val / fund) ** 2

    thd_ratio = np.sqrt(power_sum)
    thd_percent = thd_ratio * 100
    thd_db = 20 * np.log10(thd_ratio + 1e-12)

    total_power = np.sum(power_spectrum)
    noise_power = max(total_power - fundamental_power, 1e-12)
    thdn_ratio = np.sqrt(noise_power / fundamental_power)
    thdn_db = 20 * np.log10(thdn_ratio + 1e-12)

    return {
        'fundamental_mag': fund,
        'harmonics_dbc': harmonics,
        'thd_percent': thd_percent,
        'thd_db': thd_db,
        'thdn_db': thdn_db,
        'freqs': freqs,
        'spectrum': 20 * np.log10(mag + 1e-12)
    }
