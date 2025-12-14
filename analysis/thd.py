import numpy as np
from typing import Dict, Any, Optional


def _window_fn(name: str, n: int) -> np.ndarray:
    name = (name or "").lower()
    if name in ("hann", "hanning", "han"):
        return np.hanning(n)
    if name in ("rect", "boxcar", "rectangular"):
        return np.ones(n)
    raise ValueError(f"Unsupported window '{name}'. Use 'hann' or 'rect'.")


def compute_thd(
    signal: np.ndarray,
    fs: int,
    f0: Optional[float] = None,
    max_h: int = 10,
    *,
    window: str = "hann",
    freq: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute THD and THD+N for a (nearly) single-tone signal.

    Contract (used by tests and GUI):
        - positional fundamental is `f0` (legacy alias `freq` supported as kw-only)
        - returns thd_percent, thd_db, thdn_percent, thdn_db
    """

    fundamental_freq = f0 if f0 is not None else freq
    if fundamental_freq is None:
        raise ValueError("compute_thd requires a fundamental frequency via `f0` or `freq`.")

    sig = np.asarray(signal, dtype=np.float32)
    sig = sig - np.mean(sig)
    if sig.ndim > 1:
        sig = sig[:, 0]

    win = _window_fn(window, len(sig))
    spec = np.fft.rfft(sig * win)
    freqs = np.fft.rfftfreq(len(sig), 1 / fs)
    mag = np.abs(spec)
    power_spectrum = mag ** 2

    fund_idx = np.argmin(np.abs(freqs - fundamental_freq))
    fund = mag[fund_idx] + 1e-20
    fundamental_power = power_spectrum[fund_idx] + 1e-20

    harmonics = {}
    power_sum = 0.0
    for h in range(2, max_h + 1):
        idx = np.argmin(np.abs(freqs - h * fundamental_freq))
        val = mag[idx]
        harmonics[h] = 20 * np.log10(val / fund + 1e-12)
        power_sum += (val / fund) ** 2

    thd_ratio = np.sqrt(power_sum)
    thd_percent = float(thd_ratio * 100)
    thd_db = float(20 * np.log10(thd_ratio + 1e-12))

    # THD+N = RMS(all bins except the fundamental) / RMS(fundamental)
    noise_power = float(np.sum(power_spectrum) - fundamental_power)
    noise_power = max(noise_power, 1e-20)
    thdn_ratio = np.sqrt(noise_power / fundamental_power)
    thdn_percent = float(thdn_ratio * 100)
    thdn_db = float(20 * np.log10(thdn_ratio + 1e-12))

    result = {
        'fundamental_mag': float(fund),
        'harmonics_dbc': harmonics,
        'thd_percent': thd_percent,
        'thd_db': thd_db,
        'thdn_percent': thdn_percent,
        'thdn_db': thdn_db,
        'freqs': freqs,
        'spectrum': 20 * np.log10(mag + 1e-12)
    }

    # Final contract enforcement in case future refactors drop keys.
    contract_keys = {
        'thd_percent': thd_percent,
        'thd_db': thd_db,
        'thdn_percent': thdn_percent,
        'thdn_db': thdn_db,
    }
    for k, v in contract_keys.items():
        result.setdefault(k, v)

    return result
