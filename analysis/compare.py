import numpy as np
from typing import Dict, Any, Tuple
from . import thd


def align_signals(ref: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    ref_mono = ref if ref.ndim == 1 else ref[:, 0]
    tgt_mono = target if target.ndim == 1 else target[:, 0]
    n = min(len(ref_mono), len(tgt_mono))
    ref_mono = ref_mono[:n]
    tgt_mono = tgt_mono[:n]
    corr = np.correlate(tgt_mono, ref_mono, mode='full')
    lag = int(np.argmax(corr) - (len(ref_mono) - 1))
    if lag > 0:
        aligned_ref = ref[lag:]
        aligned_tgt = target[:len(aligned_ref)]
    else:
        aligned_tgt = target[-lag:]
        aligned_ref = ref[:len(aligned_tgt)]
    min_len = min(len(aligned_ref), len(aligned_tgt))
    return aligned_ref[:min_len], aligned_tgt[:min_len], lag


def gain_match(ref: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
    if ref.ndim > 1:
        ref_mono = ref[:, 0]
    else:
        ref_mono = ref
    if target.ndim > 1:
        tgt_mono = target[:, 0]
    else:
        tgt_mono = target
    rms_ref = np.sqrt(np.mean(ref_mono ** 2) + 1e-12)
    rms_tgt = np.sqrt(np.mean(tgt_mono ** 2) + 1e-12)
    gain = rms_ref / max(rms_tgt, 1e-12)
    return target * gain, 20 * np.log10(1.0 / gain + 1e-12)


def residual_metrics(ref: np.ndarray, tgt: np.ndarray, fs: int, freq: float, hmax: int = 5) -> Dict[str, Any]:
    residual = tgt - ref
    res_rms = np.sqrt(np.mean(residual ** 2) + 1e-12)
    ref_rms = np.sqrt(np.mean(ref ** 2) + 1e-12)
    snr = 20 * np.log10(ref_rms / res_rms + 1e-12)
    noise_floor = 20 * np.log10(res_rms + 1e-12)
    thd_ref = thd.compute_thd(ref, fs, freq, hmax)
    thd_tgt = thd.compute_thd(tgt, fs, freq, hmax)
    thd_delta = thd_tgt['thd_db'] - thd_ref['thd_db']
    window = np.hanning(len(ref))
    spec_ref = np.fft.rfft(ref * window)
    spec_tgt = np.fft.rfft(tgt * window)
    mag_ref = 20 * np.log10(np.abs(spec_ref) + 1e-12)
    mag_tgt = 20 * np.log10(np.abs(spec_tgt) + 1e-12)
    fr_dev = mag_tgt - mag_ref
    fr_mean_dev = float(np.median(fr_dev))
    hum_bins = []
    freqs = np.fft.rfftfreq(len(ref), 1 / fs)
    for base in (50, 60):
        for mul in range(1, 5):
            f = base * mul
            idx = np.argmin(np.abs(freqs - f))
            hum_bins.append({'freq': f, 'level_db': float(mag_tgt[idx])})
    return {
        'residual_rms_dbfs': 20 * np.log10(res_rms + 1e-12),
        'snr_db': snr,
        'noise_floor_dbfs': noise_floor,
        'thd_ref_db': thd_ref['thd_db'],
        'thd_tgt_db': thd_tgt['thd_db'],
        'thd_delta_db': thd_delta,
        'fr_dev_median_db': fr_mean_dev,
        'hum_peaks': hum_bins,
        'residual': residual,
    }
