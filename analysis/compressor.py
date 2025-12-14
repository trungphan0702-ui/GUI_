import numpy as np
from typing import Dict, Any


def build_stepped_tone(freq: float, fs: int, amp_max: float = 1.0) -> Dict[str, Any]:
    seg_dur, gap_dur = 0.25, 0.05
    amps = np.linspace(0.05, amp_max, 36)
    t_seg = np.linspace(0, seg_dur, int(fs * seg_dur), endpoint=False)
    gap = np.zeros(int(fs * gap_dur))
    tx = np.concatenate([
        np.concatenate((min(a, amp_max) * np.sin(2 * np.pi * freq * t_seg), gap)) for a in amps
    ])
    meta = {
        'seg_samples': int(seg_dur * fs),
        'gap_samples': int(gap_dur * fs),
        'amps': amps,
    }
    return {'signal': tx.astype(np.float32), 'meta': meta}


def apply_compressor(sig: np.ndarray, threshold_db: float = -18.0, ratio: float = 3.0,
                     makeup_db: float = 0.0, knee_db: float = 0.0) -> np.ndarray:
    """Apply a simple static compressor curve for testing.

    Args:
        sig: Input signal (float32 array).
        threshold_db: Threshold in dBFS where compression begins.
        ratio: Compression ratio above threshold.
        makeup_db: Makeup gain applied after compression.
        knee_db: Optional soft knee width.
    """

    sig = np.asarray(sig, dtype=np.float32)
    eps = 1e-8
    abs_sig = np.abs(sig) + eps
    level_db = 20 * np.log10(abs_sig)
    over_db = level_db - threshold_db

    if knee_db > 0:
        # Soft knee interpolation
        knee_start = -knee_db / 2
        soft_region = over_db / max(knee_db, eps)
        soft_region = np.clip(soft_region, 0.0, 1.0)
        soft_gain_db = soft_region * (over_db * (1 - 1 / ratio))
        over_db = np.where(over_db > knee_start, over_db, 0.0)
    else:
        soft_gain_db = 0.0

    compressed_db = np.where(over_db > 0, threshold_db + over_db / ratio, level_db)
    compressed_db += makeup_db
    gain_db = compressed_db - level_db + soft_gain_db
    gain_lin = 10 ** (gain_db / 20.0)
    return (sig * gain_lin).astype(np.float32)


def compression_curve(sig: np.ndarray, meta: Dict[str, Any], fs: int, freq: float) -> Dict[str, Any]:
    segN = meta['seg_samples']
    gapN = meta['gap_samples']
    amps = meta['amps']
    trim_lead, trim_tail = int(0.03 * fs), int(0.01 * fs)
    rms_in_db, rms_out_db = [], []
    for A, i in zip(amps, range(len(amps))):
        s0 = i * (segN + gapN)
        s1 = s0 + segN
        seg = sig[s0:s1]
        seg = seg[trim_lead:max(trim_lead, len(seg) - trim_tail)]
        rin = max(A / np.sqrt(2), 1e-12)
        rout = max(np.sqrt(np.mean(np.square(seg))), 1e-12)
        rms_in_db.append(20 * np.log10(rin))
        rms_out_db.append(20 * np.log10(rout))
    rms_in_db = np.array(rms_in_db)
    rms_out_db = np.array(rms_out_db)
    diff = rms_out_db - rms_in_db
    a_all, b_all = np.polyfit(rms_in_db, rms_out_db, 1)
    gain_offset_db = float(np.mean(diff))
    slope_tol, spread_tol = 0.05, 1.0
    no_compression = (abs(a_all - 1.0) < slope_tol) and ((diff.max() - diff.min()) < spread_tol)

    if no_compression:
        thr, ratio = np.nan, 1.0
    else:
        mask = diff < -0.5
        x, y = rms_in_db[mask], rms_out_db[mask]
        a, b = np.polyfit(x, y, 1)
        ratio = 1.0 / max(a, 1e-12)
        thr = b / (1 - a) if abs(1 - a) > 1e-6 else np.nan
    return {
        'in_db': rms_in_db,
        'out_db': rms_out_db,
        'gain_offset_db': gain_offset_db,
        'no_compression': no_compression,
        'thr_db': thr,
        'ratio': ratio,
    }


def compare_compression(input_sig: np.ndarray, output_sig: np.ndarray, fs: int, freq: float) -> Dict[str, Any]:
    meta = build_stepped_tone(freq, fs)
    base_curve = compression_curve(input_sig, meta['meta'], fs, freq)
    out_curve = compression_curve(output_sig, meta['meta'], fs, freq)
    delta_thr = out_curve['thr_db'] - base_curve['thr_db'] if np.isfinite(out_curve['thr_db']) and np.isfinite(base_curve['thr_db']) else np.nan
    delta_ratio = out_curve['ratio'] - base_curve['ratio']
    delta_gain = out_curve['gain_offset_db'] - base_curve['gain_offset_db']
    return {
        'input': base_curve,
        'output': out_curve,
        'delta_thr': delta_thr,
        'delta_ratio': delta_ratio,
        'delta_gain': delta_gain,
    }
