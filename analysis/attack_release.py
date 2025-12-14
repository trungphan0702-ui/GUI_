import numpy as np
from typing import Dict, Any


def generate_step_tone(freq: float, fs: int, amp: float = 0.7, duration: float = 2.0) -> np.ndarray:
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    tone = amp * np.sin(2 * np.pi * freq * t)
    # amplitude steps: first half low, second half high
    env = np.ones_like(tone)
    env[: len(env) // 2] *= 0.3
    return (tone * env).astype(np.float32)


def envelope_rms(sig: np.ndarray, fs: int, win_ms: float) -> np.ndarray:
    win = int(max(1, fs * win_ms / 1000))
    if sig.ndim > 1:
        sig = sig[:, 0]
    padded = np.pad(sig ** 2, (win, win))
    cumsum = np.cumsum(padded)
    rms = np.sqrt((cumsum[2 * win:] - cumsum[:-2 * win]) / win)
    return rms


def attack_release_times(sig: np.ndarray, fs: int, win_ms: float) -> Dict[str, float]:
    env = envelope_rms(sig, fs, win_ms)
    if len(env) < 10:
        return {'attack_ms': float('nan'), 'release_ms': float('nan')}
    mid = len(env) // 2
    first, second = env[:mid], env[mid:]
    target = np.max(second)
    start_val = np.max(first)
    if target <= 0:
        return {'attack_ms': float('nan'), 'release_ms': float('nan')}
    attack_idx = np.argmax(env >= target * 0.9)
    release_idx = mid + np.argmax(second <= target * 0.37)
    return {
        'attack_ms': attack_idx / fs * 1000.0,
        'release_ms': release_idx / fs * 1000.0,
    }


def compare_attack_release(input_sig: np.ndarray, output_sig: np.ndarray, fs: int, win_ms: float) -> Dict[str, Any]:
    in_times = attack_release_times(input_sig, fs, win_ms)
    out_times = attack_release_times(output_sig, fs, win_ms)
    return {
        'input': in_times,
        'output': out_times,
        'delta_attack': out_times['attack_ms'] - in_times['attack_ms'],
        'delta_release': out_times['release_ms'] - in_times['release_ms'],
    }
