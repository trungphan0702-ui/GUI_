"""Self-test harness for offline DSP validation and optional hardware smoke test.

Usage:
    python -m tests.self_test --mode offline
    python -m tests.self_test --mode hardware --in-dev <id> --out-dev <id>
"""

import argparse
import math
import sys
import threading

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from analysis import attack_release, compare, compressor, thd
from audio import playrec, wav_io


def _banner(msg: str) -> None:
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)


def _result(name: str, ok: bool, detail: str) -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")
    return ok


def _sine(freq: float, fs: int, duration: float, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def offline_thd_tests(fs: int = 48000) -> bool:
    freq = 1000.0
    clean = _sine(freq, fs, 1.5, amp=0.5)
    thd_clean = thd.compute_thd(clean, fs, freq)
    ok_clean = thd_clean['thd_db'] < -60 and thd_clean['thdn_db'] < -50

    dist = clean + 0.05 * np.sin(2 * np.pi * 2 * freq * np.arange(len(clean)) / fs)
    dist += 0.02 * np.sin(2 * np.pi * 3 * freq * np.arange(len(clean)) / fs)
    thd_dist = thd.compute_thd(dist, fs, freq)
    ok_dist = thd_dist['thd_db'] > -40 and thd_dist['thdn_db'] > -40

    ok = True
    ok &= _result("THD clean sine", ok_clean, f"THD={thd_clean['thd_db']:.2f} dB, THD+N={thd_clean['thdn_db']:.2f} dB")
    ok &= _result("THD distorted sine", ok_dist, f"THD={thd_dist['thd_db']:.2f} dB, THD+N={thd_dist['thdn_db']:.2f} dB")
    return ok


def offline_attack_release(fs: int = 48000) -> bool:
    sig = attack_release.generate_step_tone(freq=1000.0, fs=fs, amp=0.8, duration=2.0)
    times = attack_release.attack_release_times(sig, fs, win_ms=10.0)
    ok = math.isfinite(times['attack_ms']) and math.isfinite(times['release_ms']) and times['attack_ms'] < times['release_ms']
    return _result("Attack/Release envelope", ok, f"attack={times['attack_ms']:.1f} ms, release={times['release_ms']:.1f} ms")


def offline_compressor(fs: int = 48000) -> bool:
    freq = 1000.0
    stepped = compressor.build_stepped_tone(freq, fs, amp_max=0.9)
    threshold_db, ratio = -18.0, 3.0
    makeup_db = 6.0
    comp_out = compressor.apply_compressor(stepped['signal'], threshold_db, ratio, makeup_db, knee_db=3.0)
    base_curve = compressor.compression_curve(stepped['signal'], stepped['meta'], fs, freq)
    comp_curve = compressor.compression_curve(comp_out, stepped['meta'], fs, freq)
    thr_err = abs(comp_curve['thr_db'] - threshold_db) if math.isfinite(comp_curve['thr_db']) else float('inf')
    ratio_err = abs(comp_curve['ratio'] - ratio)
    ok = thr_err < 3.0 and ratio_err < 1.0
    detail = f"thr_est={comp_curve['thr_db']:.1f} dB (target {threshold_db} dB), ratio_est={comp_curve['ratio']:.2f} (target {ratio})"
    return _result("Compressor estimator", ok, detail)


def offline_compare(fs: int = 48000) -> bool:
    freq = 1000.0
    ref = _sine(freq, fs, 1.0, amp=0.5)
    delay_samples = int(0.005 * fs)
    gain = 0.8
    noise = np.random.randn(len(ref)) * 0.001
    tgt = np.concatenate([np.zeros(delay_samples), ref * gain])[: len(ref)] + noise
    metrics = compare.compare_signals(ref, tgt, fs, freq)

    latency_ok = abs(metrics['latency_samples'] - delay_samples) < 10
    gain_ok = abs(metrics['gain_error_db'] - (20 * np.log10(gain))) < 1.0
    residual_ok = metrics['residual_rms_dbfs'] < -30
    detail = (
        f"latency={metrics['latency_ms']:.2f} ms (expected {delay_samples/fs*1000:.2f}), "
        f"gain_err={metrics['gain_error_db']:.2f} dB (expected {20*np.log10(gain):.2f}), "
        f"residual={metrics['residual_rms_dbfs']:.1f} dBFS, SNR={metrics['snr_db']:.1f} dB"
    )
    return _result("Compare (align+gain+residual)", latency_ok and gain_ok and residual_ok, detail)


def run_offline_suite() -> bool:
    _banner("Running offline DSP self-tests")
    ok = True
    ok &= offline_thd_tests()
    ok &= offline_attack_release()
    ok &= offline_compressor()
    ok &= offline_compare()
    return ok


def run_hardware_smoke(in_dev: int, out_dev: int, fs: int = 48000) -> bool:
    _banner("Running hardware smoke test")
    print("Required: sounddevice/PortAudio, mic permission enabled, at least one input + one output device, and routing for loopback.")
    try:
        import sounddevice as sd
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        return _result("Hardware availability", False, f"sounddevice missing: {exc}")

    try:
        devs = sd.query_devices()
    except Exception as exc:
        return _result("Hardware availability", False, f"PortAudio error: {exc}")

    has_in = any(d.get('max_input_channels', 0) > 0 for d in devs)
    has_out = any(d.get('max_output_channels', 0) > 0 for d in devs)
    if not (has_in and has_out):
        print("No usable input/output devices detected. Ensure microphone/speaker permissions and hardware are available.")
        return _result("Hardware availability", True, "SKIPPED (resource missing)")

    freq = 1000.0
    tone = _sine(freq, fs, 1.0, amp=0.5)
    stop_evt = threading.Event()
    recorded = playrec.play_and_record(tone, fs, in_dev, out_dev, stop_evt, log=print)
    if recorded is None:
        return _result("playrec", False, "Playback/record failed. Check OS permissions and routing.")

    wav_io.write_wav("received.wav", recorded, fs)
    metrics = compare.compare_signals(tone, recorded, fs, freq)
    detail = (
        f"latency={metrics['latency_ms']:.2f} ms, "
        f"gain_err={metrics['gain_error_db']:.2f} dB, "
        f"noise_floor={metrics['noise_floor_dbfs']:.1f} dBFS, "
        f"THD_delta={metrics['thd_delta_db']:.1f} dB"
    )
    return _result("Hardware loopback compare", True, detail)


def parse_args(argv: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audio analysis self-test")
    parser.add_argument("--mode", choices=["offline", "hardware"], required=True)
    parser.add_argument("--in-dev", type=int, default=None, help="Input device id (hardware mode)")
    parser.add_argument("--out-dev", type=int, default=None, help="Output device id (hardware mode)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if np is None:
        print(
            "numpy is required for self_test. On Windows, create and activate a venv, then run "
            "`python -m pip install -r requirements.txt` before rerunning."
        )
        return 2
    if args.mode == "offline":
        success = run_offline_suite()
    else:
        success = run_hardware_smoke(args.in_dev, args.out_dev)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
