"""
exp24: Where, exactly, is the narrowband/broadband boundary -- and does a
spectral bandwidth metric measured on real ECG predict which side of it
real data falls on?

exp18 (real ECG), exp20 (controlled sinusoid vs. broadband pulse), and
exp23 (baseline comparison) all independently pointed at the same boundary:
serial_force beats a standard ESN when the fast component is narrowband,
loses when it's broadband. But exp20 only tested two discrete waveform
shapes. This experiment turns that into a continuous sweep, replacing the
binary "sinusoid or pulse" choice with a blend parameter alpha in [0, 1],
and measures each blend's actual spectral flatness (a standard, objective
bandwidth metric -- 0 for a pure tone, 1 for white noise) rather than just
using alpha itself as the x-axis. That makes the result a real, measurable
relationship (NRMSE vs. bandwidth) instead of two anecdotal points.

To close the loop: the same spectral flatness metric is computed on real
ECG's actual fast-band (8-30 Hz QRS) content, and plotted against the
synthetic curve. If it lands past the point where the crossover happens,
that's a real, quantitative explanation for why exp18 and exp23 both found
the standard ESN winning on real ECG's fast target -- not just "real data
behaved differently", but "real data's measured bandwidth predicts it should
behave differently, by the same yardstick the synthetic sweep uses."
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.datasets import electrocardiogram
from scipy.signal import butter, sosfiltfilt

from experiments.exp20_waveform_shape import sharp_pulse_wave
from reservoir_lab.physical import (
    AcousticReservoir,
    OptoelectronicReservoir,
    SerialPhysicalReservoir,
)
from reservoir_lab.readout import RidgeReadout
from reservoir_lab.reservoir import ESN

RESULTS_DIR = "experiments/data_results"
VISUALS_DIR = "experiments/visuals"
WASHOUT = 120
TRAIN_FRAC = 0.7
N_STEPS = 1400
SEEDS = [1, 2, 3, 4, 5, 6]
THETA, C, N_RES, REG = 0.2, 0.3, 100, 1e-3
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def spectral_flatness(signal):
    """Wiener entropy: geometric mean / arithmetic mean of the power
    spectrum. 0 = pure tone (fully narrowband), 1 = white noise (fully
    broadband). Standard, objective bandwidth metric."""
    spectrum = np.abs(np.fft.rfft(signal - signal.mean())) ** 2
    spectrum = spectrum[spectrum > 1e-15]
    geo_mean = np.exp(np.mean(np.log(spectrum)))
    arith_mean = np.mean(spectrum)
    return float(geo_mean / arith_mean)


def blended_wave(phase, alpha):
    """alpha=0 -> pure sinusoid, alpha=1 -> sharp broadband pulse train
    (exp20's sharp_pulse_wave), linear blend of the two waveform SHAPES
    (not amplitudes -- both are pre-normalized to unit std) in between."""
    sinusoid = np.sin(phase)
    sinusoid = sinusoid / sinusoid.std()
    pulse = sharp_pulse_wave(phase)
    blended = (1 - alpha) * sinusoid + alpha * pulse
    blended = blended - blended.mean()
    blended = blended / blended.std()
    return blended


def dual_timescale_task_blend(n_steps, alpha, seed=0, noise_std=0.35):
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)

    slow = 0.6 * np.sin(2 * np.pi * t / 180.0)
    gate = 0.5 * (1 + np.tanh(4 * slow))
    fast_phase = 2 * np.pi * t / 12.0 + 0.6 * np.sin(2 * np.pi * t / 33.0)

    fast_unit = blended_wave(fast_phase, alpha)
    fast_raw = 0.3 * fast_unit
    noise = rng.normal(0, noise_std, n_steps)

    fast_target = fast_raw * gate
    u = (slow + fast_raw + noise).reshape(-1, 1)
    y = np.stack([slow, fast_target], axis=1)

    split = int(n_steps * TRAIN_FRAC)
    return u[:split], y[:split], u[split:], y[split:], fast_raw


def nrmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2, axis=0)) / np.std(true, axis=0)


def build_serial_force(u_train, u_test, seed):
    stage1 = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=N_RES, theta=THETA, seed=seed)
    stage2 = AcousticReservoir(n_inputs=N_RES, n_oscillators=N_RES, c=C, seed=seed + 1)
    model = SerialPhysicalReservoir(stage1=stage1, stage2=stage2, combine="both", seed=seed)
    train_feats = model.run(u_train)
    test_feats = model.run(u_test, initial_state=model.last_state)
    return train_feats, test_feats


def build_esn_matched(u_train, u_test, seed):
    esn = ESN(n_inputs=1, n_reservoir=200, spectral_radius=1.2, sparsity=0.1, seed=seed)
    train_feats = esn.run(u_train)
    test_feats = esn.run(u_test, initial_state=esn.last_state)
    return train_feats, test_feats


ARCHS = {"serial_force": build_serial_force, "esn_matched": build_esn_matched}


def evaluate(builder, u_train, y_train, u_test, y_test, seed):
    train_feats, test_feats = builder(u_train, u_test, seed)
    readout = RidgeReadout(reg=REG).fit(train_feats, y_train, washout=WASHOUT)
    pred = readout.predict(test_feats)
    return nrmse(pred, y_test)


def real_ecg_fast_band_flatness():
    """Measures spectral flatness on real ECG's fast content using a
    high-pass filter only (5 Hz cutoff), NOT the narrow 8-30 Hz bandpass
    used for exp18's target extraction. The narrow bandpass artificially
    concentrates the spectrum into that window regardless of the
    underlying QRS transient's real shape (measured flatness ~1.7e-6 with
    the narrow band -- a filter artifact, not a property of the signal --
    vs 0.0064 with high-pass only). High-pass-only is the fair comparison
    to the synthetic waveforms, which are not band-restricted either."""
    ecg = electrocardiogram()
    fs = 360.0
    seg = ecg[: int(120 * fs)]
    seg = (seg - seg.mean()) / seg.std()
    sos_high = butter(4, 5.0, btype="high", fs=fs, output="sos")
    fast_high = sosfiltfilt(sos_high, seg)
    return spectral_flatness(fast_high)


def main():
    print("=" * 78)
    print("exp24: continuous narrowband -> broadband mapping, closed against real ECG")
    print("=" * 78)

    results = []
    flatness_by_alpha = {}
    for alpha in ALPHAS:
        for seed in SEEDS:
            u_train, y_train, u_test, y_test, fast_raw = dual_timescale_task_blend(N_STEPS, alpha, seed=seed)
            if alpha not in flatness_by_alpha:
                flatness_by_alpha[alpha] = spectral_flatness(fast_raw)

            for arch_name, builder in ARCHS.items():
                result = evaluate(builder, u_train, y_train, u_test, y_test, seed)
                results.append({
                    "alpha": alpha, "flatness": flatness_by_alpha[alpha], "seed": seed,
                    "architecture": arch_name, "nrmse_slow": result[0], "nrmse_fast": result[1],
                })

    with open(f"{RESULTS_DIR}/exp24_bandwidth_mapping.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "flatness", "seed", "architecture", "nrmse_slow", "nrmse_fast"])
        writer.writeheader()
        writer.writerows(results)

    ecg_flatness = real_ecg_fast_band_flatness()
    print(f"Real ECG fast-band (8-30 Hz) spectral flatness: {ecg_flatness:.4f}\n")

    print(f"{'alpha':<8}{'flatness':<12}{'architecture':<16}{'slow NRMSE':<20}{'fast NRMSE'}")
    print("-" * 78)
    summary = {}
    for alpha in ALPHAS:
        for arch_name in ARCHS:
            slow_vals = [r["nrmse_slow"] for r in results if r["alpha"] == alpha and r["architecture"] == arch_name]
            fast_vals = [r["nrmse_fast"] for r in results if r["alpha"] == alpha and r["architecture"] == arch_name]
            summary[(alpha, arch_name)] = (np.mean(slow_vals), np.std(slow_vals), np.mean(fast_vals), np.std(fast_vals))
            print(f"{alpha:<8}{flatness_by_alpha[alpha]:<12.4f}{arch_name:<16}"
                  f"{np.mean(slow_vals):.4f}+-{np.std(slow_vals):.4f}      "
                  f"{np.mean(fast_vals):.4f}+-{np.std(fast_vals):.4f}")
        print()

    with open(f"{RESULTS_DIR}/exp24_ecg_flatness.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["real_ecg_fast_band_flatness"])
        writer.writeheader()
        writer.writerow({"real_ecg_fast_band_flatness": ecg_flatness})

    # Plot: fast-target NRMSE vs measured spectral flatness, both architectures,
    # with the real ECG flatness marked as a vertical reference line.
    fig, ax = plt.subplots(figsize=(8, 5))
    flatness_vals = [flatness_by_alpha[a] for a in ALPHAS]
    colors = {"serial_force": "#eb6834", "esn_matched": "#9b59b6"}
    for arch_name in ARCHS:
        means = [summary[(a, arch_name)][2] for a in ALPHAS]
        stds = [summary[(a, arch_name)][3] for a in ALPHAS]
        ax.errorbar(flatness_vals, means, yerr=stds, marker="o", label=arch_name, color=colors[arch_name], capsize=3)
    ax.axvline(ecg_flatness, color="black", linestyle="--", alpha=0.7, label=f"real ECG fast-band ({ecg_flatness:.3f})")
    ax.set_xlabel("spectral flatness of fast component (0=pure tone, 1=white noise)")
    ax.set_ylabel("fast-target NRMSE (lower = better)")
    ax.set_title("exp24: does measured bandwidth predict the serial_force vs ESN crossover?")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{VISUALS_DIR}/exp24_bandwidth_mapping.png", dpi=130)
    plt.close(fig)
    print(f"Saved plot to {VISUALS_DIR}/exp24_bandwidth_mapping.png")

    # Explicit crossover check
    print("\n" + "=" * 78)
    print("CONCLUSION")
    print("=" * 78)
    crossover_alpha = None
    for alpha in ALPHAS:
        sf_fast = summary[(alpha, "serial_force")][2]
        esn_fast = summary[(alpha, "esn_matched")][2]
        winner = "serial_force" if sf_fast < esn_fast else "esn_matched"
        print(f"  alpha={alpha} (flatness={flatness_by_alpha[alpha]:.4f}): {winner} wins "
              f"(serial_force={sf_fast:.4f}, esn_matched={esn_fast:.4f})")
        if winner == "esn_matched" and crossover_alpha is None:
            crossover_alpha = alpha
    if crossover_alpha is not None:
        print(f"\nCrossover to ESN winning happens at alpha={crossover_alpha}, flatness~{flatness_by_alpha[crossover_alpha]:.4f}")
        print(f"Real ECG fast-band flatness: {ecg_flatness:.4f} "
              f"({'past' if ecg_flatness >= flatness_by_alpha[crossover_alpha] else 'before'} the crossover)")
    else:
        print("\nserial_force won at every alpha tested -- no crossover found in this range.")
        print(f"Real ECG fast-band flatness ({ecg_flatness:.4f}) may exceed the range tested here.")


if __name__ == "__main__":
    main()