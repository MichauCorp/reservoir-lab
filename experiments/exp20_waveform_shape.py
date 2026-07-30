"""
Does serial_force's advantage depend on the fast component being a clean
sinusoid, rather than a broadband/transient waveform like real signals
actually have?

The discrepancy motivating this experiment: on every synthetic task
(exp13, exp14, exp16), serial_force's win was specifically on the FAST
target. But on both real-signal tests -- exp18 (real ECG) and exp19
(fixed speech formant tracking) -- the pattern reversed or vanished:
serial_force won the SLOW target on real ECG but lost the fast target,
and parallel won outright on speech. The one thing consistently
different between the synthetic and real-signal tasks: the synthetic
"fast" component was always a clean sinusoid, while real ECG's QRS
complexes and speech's glottal pulses are broadband, transient-rich
waveforms, not narrowband oscillations.

This experiment isolates that ONE variable. Two versions of the same
task, with IDENTICAL slow component, gate, phase-modulation timing,
noise, and amplitude scale -- the only thing that changes is whether the
fast component is a smooth sinusoid or a sharp periodic pulse (broadband,
same fundamental period and phase-modulation structure, just concentrated
energy in a brief transient each cycle instead of spread across a single
frequency):

  sinusoid:  fast(t)  = sin(phase(t))
  broadband: fast(t)  = sharp_pulse(phase(t))   -- same phase(t) function

If the fast-target win depends on waveform shape, it should be present
for the sinusoid version and reduced or absent for the broadband version
-- reproducing, in a controlled synthetic setting, the pattern already
observed on real data.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.exp16_hyperparameter_sweep import (
    build_all_optical,
    build_parallel,
    build_serial_force,
)
from reservoir_lab.readout import RidgeReadout

RESULTS_DIR = "experiments/data_results"
VISUALS_DIR = "experiments/visuals"
WASHOUT = 120
TRAIN_FRAC = 0.7
N_STEPS = 1400
SEEDS = [1, 2, 3, 4, 5, 6]
THETA, C, N_RES, REG = 0.2, 0.3, 100, 1e-3


def sharp_pulse_wave(phase, sharpness=8.0):
    """Broadband periodic waveform as a function of phase (radians): a
    sharp, asymmetric transient each cycle instead of a smooth sinusoid.
    Normalized to zero mean, unit std before amplitude scaling by the
    caller, so it's directly comparable to a unit-std sinusoid."""
    wrapped = np.mod(phase, 2 * np.pi) / (2 * np.pi)  # in [0, 1)
    pulse = np.exp(-sharpness * wrapped)
    pulse = pulse - pulse.mean()
    pulse = pulse / pulse.std()
    return pulse


def dual_timescale_task(n_steps, waveform, seed=0, noise_std=0.35):
    """waveform: 'sinusoid' or 'broadband'. Everything except the fast
    component's waveform shape is identical between the two variants."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)

    slow = 0.6 * np.sin(2 * np.pi * t / 180.0)
    gate = 0.5 * (1 + np.tanh(4 * slow))

    # identical phase/timing structure for both waveform variants
    fast_phase = 2 * np.pi * t / 12.0 + 0.6 * np.sin(2 * np.pi * t / 33.0)

    if waveform == "sinusoid":
        fast_unit = np.sin(fast_phase) / np.sin(fast_phase).std()
    elif waveform == "broadband":
        fast_unit = sharp_pulse_wave(fast_phase)
    else:
        raise ValueError(f"unknown waveform: {waveform}")

    fast_raw = 0.3 * fast_unit  # same amplitude scale (std) for both variants
    noise = rng.normal(0, noise_std, n_steps)

    fast_target = fast_raw * gate
    u = (slow + fast_raw + noise).reshape(-1, 1)
    y = np.stack([slow, fast_target], axis=1)

    split = int(n_steps * TRAIN_FRAC)
    return u[:split], y[:split], u[split:], y[split:]


def nrmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2, axis=0)) / np.std(true, axis=0)


def evaluate(builder, u_train, y_train, u_test, y_test, seed, extra_kwargs=None):
    kwargs = extra_kwargs or {}
    train_feats, test_feats = builder(u_train, u_test, seed, THETA, C, N_RES, REG, **kwargs)
    readout = RidgeReadout(reg=REG).fit(train_feats, y_train, washout=WASHOUT)
    pred = readout.predict(test_feats)
    return nrmse(pred, y_test)


ARCHS = {
    "serial_force": (build_serial_force, {}),
    "parallel": (build_parallel, {}),
    "all_optical": (build_all_optical, {"theta2": 3.0}),
}


def main():
    print("=" * 74)
    print("Does the fast/slow win pattern depend on narrowband vs broadband")
    print("fast-component waveform shape?")
    print("=" * 74)

    results = []
    example_signals = {}

    for waveform in ["sinusoid", "broadband"]:
        for seed in SEEDS:
            u_train, y_train, u_test, y_test = dual_timescale_task(N_STEPS, waveform, seed=seed)
            if seed == SEEDS[0]:
                example_signals[waveform] = (u_train, y_train)

            for arch_name, (builder, kwargs) in ARCHS.items():
                result = evaluate(builder, u_train, y_train, u_test, y_test, seed, kwargs)
                results.append({
                    "waveform": waveform, "seed": seed, "architecture": arch_name,
                    "nrmse_slow": result[0], "nrmse_fast": result[1],
                })

    with open(f"{RESULTS_DIR}/exp20_waveform_shape.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["waveform", "seed", "architecture", "nrmse_slow", "nrmse_fast"])
        writer.writeheader()
        writer.writerows(results)

    print(f"{'waveform':<12}{'architecture':<16}{'NRMSE slow':<24}{'NRMSE fast'}")
    print("-" * 74)
    summary = {}
    for waveform in ["sinusoid", "broadband"]:
        for arch_name in ARCHS:
            slow_vals = [r["nrmse_slow"] for r in results if r["waveform"] == waveform and r["architecture"] == arch_name]
            fast_vals = [r["nrmse_fast"] for r in results if r["waveform"] == waveform and r["architecture"] == arch_name]
            summary[(waveform, arch_name)] = (np.mean(slow_vals), np.std(slow_vals), np.mean(fast_vals), np.std(fast_vals))
            print(f"{waveform:<12}{arch_name:<16}{np.mean(slow_vals):.4f} +- {np.std(slow_vals):.4f}        "
                  f"{np.mean(fast_vals):.4f} +- {np.std(fast_vals):.4f}")
        print()

    print(f"\nSaved raw results to {RESULTS_DIR}/exp20_waveform_shape.csv")

    # Plot 1: example waveforms, sinusoid vs broadband
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    window = slice(0, 100)
    u_sin, _ = example_signals["sinusoid"]
    u_bb, _ = example_signals["broadband"]
    axes[0].plot(u_sin[window], color="#2a78d6")
    axes[0].set_title("Sinusoid variant (input)")
    axes[1].plot(u_bb[window], color="#eb6834")
    axes[1].set_title("Broadband variant (input) -- same phase/timing, different waveform shape")
    fig.tight_layout()
    fig.savefig(f"{VISUALS_DIR}/exp20_waveform_examples.png", dpi=130)
    plt.close(fig)

    # Plot 2: fast-target NRMSE, sinusoid vs broadband, grouped by architecture
    fig, ax = plt.subplots(figsize=(8, 4.5))
    arch_names = list(ARCHS.keys())
    x = np.arange(len(arch_names))
    width = 0.35
    sin_means = [summary[("sinusoid", a)][2] for a in arch_names]
    sin_stds = [summary[("sinusoid", a)][3] for a in arch_names]
    bb_means = [summary[("broadband", a)][2] for a in arch_names]
    bb_stds = [summary[("broadband", a)][3] for a in arch_names]
    ax.bar(x - width / 2, sin_means, width, yerr=sin_stds, label="sinusoid", color="#2a78d6", capsize=3)
    ax.bar(x + width / 2, bb_means, width, yerr=bb_stds, label="broadband", color="#eb6834", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(arch_names)
    ax.set_ylabel("fast-target NRMSE (lower = better)")
    ax.set_title("Does serial_force's fast-target advantage survive a broadband fast component?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{VISUALS_DIR}/exp20_waveform_shape.png", dpi=130)
    plt.close(fig)
    print(f"Saved plots to {VISUALS_DIR}/exp20_waveform_examples.png and exp20_waveform_shape.png")


if __name__ == "__main__":
    main()