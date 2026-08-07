"""
Real-signal validation of the serial_force finding, using a genuine
physiologically-recorded human ECG signal (scipy's bundled 5-minute
ECG recording, downloaded from raw.githubusercontent.com/scipy/dataset-ecg,
not synthesized) instead of a hand-designed sine+gate task.

This file replaces the earlier exp18_real_world_eeg.py, which despite its
name and framing was built on a fully synthetic, hand-constructed signal,
not real data -- see FINDINGS.md for that history. This version is built
on non-synthetic data with independently verified coupling structure.

Every earlier synthetic task (exp10-17, and the original exp18-20) was
constructed by hand -- a slow signal, a fast signal, and an explicit gate
function written by a human/LLM. That's the single biggest acknowledged
gap in FINDINGS.md. This experiment tests whether the finding survives
contact with a real signal that has genuine, independently-verified
(not fabricated) cross-timescale coupling.

VERIFYING THE COUPLING IS REAL, NOT ASSUMED:
Real ECG is known to exhibit respiratory sinus arrhythmia -- QRS-complex
amplitude and timing genuinely vary with the respiratory cycle. Before
building any task on this, verify_pac() checks this directly in this
specific recording: binning the fast-band (8-30 Hz, QRS-relevant) signal
envelope by the slow-band (<2 Hz) component's instantaneous PHASE shows
real modulation, while naive value-correlation is near zero -- exactly
the phase-locked, not value-locked, structure real PAC research looks
for. This is genuine structure in real data, not something imposed to
match the architecture's known strength. The measured values are printed
at runtime rather than hard-coded here.

TASK CONSTRUCTION:
  - slow_target = zero-phase low-pass (<2 Hz) of the clean ECG
  - fast_target = zero-phase band-pass (8-30 Hz) of the clean ECG
  - input = the clean ECG + Gaussian noise (the reservoirs never see
    the clean signal, matching every previous experiment's framing)
  - decimated 5x (360 Hz -> 72 Hz) with proper anti-aliasing, both for
    tractable runtime and because 72 Hz is comfortably above the 30 Hz
    fast band's Nyquist requirement

The task construction and PAC verification live in experiments/ecg_utils.py
(shared with exp27); this file calls them with the original first-120-second
window to preserve exp18's exact behavior.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.ecg_utils import build_ecg_task
from experiments.exp16_hyperparameter_sweep import (
    build_all_optical,
    build_parallel,
    build_serial_force,
)
from reservoir_lab.readout import RidgeReadout

RESULTS_DIR = "experiments/data_results"
VISUALS_DIR = "experiments/visuals"
WASHOUT = 100
SEEDS = [1, 2, 3, 4, 5, 6]
THETA, C, N_RES, REG = 0.2, 0.3, 100, 1e-3


def nrmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2, axis=0)) / np.std(true, axis=0)


def evaluate(builder, u_train, y_train, u_test, y_test, extra_kwargs=None):
    kwargs = extra_kwargs or {}
    train_feats, test_feats = builder(u_train, u_test, 1, THETA, C, N_RES, REG, **kwargs)
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
    print("REAL-SIGNAL VALIDATION: genuine ECG phase-amplitude coupling")
    print("=" * 74)

    results = []
    pac_stats = None
    example_task = None

    for seed in SEEDS:
        u_train, y_train, u_test, y_test, value_corr, mod_depth = build_ecg_task(
            seed, start_seconds=0.0, segment_seconds=120
        )
        if pac_stats is None:
            pac_stats = (value_corr, mod_depth)
            example_task = (u_train, y_train, u_test, y_test)

        for arch_name, (builder, kwargs) in ARCHS.items():
            result = evaluate(builder, u_train, y_train, u_test, y_test, kwargs)
            results.append({
                "seed": seed, "architecture": arch_name,
                "nrmse_slow": result[0], "nrmse_fast": result[1],
            })

    print(f"Verified PAC in this recording: value-correlation={pac_stats[0]:.4f}, "
          f"phase-binned modulation depth={pac_stats[1]:.4f}\n")

    print(f"{'architecture':<16}{'NRMSE slow (mean+-std)':<28}{'NRMSE fast (mean+-std)'}")
    print("-" * 70)
    for arch_name in ARCHS:
        slow_vals = [r["nrmse_slow"] for r in results if r["architecture"] == arch_name]
        fast_vals = [r["nrmse_fast"] for r in results if r["architecture"] == arch_name]
        print(f"{arch_name:<16}{np.mean(slow_vals):.4f} +- {np.std(slow_vals):.4f}          "
              f"{np.mean(fast_vals):.4f} +- {np.std(fast_vals):.4f}")

    with open(f"{RESULTS_DIR}/exp18_real_signal_ecg.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "architecture", "nrmse_slow", "nrmse_fast"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved raw results to {RESULTS_DIR}/exp18_real_signal_ecg.csv")

    u_train, y_train, _, _ = example_task
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    window = slice(0, 500)
    axes[0].plot(u_train[window], color="#2a78d6", linewidth=0.8)
    axes[0].set_title("Noisy input (real ECG + noise)")
    axes[1].plot(y_train[window, 0], color="#1baf7a")
    axes[1].set_title("Slow target (<2 Hz component of clean ECG)")
    axes[2].plot(y_train[window, 1], color="#eb6834", linewidth=0.8)
    axes[2].set_title("Fast target (8-30 Hz component of clean ECG)")
    fig.tight_layout()
    fig.savefig(f"{VISUALS_DIR}/exp18_real_signal_ecg.png", dpi=130)
    plt.close(fig)
    print(f"Saved task visualization to {VISUALS_DIR}/exp18_real_signal_ecg.png")


if __name__ == "__main__":
    main()