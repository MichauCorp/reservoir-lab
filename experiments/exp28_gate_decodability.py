"""
exp28: Gate decodability from acoustic states on real ECG segments.

exp14 established the mechanism on synthetic narrowband content: a linear
decoder trained on acoustic states ALONE (no optical, no full-task readout)
recovers the true gate signal at NRMSE 0.298 when the acoustic is
force-driven by optical's output, vs. 1.232 when driven directly by raw
input. That's the mechanism -- the optical stage injects gate-relevant
information into the acoustic state.

exp27_ecg_segment_scan just refuted exp26's depth prediction on real data:
even at the highest natural modulation depth (0.9005, segment 180-240s),
serial_force lost the fast target to a size-matched ESN (0.8333 vs 0.7315).
And exp27_ecg_regate_depth showed serial_force's fast NRMSE stays FLAT
across amplified depth on real broadband content -- it can't benefit from
the gate information at all.

This experiment tests whether the MECHANISM itself breaks down on real
broadband content. It reuses exp14's exact methodology (linear decoder on
acoustic states alone to recover the gate signal) but applies it to real
ECG segments:

  - first-120s (natural depth ~0.35) -- exp18's original segment
  - highest-depth 180-240s (natural depth ~0.90) -- exp27's chosen segment
  - re-gated versions of first-120s at amplified depths (0.6, 1.0, 1.5,
    2.0) -- reusing exp27_ecg_regate_depth's re-gating logic, to test
    whether decodability improves with depth on real broadband content

The "gate" signal for real ECG is the smoothed fast-band envelope -- the
same envelope_smooth from verify_pac() (uniform_filter1d, 300ms window).
This is the signal whose phase-locked modulation the slow component drives,
and it's what the acoustic state needs to encode.

Discriminating predictions:
  - If force-coupled decodability is much worse on real ECG than 0.298
    (even at high depth), the mechanism breaks down on broadband -- this
    explains why exp27 refuted exp26's prediction.
  - If decodability improves with depth on real broadband, the mechanism
    survives but something else limits serial_force's performance.
  - If decodability is fine but serial_force still loses, the bottleneck
    is elsewhere (readout, not state encoding).
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
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, decimate, hilbert, sosfiltfilt

from experiments.ecg_utils import FS_NATIVE
from reservoir_lab.physical import (
    AcousticReservoir,
    OptoelectronicReservoir,
)
from reservoir_lab.readout import RidgeReadout

RESULTS_DIR = "experiments/data_results"
VISUALS_DIR = "experiments/visuals"
WASHOUT = 120
TRAIN_FRAC = 0.7
DECIMATE_FACTOR = 5
SEEDS = [1, 2, 3, 4, 5, 6]
THETA, C, N_RES, REG = 0.2, 0.3, 100, 1e-3

# exp14's synthetic narrowband baseline for comparison.
EXP14_COUPLED_NRMSE = 0.298
EXP14_DIRECT_NRMSE = 1.232

# Segments to test: (label, start_seconds, segment_seconds, re_gate_depth or None)
SEGMENTS = [
    ("first_120s", 0.0, 120, None),
    ("high_depth_180_240s", 180.0, 60, None),
    ("regate_0.6", 0.0, 120, 0.6),
    ("regate_1.0", 0.0, 120, 1.0),
    ("regate_1.5", 0.0, 120, 1.5),
    ("regate_2.0", 0.0, 120, 2.0),
]


def extract_segment(start_seconds, segment_seconds, re_gate_depth=None):
    """Extract a segment of the ECG recording, standardize it, filter into
    slow/fast components, and optionally re-gate the fast band at an
    amplified depth (exp27_ecg_regate_depth's logic).

    Returns (slow_native, fast_native, seg, gate_native) at native 360 Hz.
    gate_native is the smoothed fast-band envelope (the gate signal).
    """
    ecg = electrocardiogram()
    start_native = int(start_seconds * FS_NATIVE)
    n_native = int(segment_seconds * FS_NATIVE)
    seg = ecg[start_native:start_native + n_native]
    seg = (seg - seg.mean()) / seg.std()

    sos_low = butter(4, 2.0, btype="low", fs=FS_NATIVE, output="sos")
    sos_fast = butter(4, [8.0, 30.0], btype="band", fs=FS_NATIVE, output="sos")
    slow_native = sosfiltfilt(sos_low, seg)
    fast_native = sosfiltfilt(sos_fast, seg)

    if re_gate_depth is not None:
        # exp27_ecg_regate_depth's re-gating: gate = 0.5 + depth*(gate_full-0.5)
        gate_full = 0.5 * (1 + np.tanh(4 * slow_native / slow_native.std()))
        gate = 0.5 + re_gate_depth * (gate_full - 0.5)
        fast_native = fast_native * gate

    # The gate signal = smoothed fast-band envelope (same as verify_pac's
    # envelope_smooth, which is what the slow component phase-modulates).
    envelope = np.abs(hilbert(fast_native))
    gate_native = uniform_filter1d(envelope, size=int(FS_NATIVE * 0.3))

    return slow_native, fast_native, seg, gate_native


def build_task(seed, start_seconds, segment_seconds, re_gate_depth=None, noise_std=0.3):
    """Build the task (noisy input, slow/fast targets, gate signal) for a
    segment, decimated to 72 Hz, with 70/30 train/test split."""
    slow_native, fast_native, seg, gate_native = extract_segment(
        start_seconds, segment_seconds, re_gate_depth
    )

    slow = decimate(slow_native, DECIMATE_FACTOR, zero_phase=True)
    fast = decimate(fast_native, DECIMATE_FACTOR, zero_phase=True)
    clean = decimate(seg, DECIMATE_FACTOR, zero_phase=True)
    gate = decimate(gate_native, DECIMATE_FACTOR, zero_phase=True)

    rng = np.random.default_rng(seed)
    noisy_input = clean + rng.normal(0, noise_std, len(clean))

    u = noisy_input.reshape(-1, 1)
    y = np.stack([slow, fast], axis=1)
    split = int(len(u) * TRAIN_FRAC)
    return u[:split], y[:split], u[split:], y[split:], gate[:split], gate[split:]


def nrmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2, axis=0)) / np.std(true, axis=0)


def acoustic_direct_states(u_train, u_test, seed):
    ac = AcousticReservoir(n_inputs=1, n_oscillators=N_RES, c=C, seed=seed + 1)
    tr = ac.run(u_train)
    te = ac.run(u_test, initial_state=ac.last_state)
    return tr, te


def acoustic_force_coupled_states(u_train, u_test, seed):
    optical = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=N_RES, theta=THETA, seed=seed)
    acoustic = AcousticReservoir(n_inputs=N_RES, n_oscillators=N_RES, c=C, seed=seed + 1)
    opt_tr = optical.run(u_train)
    ac_tr = acoustic.run(opt_tr)
    opt_te = optical.run(u_test, initial_state=optical.last_state)
    ac_te = acoustic.run(opt_te, initial_state=acoustic.last_state)
    return ac_tr, ac_te


def decode_gate(states_train, gate_train, states_test, gate_test):
    readout = RidgeReadout(reg=REG).fit(states_train, gate_train.reshape(-1, 1), washout=WASHOUT)
    pred = readout.predict(states_test)
    return nrmse(pred, gate_test.reshape(-1, 1))[0]


def main():
    print("=" * 78)
    print("exp28: gate decodability from acoustic states on real ECG")
    print("=" * 78)
    print(f"exp14 synthetic narrowband baseline: coupled={EXP14_COUPLED_NRMSE}, "
          f"direct={EXP14_DIRECT_NRMSE}\n")

    results = []
    for label, start_seconds, segment_seconds, re_gate_depth in SEGMENTS:
        for seed in SEEDS:
            u_train, _y_train, u_test, _y_test, gate_train, gate_test = build_task(
                seed, start_seconds, segment_seconds, re_gate_depth
            )

            direct_tr, direct_te = acoustic_direct_states(u_train, u_test, seed)
            coupled_tr, coupled_te = acoustic_force_coupled_states(u_train, u_test, seed)

            nrmse_direct = decode_gate(direct_tr, gate_train, direct_te, gate_test)
            nrmse_coupled = decode_gate(coupled_tr, gate_train, coupled_te, gate_test)

            results.append({
                "segment": label, "start_seconds": start_seconds,
                "segment_seconds": segment_seconds, "re_gate_depth": re_gate_depth,
                "seed": seed, "nrmse_direct": nrmse_direct, "nrmse_coupled": nrmse_coupled,
            })

    with open(f"{RESULTS_DIR}/exp28_gate_decodability.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["segment", "start_seconds", "segment_seconds",
                                               "re_gate_depth", "seed", "nrmse_direct", "nrmse_coupled"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved raw results to {RESULTS_DIR}/exp28_gate_decodability.csv\n")

    # ---- report ---------------------------------------------------------
    print(f"{'segment':<22}{'direct NRMSE':<20}{'coupled NRMSE':<20}{'coupled-direct'}")
    print("-" * 78)
    summary = {}
    for label, start_seconds, segment_seconds, re_gate_depth in SEGMENTS:
        direct_vals = [r["nrmse_direct"] for r in results if r["segment"] == label]
        coupled_vals = [r["nrmse_coupled"] for r in results if r["segment"] == label]
        d_mean, d_std = np.mean(direct_vals), np.std(direct_vals)
        c_mean, c_std = np.mean(coupled_vals), np.std(coupled_vals)
        summary[label] = (d_mean, d_std, c_mean, c_std)
        print(f"{label:<22}{d_mean:.4f} +- {d_std:.4f}      "
              f"{c_mean:.4f} +- {c_std:.4f}      {c_mean - d_mean:+.4f}")

    # ---- plot -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = [s[0] for s in SEGMENTS]
    x = np.arange(len(labels))
    width = 0.35

    direct_means = [summary[l][0] for l in labels]
    direct_stds = [summary[l][1] for l in labels]
    coupled_means = [summary[l][2] for l in labels]
    coupled_stds = [summary[l][3] for l in labels]

    ax.bar(x - width / 2, direct_means, width, yerr=direct_stds, capsize=3,
           label="acoustic_direct", color="#9b9b9b")
    ax.bar(x + width / 2, coupled_means, width, yerr=coupled_stds, capsize=3,
           label="acoustic_force_coupled", color="#eb6834")

    # exp14 synthetic narrowband reference lines
    ax.axhline(EXP14_COUPLED_NRMSE, color="#eb6834", linestyle="--", alpha=0.7,
               label=f"exp14 coupled baseline ({EXP14_COUPLED_NRMSE})")
    ax.axhline(EXP14_DIRECT_NRMSE, color="#9b9b9b", linestyle="--", alpha=0.7,
               label=f"exp14 direct baseline ({EXP14_DIRECT_NRMSE})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("gate decodability NRMSE (lower = better)")
    ax.set_title("exp28: gate decodability from acoustic states on real ECG segments")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{VISUALS_DIR}/exp28_gate_decodability.png", dpi=130)
    plt.close(fig)
    print(f"Saved plot to {VISUALS_DIR}/exp28_gate_decodability.png")

    # ---- conclusion -----------------------------------------------------
    print("\n" + "=" * 78)
    print("CONCLUSION")
    print("=" * 78)

    first_coupled = summary["first_120s"][2]
    high_coupled = summary["high_depth_180_240s"][2]
    print(f"  first_120s coupled decodability: {first_coupled:.4f} "
          f"(exp14 baseline: {EXP14_COUPLED_NRMSE})")
    print(f"  high_depth_180_240s coupled decodability: {high_coupled:.4f}")

    # Check if re-gating improves decodability
    regate_coupled = [summary[f"regate_{d}"][2] for d in [0.6, 1.0, 1.5, 2.0]]
    regate_improved = regate_coupled[-1] < regate_coupled[0]

    if first_coupled > EXP14_COUPLED_NRMSE * 1.5:
        print("  -> Force-coupled decodability is MUCH WORSE on real ECG than exp14's")
        print("     synthetic narrowband baseline. The mechanism BREAKS DOWN on broadband")
        print("     content -- the optical stage cannot inject gate-relevant information")
        print("     into the acoustic state. This explains why exp27 refuted exp26's")
        print("     depth prediction: depth doesn't help because the gate information")
        print("     never reaches the acoustic state in the first place.")
    elif first_coupled < EXP14_COUPLED_NRMSE * 1.2:
        print("  -> Force-coupled decodability is COMPARABLE to exp14's synthetic baseline.")
        print("     The mechanism SURVIVES on real ECG. The bottleneck is elsewhere")
        print("     (readout, not state encoding).")
    else:
        print("  -> Force-coupled decodability is MODERATELY worse than exp14's baseline.")
        print("     The mechanism partially degrades on broadband content.")

    if regate_improved:
        print(f"  Re-gating improved decodability ({regate_coupled[0]:.4f} -> "
              f"{regate_coupled[-1]:.4f}): depth helps the mechanism on real broadband.")
    else:
        print(f"  Re-gating did NOT improve decodability ({regate_coupled[0]:.4f} -> "
              f"{regate_coupled[-1]:.4f}): depth does not help the mechanism on real broadband.")


if __name__ == "__main__":
    main()