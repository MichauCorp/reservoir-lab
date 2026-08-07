"""
exp27: Segment-scan of the real ECG recording for high modulation depth.

exp18 built a task from the first 120 seconds of scipy's bundled real ECG
recording (electrocardiogram(), 5 minutes total, 360 Hz) and found a
standard size-matched ESN beats serial_force on the fast target there.

exp24, exp25, and exp26 later explained why: serial_force only beats a
standard ESN when the task's cross-timescale gate modulation depth is
strong/near-saturating (measured via the SAME phase-binning method as
exp18's verify_pac(), roughly >=0.9-1.0 on that scale). The first-120-second
ECG segment has weak modulation depth (~0.35), which is why the ESN won
there.

scipy's docs describe this ECG recording as containing "several areas with
different morphology" including pathological changes -- meaning different
segments of the SAME already-verified recording likely have different, and
possibly much stronger, modulation depth than the resting segment exp18
happened to use. This experiment tests exp26's prediction directly: does
serial_force win on a segment with genuinely higher modulation depth?

Design:
  1. Scan the full recording (all 108000 samples / 300 seconds) in sliding
     windows -- 60-second windows, 30-second step (50% overlap), giving 9
     windows across the recording.
  2. For each window, compute modulation depth using the EXACT same method
     as exp18's verify_pac() (imported from experiments.ecg_utils, not
     reimplemented). Standardize each window (zero mean, unit std) before
     filtering, same as exp18 does. Same filters: 4th-order Butterworth
     low-pass at 2 Hz for slow, 4th-order Butterworth bandpass 8-30 Hz for
     fast (both via scipy.signal.butter + sosfiltfilt).
  3. Rank all 9 windows by measured modulation depth; identify the single
     window with the highest depth.
  4. Build the same task construction exp18 uses (build_ecg_task's pattern:
     noisy input = standardized clean segment + Gaussian noise, std=0.3;
     slow target = low-pass component; fast target = bandpass component;
     decimate 5x from 360 Hz to 72 Hz with zero-phase decimation; 70/30
     train/test split) applied to the highest-depth window.
  5. Run three architectures on this new segment: serial_force, parallel,
     and a size-matched ESN (n_reservoir=200, matching the combined
     architectures' 200 total features via "both taps" -- NOT exp19's
     original bug of comparing a 100-unit ESN against 200-unit combined
     architectures). Use readout regularization reg=1e-3 (established as
     correct throughout this project; NOT 1e-6 which caused catastrophic
     instability for serial_force in exp16). Use washout=120. Use 6 seeds:
     [1,2,3,4,5,6].
  6. Report slow and fast NRMSE for all three architectures, and explicitly
     compare against exp18's original result (serial_force fast NRMSE
     0.8821, esn_matched fast NRMSE 0.7131 from exp23's size-matched run)
     to make the before/after contrast explicit.
  7. Print an explicit CONCLUSION section stating: the measured modulation
     depth of the chosen segment, whether serial_force won or lost the fast
     target, and whether this confirms or refutes exp26's prediction
     (serial_force should win when depth is high enough, regardless of
     which real segment it comes from).

Outputs:
  - experiments/data_results/exp27_ecg_segment_scan.csv  (per-seed results)
  - experiments/data_results/exp27_window_scan.csv       (depth-by-window)
  - experiments/visuals/exp27_ecg_segment_scan.png       (bar chart + NRMSE)
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

from experiments.ecg_utils import FS_NATIVE, build_ecg_task, verify_pac
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
DECIMATE_FACTOR = 5
SEEDS = [1, 2, 3, 4, 5, 6]
THETA, C, N_RES, REG = 0.2, 0.3, 100, 1e-3

# Window scan parameters
WINDOW_SECONDS = 60
STEP_SECONDS = 30
TOTAL_SECONDS = 300  # full recording length

# exp18's original results (from experiments/data_results/exp18_real_signal_ecg.csv)
# and exp23's size-matched ESN on the same first-120s segment.
EXP18_SERIAL_FORCE_FAST = 0.8821
EXP18_SERIAL_FORCE_SLOW = 0.3393
EXP18_PARALLEL_FAST = 0.9146
EXP18_PARALLEL_SLOW = 0.3512
EXP23_ESN_MATCHED_FAST = 0.7131
EXP23_ESN_MATCHED_SLOW = 0.3615


def scan_windows():
    """Scan the full recording in sliding windows, computing modulation
    depth for each using exp18's exact verify_pac() methodology.

    Returns a list of dicts: {start_seconds, end_seconds, value_corr,
    modulation_depth}.
    """
    ecg = electrocardiogram()
    windows = []
    start = 0
    while start + WINDOW_SECONDS <= TOTAL_SECONDS:
        end = start + WINDOW_SECONDS
        start_native = int(start * FS_NATIVE)
        n_native = int(WINDOW_SECONDS * FS_NATIVE)
        seg = ecg[start_native:start_native + n_native]
        seg = (seg - seg.mean()) / seg.std()

        sos_low = butter(4, 2.0, btype="low", fs=FS_NATIVE, output="sos")
        sos_fast = butter(4, [8.0, 30.0], btype="band", fs=FS_NATIVE, output="sos")
        low_native = sosfiltfilt(sos_low, seg)
        fast_native = sosfiltfilt(sos_fast, seg)

        value_corr, mod_depth = verify_pac(low_native, fast_native)
        windows.append({
            "start_seconds": start,
            "end_seconds": end,
            "value_corr": value_corr,
            "modulation_depth": mod_depth,
        })
        start += STEP_SECONDS
    return windows


def nrmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2, axis=0)) / np.std(true, axis=0)


def build_serial_force(u_train, u_test, seed):
    stage1 = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=N_RES, theta=THETA, seed=seed)
    stage2 = AcousticReservoir(n_inputs=N_RES, n_oscillators=N_RES, c=C, seed=seed + 1)
    model = SerialPhysicalReservoir(stage1=stage1, stage2=stage2, combine="both", seed=seed)
    train_feats = model.run(u_train)
    test_feats = model.run(u_test, initial_state=model.last_state)
    return train_feats, test_feats


def build_parallel(u_train, u_test, seed):
    optical = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=N_RES, theta=THETA, seed=seed)
    acoustic = AcousticReservoir(n_inputs=1, n_oscillators=N_RES, c=C, seed=seed + 1)
    opt_tr, ac_tr = optical.run(u_train), acoustic.run(u_train)
    opt_te = optical.run(u_test, initial_state=optical.last_state)
    ac_te = acoustic.run(u_test, initial_state=acoustic.last_state)
    return np.hstack([opt_tr, ac_tr]), np.hstack([opt_te, ac_te])


def build_esn_matched(u_train, u_test, seed):
    esn = ESN(n_inputs=1, n_reservoir=200, spectral_radius=1.2, sparsity=0.1, seed=seed)
    train_feats = esn.run(u_train)
    test_feats = esn.run(u_test, initial_state=esn.last_state)
    return train_feats, test_feats


ARCHS = {
    "serial_force": build_serial_force,
    "parallel": build_parallel,
    "esn_matched": build_esn_matched,
}


def evaluate(builder, u_train, y_train, u_test, y_test, seed):
    train_feats, test_feats = builder(u_train, u_test, seed)
    readout = RidgeReadout(reg=REG).fit(train_feats, y_train, washout=WASHOUT)
    pred = readout.predict(test_feats)
    return nrmse(pred, y_test)


def main():
    print("=" * 78)
    print("exp27: segment-scan of real ECG for high modulation depth")
    print("=" * 78)

    # ---- Step 1: scan all windows -------------------------------------
    windows = scan_windows()
    windows_sorted = sorted(windows, key=lambda w: w["modulation_depth"], reverse=True)

    print(f"\nWindow scan: {len(windows)} windows "
          f"({WINDOW_SECONDS}s window, {STEP_SECONDS}s step)\n")
    print(f"{'window':<8}{'start (s)':<12}{'end (s)':<10}{'value_corr':<14}{'mod_depth':<12}")
    print("-" * 60)
    for i, w in enumerate(windows):
        print(f"{i:<8}{w['start_seconds']:<12}{w['end_seconds']:<10}"
              f"{w['value_corr']:<14.4f}{w['modulation_depth']:<12.4f}")

    print("\nRanked by modulation depth (highest first):")
    for rank, w in enumerate(windows_sorted, 1):
        print(f"  #{rank}: start={w['start_seconds']}s, end={w['end_seconds']}s, "
              f"mod_depth={w['modulation_depth']:.4f}")

    best_window = windows_sorted[0]
    best_start = best_window["start_seconds"]
    best_depth = best_window["modulation_depth"]
    print(f"\nChosen segment: start={best_start}s, end={best_start + WINDOW_SECONDS}s, "
          f"modulation depth={best_depth:.4f}")

    # Save window scan results
    with open(f"{RESULTS_DIR}/exp27_window_scan.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["start_seconds", "end_seconds",
                                               "value_corr", "modulation_depth"])
        writer.writeheader()
        writer.writerows(windows)
    print(f"Saved window scan to {RESULTS_DIR}/exp27_window_scan.csv")

    # ---- Step 2: build task on highest-depth window --------------------
    print(f"\nBuilding task on highest-depth segment (start={best_start}s)...")

    results = []
    for seed in SEEDS:
        u_train, y_train, u_test, y_test, _value_corr, _mod_depth = build_ecg_task(
            seed, start_seconds=best_start, segment_seconds=WINDOW_SECONDS
        )
        for arch_name, builder in ARCHS.items():
            result = evaluate(builder, u_train, y_train, u_test, y_test, seed)
            results.append({
                "seed": seed, "architecture": arch_name,
                "nrmse_slow": result[0], "nrmse_fast": result[1],
            })

    with open(f"{RESULTS_DIR}/exp27_ecg_segment_scan.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "architecture", "nrmse_slow", "nrmse_fast"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved raw results to {RESULTS_DIR}/exp27_ecg_segment_scan.csv")

    # ---- Step 3: report -------------------------------------------------
    print(f"\n{'architecture':<16}{'NRMSE slow (mean+-std)':<28}{'NRMSE fast (mean+-std)'}")
    print("-" * 70)
    summary = {}
    for arch_name in ARCHS:
        slow_vals = [r["nrmse_slow"] for r in results if r["architecture"] == arch_name]
        fast_vals = [r["nrmse_fast"] for r in results if r["architecture"] == arch_name]
        summary[arch_name] = (np.mean(slow_vals), np.std(slow_vals),
                              np.mean(fast_vals), np.std(fast_vals))
        print(f"{arch_name:<16}{np.mean(slow_vals):.4f} +- {np.std(slow_vals):.4f}          "
              f"{np.mean(fast_vals):.4f} +- {np.std(fast_vals):.4f}")

    # ---- Step 4: comparison vs exp18 ------------------------------------
    print("\n" + "=" * 78)
    print("COMPARISON vs exp18's original first-120s segment")
    print("=" * 78)
    print("  exp18 (first 120s, depth~0.35):")
    print(f"    serial_force: slow={EXP18_SERIAL_FORCE_SLOW:.4f}, fast={EXP18_SERIAL_FORCE_FAST:.4f}")
    print(f"    parallel:     slow={EXP18_PARALLEL_SLOW:.4f}, fast={EXP18_PARALLEL_FAST:.4f}")
    print(f"    esn_matched:  slow={EXP23_ESN_MATCHED_SLOW:.4f}, fast={EXP23_ESN_MATCHED_FAST:.4f}")
    print(f"  exp27 (start={best_start}s, depth={best_depth:.4f}):")
    for arch_name in ARCHS:
        s_mean, s_std, f_mean, f_std = summary[arch_name]
        print(f"    {arch_name:<16} slow={s_mean:.4f} +- {s_std:.4f}, fast={f_mean:.4f} +- {f_std:.4f}")

    # ---- Step 5: conclusion ---------------------------------------------
    print("\n" + "=" * 78)
    print("CONCLUSION")
    print("=" * 78)
    sf_fast = summary["serial_force"][2]
    esn_fast = summary["esn_matched"][2]
    par_fast = summary["parallel"][2]

    print(f"  Chosen segment: start={best_start}s, end={best_start + WINDOW_SECONDS}s")
    print(f"  Measured modulation depth: {best_depth:.4f}")
    print(f"  Fast-target NRMSE: serial_force={sf_fast:.4f}, parallel={par_fast:.4f}, "
          f"esn_matched={esn_fast:.4f}")

    if sf_fast < esn_fast:
        print(f"  serial_force WON the fast target (beat esn_matched by {esn_fast - sf_fast:.4f}).")
        print("  -> CONFIRMS exp26's prediction: serial_force wins when modulation depth is")
        print("     high enough, regardless of which real segment it comes from.")
    else:
        print(f"  serial_force LOST the fast target (esn_matched beat it by {sf_fast - esn_fast:.4f}).")
        print("  -> REFUTES exp26's prediction: even at higher modulation depth, serial_force")
        print("     does not win on this real segment.")

    # ---- Step 6: plot ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: bar chart of modulation depth per window
    starts = [w["start_seconds"] for w in windows]
    depths = [w["modulation_depth"] for w in windows]
    colors = ["#eb6834" if w["start_seconds"] == best_start else "#9b9b9b" for w in windows]
    ax1.bar(starts, depths, width=STEP_SECONDS * 0.8, color=colors, edgecolor="black", linewidth=0.5)
    ax1.axhline(0.35, color="black", linestyle="--", alpha=0.7, label="exp18 first-120s depth (~0.35)")
    ax1.set_xlabel("window start (seconds)")
    ax1.set_ylabel("measured modulation depth")
    ax1.set_title("exp27: modulation depth by window (chosen highlighted)")
    ax1.legend(fontsize=8)

    # Right: NRMSE comparison across architectures
    arch_names = list(ARCHS.keys())
    fast_means = [summary[a][2] for a in arch_names]
    fast_stds = [summary[a][3] for a in arch_names]
    slow_means = [summary[a][0] for a in arch_names]
    slow_stds = [summary[a][1] for a in arch_names]
    x = np.arange(len(arch_names))
    width = 0.35
    ax2.bar(x - width / 2, fast_means, width, yerr=fast_stds, capsize=3,
            label="fast NRMSE", color="#eb6834")
    ax2.bar(x + width / 2, slow_means, width, yerr=slow_stds, capsize=3,
            label="slow NRMSE", color="#2a78d6")
    ax2.set_xticks(x)
    ax2.set_xticklabels(arch_names)
    ax2.set_ylabel("NRMSE (lower = better)")
    ax2.set_title(f"exp27: NRMSE on highest-depth segment (start={best_start}s)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{VISUALS_DIR}/exp27_ecg_segment_scan.png", dpi=130)
    plt.close(fig)
    print(f"\nSaved plot to {VISUALS_DIR}/exp27_ecg_segment_scan.png")


if __name__ == "__main__":
    main()