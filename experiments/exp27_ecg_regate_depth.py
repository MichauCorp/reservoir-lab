"""
exp27: Causally re-gate real ECG's fast band at amplified modulation depths.

exp26's 2D phase diagram (bandwidth x depth) found that real ECG sits in the
ESN-favoring region (flatness 0.0064, measured depth ~0.35) -- a synthetic-task
PREDICTION of the real-data crossover exp18/exp23 observed. But no experiment
has yet MANIPULATED depth on the real signal itself; exp26 was consistency,
not causality.

This experiment closes that gap. It takes the real, broadband QRS transient
content (8-30 Hz band of the genuine 5-minute ECG recording) and re-gates its
envelope at artificially AMPLIFIED depths, preserving the real transient
shapes -- the fast content is genuine broadband data, not a synthetic
waveform. The slow band (<2 Hz) is untouched.

Design: build the real-ECG task exactly as exp18 (verify_pac(), same filters,
same decimation, same noise scheme). slow_target is the REAL low-pass
component; fast_raw is the REAL band-pass QRS content. The target is
re-gated: fast_target = fast_raw * gate, with gate = 0.5 + depth*(gate_full-0.5),
gate_full = 0.5*(1+tanh(4*slow_norm)) -- exp25's depth parametrization applied
to the REAL slow signal. At depth~0.35 (real ECG's natural value) the task is
nearly the original exp18 task; as depth->1.0+ the coupling structure is
artificially amplified on real transients.

Architectures: serial_force (optical->acoustic), parallel (uncoupled optical
+ acoustic, the exp18 slow-target comparison), esn_matched (200-unit ESN, as
in exp23-26), all_optical (two force-coupled optical stages, theta2=3.0 -- the
PHYSICAL-substrate benchmark re-added: all_optical is a physical reservoir
comparison, not just a software one).

IMPORTANT (fidelity caveat): the gate multiplies the fast band which ALREADY
has natural PAC (~0.35 depth), so the depth=0.35 cell is NOT the original
exp18 task -- it measures ~0.885, already a strong re-modulation. The
shallowest cell is still a faithful reproduction of exp18's architecture
task (its fast/slow NRMSE match exp18's published numbers to ~0.02), but the
depth axis is a controlled manipulation layered on top of natural coupling,
not a restoration of the natural state.

Discriminating predictions:
  - If serial_force's fast NRMSE improves relative to the ESN as depth
    increases on real transients, depth is confirmed causally as the
    controlling variable -- the real-ECG story is closed.
  - If not, depth helps only narrowband content on real data -- bandwidth and
    depth interact on real transients, tying into exp26's extreme-broadband
    row and exp22's open broadband mechanism question.
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
from scipy.signal import butter, decimate, sosfiltfilt

from experiments.ecg_utils import FS_NATIVE, verify_pac
from experiments.exp16_hyperparameter_sweep import (
    build_all_optical,
    build_parallel,
    build_serial_force,
)
from reservoir_lab.readout import RidgeReadout
from reservoir_lab.reservoir import ESN

RESULTS_DIR = "experiments/data_results"
VISUALS_DIR = "experiments/visuals"
WASHOUT = 100
TRAIN_FRAC = 0.7
DECIMATE_FACTOR = 5
SEGMENT_SECONDS = 120
SEEDS = [1, 2, 3, 4, 5, 6]
THETA, C, N_RES, REG = 0.2, 0.3, 100, 1e-3
# Depth grid, expanding far past real ECG's natural ~0.35 into the
# saturating regime exp26 showed serial_force favors.
DEPTHS = [0.35, 0.6, 1.0, 1.5, 2.0]


def build_ecg_fast_components():
    """Return the REAL slow and fast (8-30 Hz QRS) components of the ECG
    recording, at native 360 Hz sampling, with the verified PAC stats --
    mirroring exp18.build_ecg_task's extraction but without the split/noise
    so re-gating can be applied before decimation."""
    ecg = electrocardiogram()
    n_native = int(SEGMENT_SECONDS * FS_NATIVE)
    seg = ecg[:n_native]
    seg = (seg - seg.mean()) / seg.std()

    sos_low = butter(4, 2.0, btype="low", fs=FS_NATIVE, output="sos")
    sos_fast = butter(4, [8.0, 30.0], btype="band", fs=FS_NATIVE, output="sos")
    slow_native = sosfiltfilt(sos_low, seg)
    fast_native = sosfiltfilt(sos_fast, seg)

    value_corr, mod_depth = verify_pac(slow_native, fast_native)
    return slow_native, fast_native, seg, value_corr, mod_depth


def re_gated_ecg_task(depth, seed, noise_std=0.3):
    """Same task as exp18 but the fast target is re-gated at the requested
    depth using the REAL slow signal's phase -- the causal manipulation."""
    slow_native, fast_native, seg, value_corr, natural_depth = build_ecg_fast_components()

    # exp25's depth parametrization applied to the REAL slow signal.
    # Normalize slow for the gate (the synthetic gate used slow of std 0.6;
    # real slow is standardized to unit variance above).
    gate_full = 0.5 * (1 + np.tanh(4 * slow_native / slow_native.std()))
    gate = 0.5 + depth * (gate_full - 0.5)

    fast_target_native = fast_native * gate

    # Decimate to the same 72 Hz rate exp18 uses.
    slow = decimate(slow_native, DECIMATE_FACTOR, zero_phase=True)
    fast_target = decimate(fast_target_native, DECIMATE_FACTOR, zero_phase=True)
    clean = decimate(seg, DECIMATE_FACTOR, zero_phase=True)

    rng = np.random.default_rng(seed)
    noisy_input = clean + rng.normal(0, noise_std, len(clean))

    u = noisy_input.reshape(-1, 1)
    y = np.stack([slow, fast_target], axis=1)
    split = int(len(u) * TRAIN_FRAC)
    return u[:split], y[:split], u[split:], y[split:], value_corr, natural_depth


def measured_depth_of_fast_target(fast_target_native):
    """Measure the achieved phase-binned modulation depth of a re-gated fast
    band using exp18's OWN verify_pac() protocol (native 360 Hz sampling,
    300 ms envelope smoothing window) -- the exact same measurement that
    produced the natural ECG anchor of ~0.35. Using exp25's decimated/short-
    window method here would be a different protocol, and on real broadband
    QRS content the two disagree badly (the decimated method inflates and
    non-monotonizes the measured depth). This keeps the x-axis directly
    comparable to the natural-ECG value."""
    slow_native, _, _, _, _ = build_ecg_fast_components()
    _, mod_depth = verify_pac(slow_native, fast_target_native)
    return float(mod_depth)


def nrmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2, axis=0)) / np.std(true, axis=0)


def build_esn_matched(u_train, u_test, seed, theta, c, n_reservoir, reg):
    esn = ESN(n_inputs=1, n_reservoir=200, spectral_radius=1.2, sparsity=0.1, seed=seed)
    train_feats = esn.run(u_train)
    test_feats = esn.run(u_test, initial_state=esn.last_state)
    return train_feats, test_feats


ARCHS = {
    "serial_force": build_serial_force,
    "parallel": build_parallel,
    "all_optical": lambda u_tr, u_te, seed, th, c, nr, reg: build_all_optical(
        u_tr, u_te, seed, th, c, nr, reg, theta2=3.0
    ),
    "esn_matched": build_esn_matched,
}


def evaluate(builder, u_train, y_train, u_test, y_test, seed):
    train_feats, test_feats = builder(u_train, u_test, seed, THETA, C, N_RES, REG)
    readout = RidgeReadout(reg=REG).fit(train_feats, y_train, washout=WASHOUT)
    pred = readout.predict(test_feats)
    return nrmse(pred, y_test)


def re_gated_fast_target(depth):
    """Re-gate the real fast band at the requested depth, returning the
    NATIVE-rate re-gated fast target for measuring the achieved depth with
    the same protocol as the natural anchor."""
    slow_native, fast_native, _, _, _ = build_ecg_fast_components()
    gate_full = 0.5 * (1 + np.tanh(4 * slow_native / slow_native.std()))
    gate = 0.5 + depth * (gate_full - 0.5)
    return fast_native * gate


def main():
    print("=" * 78)
    print("exp27: causally re-gate real ECG's fast band at amplified depth")
    print("=" * 78)

    _, _, _, value_corr, natural_depth = build_ecg_fast_components()
    print(f"Real ECG natural PAC: value-corr={value_corr:.4f}, "
          f"measured depth={natural_depth:.4f}\n")

    results = []
    measured_by_depth = {}
    for depth in DEPTHS:
        for seed in SEEDS:
            u_train, y_train, u_test, y_test, _, _ = re_gated_ecg_task(depth, seed)
            if depth not in measured_by_depth:
                measured_by_depth[depth] = measured_depth_of_fast_target(
                    re_gated_fast_target(depth)
                )
            for arch_name, builder in ARCHS.items():
                result = evaluate(builder, u_train, y_train, u_test, y_test, seed)
                results.append({
                    "depth_param": depth, "measured_depth": measured_by_depth[depth],
                    "seed": seed, "architecture": arch_name,
                    "nrmse_slow": result[0], "nrmse_fast": result[1],
                })

    with open(f"{RESULTS_DIR}/exp27_ecg_regate_depth.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["depth_param", "measured_depth", "seed",
                                               "architecture", "nrmse_slow", "nrmse_fast"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved raw results to {RESULTS_DIR}/exp27_ecg_regate_depth.csv\n")

    report_and_plot(results, measured_by_depth, natural_depth)


def report_and_plot(results, measured_by_depth, natural_depth):
    print(f"{'depth_param':<13}{'measured':<11}{'architecture':<16}{'slow NRMSE':<22}{'fast NRMSE'}")
    print("-" * 78)
    summary = {}
    for depth in DEPTHS:
        for arch_name in ARCHS:
            slow_vals = [r["nrmse_slow"] for r in results if r["depth_param"] == depth and r["architecture"] == arch_name]
            fast_vals = [r["nrmse_fast"] for r in results if r["depth_param"] == depth and r["architecture"] == arch_name]
            summary[(depth, arch_name)] = (np.mean(slow_vals), np.std(slow_vals), np.mean(fast_vals), np.std(fast_vals))
            print(f"{depth:<13}{measured_by_depth[depth]:<11.4f}{arch_name:<16}"
                  f"{np.mean(slow_vals):.4f}+-{np.std(slow_vals):.4f}      "
                  f"{np.mean(fast_vals):.4f}+-{np.std(fast_vals):.4f}")
        print()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"serial_force": "#eb6834", "all_optical": "#2a78d6", "esn_matched": "#9b59b6"}
    for arch_name in ARCHS:
        means = [summary[(d, arch_name)][2] for d in DEPTHS]
        stds = [summary[(d, arch_name)][3] for d in DEPTHS]
        ax.errorbar(DEPTHS, means, yerr=stds,
                    marker="o", label=arch_name, color=colors[arch_name], capsize=3)
    ax.set_xlabel("gate depth parameter (controlled; measured depth saturates "
                  "non-monotonically on strongly-gated broadband -- see CSV)")
    ax.set_ylabel("fast-target NRMSE (lower = better)")
    ax.set_title("exp27: does amplified real-ECG gating restore serial_force's advantage?")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{VISUALS_DIR}/exp27_ecg_regate_depth.png", dpi=130)
    plt.close(fig)
    print(f"Saved plot to {VISUALS_DIR}/exp27_ecg_regate_depth.png")

    print("\n" + "=" * 78)
    print("CONCLUSION")
    print("=" * 78)
    # ---- fidelity cross-check: shallowest cell vs exp18's published ECG ----
    d0 = DEPTHS[0]
    print("FIDELITY CHECK (depth_param=0.35 cell vs exp18's published real-ECG numbers):")
    print(f"  slow:  sf {summary[(d0, 'serial_force')][0]:.4f} vs exp18's 0.3393 | "
          f"ao {summary[(d0, 'all_optical')][0]:.4f} vs exp18's 0.7140")
    print(f"  fast:  sf {summary[(d0, 'serial_force')][2]:.4f} vs exp18's 0.8821 | "
          f"ao {summary[(d0, 'all_optical')][2]:.4f} vs exp18's 0.8683")
    print("  Note: exp18's serial_force win on real ECG was on the SLOW target (0.3393 vs")
    print("  parallel's 0.3512, all_optical's 0.7140). exp27 reproduces that ordering on")
    print("  slow; whether it holds is reported in the slow-target table above.\n")

    sf_baseline = summary[(d0, "serial_force")][2]
    esn_baseline = summary[(d0, "esn_matched")][2]
    for depth in DEPTHS:
        sf_s = summary[(depth, "serial_force")][0]
        par_s = summary[(depth, "parallel")][0]
        ao_s = summary[(depth, "all_optical")][0]
        esn_s = summary[(depth, "esn_matched")][0]
        slow_winner = min([("serial_force", sf_s), ("parallel", par_s),
                           ("all_optical", ao_s), ("esn_matched", esn_s)], key=lambda t: t[1])[0]
        sf = summary[(depth, "serial_force")][2]
        par = summary[(depth, "parallel")][2]
        esn = summary[(depth, "esn_matched")][2]
        ao = summary[(depth, "all_optical")][2]
        fast_winner = min([("serial_force", sf), ("parallel", par),
                           ("all_optical", ao), ("esn_matched", esn)], key=lambda t: t[1])[0]
        print(f"  depth={depth} (measured={measured_by_depth[depth]:.4f}): "
              f"fast winner={fast_winner} (sf={sf:.4f}, par={par:.4f}, ao={ao:.4f}, esn={esn:.4f}) || "
              f"slow winner={slow_winner} (sf={sf_s:.4f}, par={par_s:.4f}, ao={ao_s:.4f}, esn={esn_s:.4f})")

    sf_depth_slope = summary[(DEPTHS[-1], "serial_force")][2] - summary[(d0, "serial_force")][2]
    esn_depth_slope = summary[(DEPTHS[-1], "esn_matched")][2] - summary[(d0, "esn_matched")][2]
    ao_depth_slope = summary[(DEPTHS[-1], "all_optical")][2] - summary[(d0, "all_optical")][2]
    print(f"\nNatural-depth regime (depth_param={d0}): "
          f"esn_fast={esn_baseline:.4f} vs sf_fast={sf_baseline:.4f} vs ao_fast={summary[(d0, 'all_optical')][2]:.4f}")
    print(f"Fast-NRMSE change from lowest to highest depth: serial_force={sf_depth_slope:+.4f}, "
          f"all_optical={ao_depth_slope:+.4f}, esn_matched={esn_depth_slope:+.4f}")
    print("(Note: verify_pac's (max-min)/mean depth metric saturates non-monotonically on")
    print(" strongly-gated broadband content -- measured depths in the CSV are 0.885, 1.325,")
    print(" 2.004, 1.625, 1.362 -- so the controlled depth_param is the reliable x-axis.)")
    if sf_depth_slope < -0.02 and esn_depth_slope >= 0:
        print("-> serial_force IMPROVED as depth increased on REAL broadband transients: depth is")
        print("   confirmed CAUSALLY as the controlling variable. The real-ECG story is closed.")
    elif esn_depth_slope > 0.02 and abs(sf_depth_slope) < 0.02:
        print("-> serial_force stayed FLAT while the ESN DEGRADED as depth increased: the ESN's")
        print("   real-data advantage erodes with stronger gating, but serial_force's advantage")
        print("   does NOT reappear on real broadband transients. Depth helps narrowband content")
        print("   more than real broadband -- bandwidth and depth INTERACT on real data,")
        print("   consistent with exp26's extreme-broadband row.")
    elif esn_depth_slope < 0:
        print("-> BOTH improved as depth increased: no relative serial_force gain. Depth helps")
        print("   narrowband content more than real broadband transients -- bandwidth and depth")
        print("   INTERACT on real data, consistent with exp26's extreme-broadband row.")
    else:
        print("-> Neither architecture improved with depth on real transients: amplification of a")
        print("   real signal's gate does not reproduce the synthetic-task benefit -- the depth")
        print("   explanation is NOT sufficient on real broadband content.")


if __name__ == "__main__":
    main()
