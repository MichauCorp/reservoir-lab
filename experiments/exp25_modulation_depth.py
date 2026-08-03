"""
exp25: Does gate modulation depth -- not fast-component bandwidth -- explain
why serial_force loses to a standard ESN on real ECG's fast target?

exp24 mapped fast-component spectral flatness continuously and found
serial_force winning at every point tested, including flatness levels far
exceeding real ECG's actual measured value (0.0064, matched here by
alpha=0.15 in exp24's blend parametrization). That ruled out fast-component
bandwidth as a sufficient explanation for exp18/exp23's finding that a
standard ESN beats serial_force on real ECG's fast target.

The other major difference between the synthetic task and real ECG: our
synthetic gate is strongly saturating (tanh(4*slow) pins close to 0 or 1
across most of the cycle), while exp18's own verification of real
phase-amplitude coupling found a much weaker modulation depth (~35-39%,
measured via phase-binning the fast-band envelope -- see exp18's
verify_pac()). This experiment fixes the fast component's bandwidth at the
real-ECG-matched value and sweeps GATE MODULATION DEPTH instead, using the
identical phase-binning measurement exp18 used, so the x-axis is directly
comparable to the real ECG number rather than an arbitrary parameter.

If weak modulation depth is the real explanation, serial_force's advantage
should shrink or reverse near depth~0.35-0.39, the real ECG value.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert

from experiments.exp18_real_signal_ecg import verify_pac as ecg_verify_pac
from experiments.exp24_bandwidth_mapping import blended_wave
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
ALPHA_FAST = 0.15  # matches real ECG's measured fast-band spectral flatness (exp24)
DEPTHS = [0.05, 0.10, 0.18, 0.30, 0.5, 0.75, 1.0]  # 0.18 measured ~matches real ECG (0.348)


def measured_modulation_depth(slow, fast_target):
    """Same phase-binning methodology as exp18's verify_pac(), applied to
    the synthetic task's slow/fast_target pair, so the x-axis is directly
    comparable to the real ECG measurement."""
    envelope = np.abs(hilbert(fast_target))
    envelope_smooth = uniform_filter1d(envelope, size=15)
    slow_phase = np.angle(hilbert(slow - slow.mean()))
    bins = np.linspace(-np.pi, np.pi, 19)
    idx = np.digitize(slow_phase, bins)
    means = np.array([envelope_smooth[idx == b].mean() for b in range(1, 19) if np.any(idx == b)])
    return float((means.max() - means.min()) / means.mean())


def dual_timescale_task_depth(n_steps, depth, seed=0, noise_std=0.35):
    """alpha_fast fixed at ALPHA_FAST (real-ECG-matched bandwidth); depth
    scales the gate's deviation from its 0.5 baseline -- depth=1.0 matches
    every prior experiment's saturating tanh gate, depth->0 approaches no
    gating at all."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)

    slow = 0.6 * np.sin(2 * np.pi * t / 180.0)
    gate_full = 0.5 * (1 + np.tanh(4 * slow))
    gate = 0.5 + depth * (gate_full - 0.5)

    fast_phase = 2 * np.pi * t / 12.0 + 0.6 * np.sin(2 * np.pi * t / 33.0)
    fast_unit = blended_wave(fast_phase, ALPHA_FAST)
    fast_raw = 0.3 * fast_unit
    noise = rng.normal(0, noise_std, n_steps)

    fast_target = fast_raw * gate
    u = (slow + fast_raw + noise).reshape(-1, 1)
    y = np.stack([slow, fast_target], axis=1)

    split = int(n_steps * TRAIN_FRAC)
    return u[:split], y[:split], u[split:], y[split:], slow, fast_target


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


def real_ecg_modulation_depth():
    """Reuses exp18's own verify_pac() on real ECG for a directly
    comparable reference value, rather than a separately-implemented
    measurement that might not match exactly."""
    from scipy.datasets import electrocardiogram
    from scipy.signal import butter, sosfiltfilt

    ecg = electrocardiogram()
    fs = 360.0
    seg = ecg[: int(120 * fs)]
    seg = (seg - seg.mean()) / seg.std()
    sos_low = butter(4, 2.0, btype="low", fs=fs, output="sos")
    sos_fast = butter(4, [8.0, 30.0], btype="band", fs=fs, output="sos")
    low = sosfiltfilt(sos_low, seg)
    fast_band = sosfiltfilt(sos_fast, seg)
    _, mod_depth = ecg_verify_pac(low, fast_band)
    return mod_depth


def main():
    print("=" * 78)
    print("exp25: gate modulation depth sweep at fixed real-ECG-matched bandwidth")
    print("=" * 78)
    print(f"Fast component fixed at alpha={ALPHA_FAST} (matches real ECG spectral flatness)\n")

    results = []
    depth_measured = {}
    for depth in DEPTHS:
        for seed in SEEDS:
            u_train, y_train, u_test, y_test, slow, fast_target = dual_timescale_task_depth(N_STEPS, depth, seed=seed)
            if depth not in depth_measured:
                depth_measured[depth] = measured_modulation_depth(slow, fast_target)

            for arch_name, builder in ARCHS.items():
                result = evaluate(builder, u_train, y_train, u_test, y_test, seed)
                results.append({
                    "depth_param": depth, "measured_depth": depth_measured[depth], "seed": seed,
                    "architecture": arch_name, "nrmse_slow": result[0], "nrmse_fast": result[1],
                })

    with open(f"{RESULTS_DIR}/exp25_modulation_depth.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["depth_param", "measured_depth", "seed", "architecture", "nrmse_slow", "nrmse_fast"])
        writer.writeheader()
        writer.writerows(results)

    ecg_depth = real_ecg_modulation_depth()
    print(f"Real ECG measured modulation depth (exp18 methodology): {ecg_depth:.4f}\n")

    print(f"{'depth_param':<14}{'measured':<12}{'architecture':<16}{'slow NRMSE':<20}{'fast NRMSE'}")
    print("-" * 82)
    summary = {}
    for depth in DEPTHS:
        for arch_name in ARCHS:
            slow_vals = [r["nrmse_slow"] for r in results if r["depth_param"] == depth and r["architecture"] == arch_name]
            fast_vals = [r["nrmse_fast"] for r in results if r["depth_param"] == depth and r["architecture"] == arch_name]
            summary[(depth, arch_name)] = (np.mean(slow_vals), np.std(slow_vals), np.mean(fast_vals), np.std(fast_vals))
            print(f"{depth:<14}{depth_measured[depth]:<12.4f}{arch_name:<16}"
                  f"{np.mean(slow_vals):.4f}+-{np.std(slow_vals):.4f}      "
                  f"{np.mean(fast_vals):.4f}+-{np.std(fast_vals):.4f}")
        print()

    fig, ax = plt.subplots(figsize=(8, 5))
    measured_vals = [depth_measured[d] for d in DEPTHS]
    colors = {"serial_force": "#eb6834", "esn_matched": "#9b59b6"}
    for arch_name in ARCHS:
        means = [summary[(d, arch_name)][2] for d in DEPTHS]
        stds = [summary[(d, arch_name)][3] for d in DEPTHS]
        ax.errorbar(measured_vals, means, yerr=stds, marker="o", label=arch_name, color=colors[arch_name], capsize=3)
    ax.axvline(ecg_depth, color="black", linestyle="--", alpha=0.7, label=f"real ECG ({ecg_depth:.3f})")
    ax.set_xlabel("measured gate modulation depth (phase-binned, same method as exp18)")
    ax.set_ylabel("fast-target NRMSE (lower = better)")
    ax.set_title("exp25: does weak coupling strength explain the real-ECG crossover?")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{VISUALS_DIR}/exp25_modulation_depth.png", dpi=130)
    plt.close(fig)
    print(f"Saved plot to {VISUALS_DIR}/exp25_modulation_depth.png")

    print("\n" + "=" * 78)
    print("CONCLUSION")
    print("=" * 78)
    winners = []
    for depth in DEPTHS:
        sf_fast = summary[(depth, "serial_force")][2]
        esn_fast = summary[(depth, "esn_matched")][2]
        winner = "serial_force" if sf_fast < esn_fast else "esn_matched"
        winners.append((depth, depth_measured[depth], winner))
        print(f"  depth_param={depth} (measured={depth_measured[depth]:.4f}): {winner} wins "
              f"(serial_force={sf_fast:.4f}, esn_matched={esn_fast:.4f})")

    # Find where it switches FROM esn_matched winning TO serial_force winning
    # (the pattern here is esn_matched wins at low/moderate depth, serial_force
    # only takes over at high, near-saturating depth -- the opposite of a
    # simple monotonic story, so report the actual switch point explicitly).
    switch_to_sf = None
    for i in range(1, len(winners)):
        if winners[i - 1][2] == "esn_matched" and winners[i][2] == "serial_force":
            switch_to_sf = winners[i]
            break

    if switch_to_sf is not None:
        esn_win_range = [w for w in winners if w[2] == "esn_matched"]
        print(f"esn_matched wins across measured depth {esn_win_range[0][1]:.3f}-{esn_win_range[-1][1]:.3f}; "
              f"serial_force only takes over at measured depth >= {switch_to_sf[1]:.3f} "
              f"(near-saturating gating, close to every prior synthetic task's default).")
        print(f"\nReal ECG's measured modulation depth: {ecg_depth:.4f}")
        if esn_win_range[0][1] <= ecg_depth <= esn_win_range[-1][1]:
            print("Real ECG falls WITHIN the esn_matched-favoring range.")
            print("This is a much cleaner explanation than exp24's bandwidth hypothesis: real ECG's")
            print("coupling is simply too WEAK (partial, physiological PAC) to fall in the regime")
            print("where serial_force's advantage appears -- every prior synthetic task used")
            print("near-saturating gating (depth~1.9-2.0 on this scale), far stronger than any")
            print("real phase-amplitude coupling is likely to be.")
        else:
            print("Real ECG does not fall within the esn_matched-favoring range -- modulation depth")
            print("alone does not fully explain the real-data result either.")
    else:
        print("\nNo consistent switch pattern found across the depths tested.")


if __name__ == "__main__":
    main()