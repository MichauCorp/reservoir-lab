"""
exp22: Does tuning optical's own timescale (theta), rather than filtering
the input, recover serial_force's broadband slow-target performance
WITHOUT destroying fast-target performance?

Motivation: exp21 confirmed the mechanism -- broadband fast content
reaching optical unfiltered degrades serial_force's slow-target tracking
-- but its "fix" (a low-pass filter removing the fast content before any
reservoir sees it) also destroyed fast-target performance completely
(NRMSE ~1.0, no better than predicting the mean). exp21's own motivating
hypothesis was that this happens because optical's SHORT memory
(theta=0.2) smears sharp transients into a noisier drive for acoustic.
If that's the real mechanism, increasing optical's own theta -- longer
intrinsic memory, less smearing -- should recover slow-target performance
WITHOUT deleting any information, which would be a genuine fix rather
than the all-or-nothing tradeoff exp21 found.

Design: sweep optical theta over a range for serial_force and parallel, on
both waveform types from exp20 (sinusoid as a reference showing the sweep
doesn't just uniformly help everything; broadband is the case of
interest). c, n_reservoir, reg held fixed at the values used throughout
exp16-21. Same task generator as exp20/21 (imported directly, not
reimplemented), same 6 seeds.

What "success" would look like: a theta value where serial_force's
broadband slow-target NRMSE drops toward its sinusoid level (~0.21)
WHILE fast-target NRMSE stays well below 1.0 (unlike exp21's prefilter,
which traded fast-target performance away entirely to fix slow).

IMPORTANT METHODOLOGICAL NOTE: exp16's build_serial_force/build_parallel
do not accept an explicit `dt` for the optical stage -- OptoelectronicReservoir
falls back to dt=theta/10, which means sweeping theta ALSO stretches the
virtual-node delay window (tau = n_nodes * dt) proportionally. This is
the exact same confound identified and fixed in exp14 (for all_optical)
and reintroduced in exp16 (also for all_optical, before being fixed
again). To test theta's effect on memory/response-time in isolation --
not entangled with delay-window size -- this file defines its own local
builders with dt fixed at 0.02 (matching theta=0.2's default) across the
whole sweep, rather than reusing exp16's confounded builders.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.exp20_waveform_shape import dual_timescale_task
from reservoir_lab.physical import (
    AcousticReservoir,
    OptoelectronicReservoir,
    SerialPhysicalReservoir,
)
from reservoir_lab.readout import RidgeReadout

RESULTS_DIR = "experiments/data_results"
VISUALS_DIR = "experiments/visuals"
WASHOUT = 120
N_STEPS = 1400
SEEDS = [1, 2, 3, 4, 5, 6]
C, N_RES, REG = 0.3, 100, 1e-3
THETAS = [0.2, 0.4, 0.8, 1.5, 3.0, 6.0]
OPTICAL_DT = 0.02  # fixed across the sweep -- decouples response-time from delay-window size


def build_serial_force_fixed_dt(u_train, u_test, seed, theta):
    optical = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=N_RES, theta=theta, dt=OPTICAL_DT, seed=seed)
    acoustic = AcousticReservoir(n_inputs=N_RES, n_oscillators=N_RES, c=C, seed=seed + 1)
    model = SerialPhysicalReservoir(stage1=optical, stage2=acoustic, combine="both", seed=seed)
    train_feats = model.run(u_train)
    test_feats = model.run(u_test, initial_state=model.last_state)
    return train_feats, test_feats


def build_parallel_fixed_dt(u_train, u_test, seed, theta):
    optical = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=N_RES, theta=theta, dt=OPTICAL_DT, seed=seed)
    acoustic = AcousticReservoir(n_inputs=1, n_oscillators=N_RES, c=C, seed=seed + 1)
    opt_tr, ac_tr = optical.run(u_train), acoustic.run(u_train)
    opt_te = optical.run(u_test, initial_state=optical.last_state)
    ac_te = acoustic.run(u_test, initial_state=acoustic.last_state)
    return np.hstack([opt_tr, ac_tr]), np.hstack([opt_te, ac_te])


ARCHS = {"serial_force": build_serial_force_fixed_dt, "parallel": build_parallel_fixed_dt}


def nrmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2, axis=0)) / np.std(true, axis=0)


def evaluate(builder, u_train, y_train, u_test, y_test, seed, theta):
    train_feats, test_feats = builder(u_train, u_test, seed, theta)
    readout = RidgeReadout(reg=REG).fit(train_feats, y_train, washout=WASHOUT)
    pred = readout.predict(test_feats)
    return nrmse(pred, y_test)


def main():
    print("=" * 74)
    print("exp22: Does tuning optical's theta (not filtering) fix broadband")
    print("performance without destroying the fast target?")
    print("=" * 74)

    results = []
    for waveform in ["sinusoid", "broadband"]:
        for theta in THETAS:
            for seed in SEEDS:
                u_train, y_train, u_test, y_test = dual_timescale_task(N_STEPS, waveform, seed=seed)
                for arch_name, builder in ARCHS.items():
                    result = evaluate(builder, u_train, y_train, u_test, y_test, seed, theta)
                    results.append({
                        "waveform": waveform, "theta": theta, "seed": seed, "architecture": arch_name,
                        "nrmse_slow": result[0], "nrmse_fast": result[1],
                    })

    with open(f"{RESULTS_DIR}/exp22_optical_theta_sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["waveform", "theta", "seed", "architecture", "nrmse_slow", "nrmse_fast"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved raw results to {RESULTS_DIR}/exp22_optical_theta_sweep.csv\n")

    summary = {}
    for waveform in ["sinusoid", "broadband"]:
        for theta in THETAS:
            for arch_name in ARCHS:
                cell = [r for r in results if r["waveform"] == waveform and r["theta"] == theta and r["architecture"] == arch_name]
                slow_vals = [r["nrmse_slow"] for r in cell]
                fast_vals = [r["nrmse_fast"] for r in cell]
                summary[(waveform, theta, arch_name)] = (
                    np.mean(slow_vals), np.std(slow_vals), np.mean(fast_vals), np.std(fast_vals),
                )

    print(f"{'waveform':<12}{'theta':<8}{'architecture':<16}{'slow NRMSE':<20}{'fast NRMSE'}")
    print("-" * 74)
    for waveform in ["sinusoid", "broadband"]:
        for theta in THETAS:
            for arch_name in ARCHS:
                sm, ss, fm, fs = summary[(waveform, theta, arch_name)]
                print(f"{waveform:<12}{theta:<8}{arch_name:<16}{sm:.4f}+-{ss:.4f}      {fm:.4f}+-{fs:.4f}")
        print()

    # Plot: slow and fast NRMSE vs theta, broadband only (the case of interest)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"serial_force": "#eb6834", "parallel": "#2a78d6"}
    for idx, target in enumerate(["slow", "fast"]):
        ax = axes[idx]
        target_i = 0 if target == "slow" else 2
        std_i = 1 if target == "slow" else 3
        for arch_name in ARCHS:
            means = [summary[("broadband", th, arch_name)][target_i] for th in THETAS]
            stds = [summary[("broadband", th, arch_name)][std_i] for th in THETAS]
            ax.errorbar(THETAS, means, yerr=stds, marker="o", label=arch_name, color=colors[arch_name], capsize=3)
        ax.set_xlabel("optical theta")
        ax.set_ylabel(f"{target}-target NRMSE (lower = better)")
        ax.set_title(f"Broadband, {target} target")
        ax.legend(fontsize=8)
        ax.set_xscale("log")
    fig.suptitle("exp22: Optical theta sweep on broadband -- does longer memory fix both targets?", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{VISUALS_DIR}/exp22_optical_theta_sweep.png", dpi=130)
    plt.close(fig)
    print(f"Saved plot to {VISUALS_DIR}/exp22_optical_theta_sweep.png")

    # Conclusion: is there a theta where serial_force's broadband slow-target
    # approaches its sinusoid level AND fast-target stays well below 1.0?
    sinusoid_sf_slow_baseline = summary[("sinusoid", 0.2, "serial_force")][0]
    print("\n" + "=" * 74)
    print("CONCLUSION")
    print("=" * 74)
    print(f"serial_force sinusoid slow-target baseline (theta=0.2): {sinusoid_sf_slow_baseline:.4f}")
    best_theta, best_slow, best_fast = None, np.inf, None
    for theta in THETAS:
        slow, _, fast, _ = summary[("broadband", theta, "serial_force")]
        print(f"  theta={theta:<6} broadband serial_force: slow={slow:.4f}  fast={fast:.4f}")
        if slow < best_slow and fast < 0.95:  # meaningfully better than predicting the mean
            best_theta, best_slow, best_fast = theta, slow, fast
    if best_theta is not None:
        print(f"\nBest candidate: theta={best_theta}, slow={best_slow:.4f}, fast={best_fast:.4f}")
    else:
        print("\nNo theta value achieved both a slow-target improvement and a non-degenerate fast-target NRMSE.")


if __name__ == "__main__":
    main()