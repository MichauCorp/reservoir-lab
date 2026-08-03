"""
exp23: Does ANY optical-acoustic combination beat the relevant competitors --
a single optical reservoir alone, a single acoustic reservoir alone, and a
properly size-matched standard ESN -- across the tasks this investigation
has actually established as trustworthy?

This is the comparison the whole investigation has been implicitly assuming
rather than directly testing. exp10-22 compared different ways of COMBINING
optical and acoustic against each other (parallel vs serial_force vs
parametric vs mutual vs all_optical), but never against the more basic
question: is combining better than not combining at all, and is any of this
better than a conventional, general-purpose reservoir?

One real gap fixed here: exp19's ESN baseline used n_reservoir=100 while
parallel/serial_force get 200 total features (100+100, both taps read out)
-- a 2x size disadvantage for the ESN that made that comparison unfair.
Here, the ESN is size-matched to 200 units, the same total budget as the
combined architectures.

Architectures:
  esn_matched      -- standard ESN, n_reservoir=200 (size-matched to the
                      combined architectures' total feature budget)
  esn_small        -- standard ESN, n_reservoir=100 (matched to EACH single
                      substrate, for reference)
  optical_only      -- single OptoelectronicReservoir, n=100
  acoustic_only     -- single AcousticReservoir, n=100
  parallel         -- both substrates, independently driven, n=100 each
  serial_force      -- both substrates, force-coupled, n=100 each

Tasks (the three most trustworthy from this investigation):
  dual_timescale/sinusoid   -- where serial_force's advantage is strongest
  dual_timescale/broadband  -- where it erodes or reverses (exp20)
  real_ecg                  -- genuine physiological data, verified PAC (exp18)

Same hyperparameters and regularization (theta=0.2, c=0.3, reg=1e-3)
established as correct throughout exp10-22. 6 seeds per cell.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.exp18_real_signal_ecg import build_ecg_task
from experiments.exp20_waveform_shape import dual_timescale_task
from reservoir_lab.physical import (
    AcousticReservoir,
    OptoelectronicReservoir,
    SerialPhysicalReservoir,
)
from reservoir_lab.readout import RidgeReadout
from reservoir_lab.reservoir import ESN

WASHOUT = 120
N_STEPS = 1400
SEEDS = [1, 2, 3, 4, 5, 6]
THETA, C, N_RES, REG = 0.2, 0.3, 100, 1e-3
RESULTS_DIR = "experiments/data_results"
VISUALS_DIR = "experiments/visuals"


def nrmse(pred, true):
    true2d = true if true.ndim > 1 else true.reshape(-1, 1)
    pred2d = pred if pred.ndim > 1 else pred.reshape(-1, 1)
    return np.sqrt(np.mean((pred2d - true2d) ** 2, axis=0)) / np.std(true2d, axis=0)


# =====================================================================
# Architecture builders. Each returns (train_feats, test_feats).
# =====================================================================

def build_esn(u_train, u_test, seed, n_reservoir):
    esn = ESN(n_inputs=1, n_reservoir=n_reservoir, spectral_radius=1.2, sparsity=0.1, seed=seed)
    train_feats = esn.run(u_train)
    test_feats = esn.run(u_test, initial_state=esn.last_state)
    return train_feats, test_feats


def build_optical_only(u_train, u_test, seed):
    optical = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=N_RES, theta=THETA, seed=seed)
    train_feats = optical.run(u_train)
    test_feats = optical.run(u_test, initial_state=optical.last_state)
    return train_feats, test_feats


def build_acoustic_only(u_train, u_test, seed):
    acoustic = AcousticReservoir(n_inputs=1, n_oscillators=N_RES, c=C, seed=seed + 1)
    train_feats = acoustic.run(u_train)
    test_feats = acoustic.run(u_test, initial_state=acoustic.last_state)
    return train_feats, test_feats


def build_parallel(u_train, u_test, seed):
    optical = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=N_RES, theta=THETA, seed=seed)
    acoustic = AcousticReservoir(n_inputs=1, n_oscillators=N_RES, c=C, seed=seed + 1)
    opt_tr, ac_tr = optical.run(u_train), acoustic.run(u_train)
    opt_te = optical.run(u_test, initial_state=optical.last_state)
    ac_te = acoustic.run(u_test, initial_state=acoustic.last_state)
    return np.hstack([opt_tr, ac_tr]), np.hstack([opt_te, ac_te])


def build_serial_force(u_train, u_test, seed):
    stage1 = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=N_RES, theta=THETA, seed=seed)
    stage2 = AcousticReservoir(n_inputs=N_RES, n_oscillators=N_RES, c=C, seed=seed + 1)
    model = SerialPhysicalReservoir(stage1=stage1, stage2=stage2, combine="both", seed=seed)
    train_feats = model.run(u_train)
    test_feats = model.run(u_test, initial_state=model.last_state)
    return train_feats, test_feats


ARCHS = {
    "esn_matched": lambda utr, ute, sd: build_esn(utr, ute, sd, 200),
    "esn_small": lambda utr, ute, sd: build_esn(utr, ute, sd, 100),
    "optical_only": build_optical_only,
    "acoustic_only": build_acoustic_only,
    "parallel": build_parallel,
    "serial_force": build_serial_force,
}


# =====================================================================
# Tasks
# =====================================================================

def task_sinusoid(seed):
    u_train, y_train, u_test, y_test = dual_timescale_task(N_STEPS, "sinusoid", seed=seed)
    return u_train, y_train, u_test, y_test


def task_broadband(seed):
    u_train, y_train, u_test, y_test = dual_timescale_task(N_STEPS, "broadband", seed=seed)
    return u_train, y_train, u_test, y_test


def task_real_ecg(seed):
    u_train, y_train, u_test, y_test, _, _ = build_ecg_task(seed)
    return u_train, y_train, u_test, y_test


TASKS = {
    "sinusoid": task_sinusoid,
    "broadband": task_broadband,
    "real_ecg": task_real_ecg,
}


def evaluate(builder, u_train, y_train, u_test, y_test, seed):
    train_feats, test_feats = builder(u_train, u_test, seed)
    readout = RidgeReadout(reg=REG).fit(train_feats, y_train, washout=WASHOUT)
    pred = readout.predict(test_feats)
    return nrmse(pred, y_test)


def main():
    print("=" * 78)
    print("exp23: Does any optical-acoustic combination beat single substrates")
    print("or a size-matched standard ESN?")
    print("=" * 78)

    results = []
    for task_name, task_fn in TASKS.items():
        for seed in SEEDS:
            u_train, y_train, u_test, y_test = task_fn(seed)
            for arch_name, builder in ARCHS.items():
                try:
                    result = evaluate(builder, u_train, y_train, u_test, y_test, seed)
                except (np.linalg.LinAlgError, ValueError) as e:
                    print(f"  FAILED: {task_name}, {arch_name}, seed={seed}: {e}")
                    continue
                row = {"task": task_name, "architecture": arch_name, "seed": seed}
                if result.shape[0] == 2:
                    row["nrmse_slow"], row["nrmse_fast"] = result[0], result[1]
                else:
                    row["nrmse_slow"], row["nrmse_fast"] = result[0], None
                results.append(row)

    with open(f"{RESULTS_DIR}/exp23_vs_baselines.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "architecture", "seed", "nrmse_slow", "nrmse_fast"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved raw results to {RESULTS_DIR}/exp23_vs_baselines.csv\n")

    print(f"{'task':<12}{'architecture':<16}{'slow/primary NRMSE':<24}{'fast NRMSE'}")
    print("-" * 78)
    summary = {}
    for task_name in TASKS:
        for arch_name in ARCHS:
            slow_vals = [r["nrmse_slow"] for r in results if r["task"] == task_name and r["architecture"] == arch_name]
            fast_vals = [r["nrmse_fast"] for r in results if r["task"] == task_name and r["architecture"] == arch_name and r["nrmse_fast"] is not None]
            slow_mean, slow_std = np.mean(slow_vals), np.std(slow_vals)
            fast_str = f"{np.mean(fast_vals):.4f} +- {np.std(fast_vals):.4f}" if fast_vals else "n/a"
            summary[(task_name, arch_name)] = (slow_mean, slow_std, np.mean(fast_vals) if fast_vals else None)
            print(f"{task_name:<12}{arch_name:<16}{slow_mean:.4f} +- {slow_std:.4f}        {fast_str}")
        print()

    # Plot: fast-target NRMSE across all architectures, one panel per task
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    arch_names = list(ARCHS.keys())
    colors = ["#9b59b6", "#c39bd3", "#2a78d6", "#1baf7a", "#5dade2", "#eb6834"]
    for ax, task_name in zip(axes, TASKS, strict=False):
        means = [summary[(task_name, a)][2] for a in arch_names]
        means_plot = [m if m is not None else 0 for m in means]
        ax.bar(arch_names, means_plot, color=colors)
        ax.set_title(task_name)
        ax.set_xticks(range(len(arch_names)))
        ax.set_xticklabels(arch_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("fast-target NRMSE (lower = better)")
    fig.suptitle("exp23: fast-target NRMSE across all architectures and tasks")
    fig.tight_layout()
    fig.savefig(f"{VISUALS_DIR}/exp23_vs_baselines.png", dpi=130)
    plt.close(fig)
    print(f"Saved plot to {VISUALS_DIR}/exp23_vs_baselines.png")


if __name__ == "__main__":
    main()