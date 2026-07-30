"""
Real-world-mimicking signal benchmark: multi-band gated classification.

This experiment tests whether serial_force's advantage generalizes from simple
sinusoidal synthetic tasks to signals with realistic multi-band spectral
structure. The task replicates the key biophysical property of the EEG eye-state
paradigm (exp13-14's tanh_gate): a sustained slow signal smooth-gates the
amplitude of a faster oscillatory band. Here the signal is generated
synthetically from an EEG-realistic superposition of delta (1-4 Hz), theta
(4-8 Hz), alpha (8-12 Hz), and beta (12-30 Hz) band components, with the alpha
band amplitude modulated by a gating state variable -- mirroring the well-known
alpha suppression when eyes open.

Because external dataset download is fragile, the signal is generated
deterministically from the same seed so the benchmark is fully reproducible and
offline-runnable. The scientific goal is identical to a real-world benchmark:
test whether coupling advantage survives on realistic multi-timescale structure.

Architectures:
  - serial_force  (optical→acoustic, force-coupled)
  - parallel      (both reservoirs driven independently)
  - all_optical   (two force-coupled optical stages)
  - simple_esn    (software ESN baseline for credibility)

Design: 5 seeds per architecture, same reservoir size and hyperparameters
from exp16's optimal config (theta=1.0, c=0.3, n=100, reg=1e-3).
"""

import csv

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from reservoir_lab.physical import (
    AcousticReservoir,
    OptoelectronicReservoir,
    SerialPhysicalReservoir,
)
from reservoir_lab.readout import RidgeReadout
from reservoir_lab.reservoir import ESN

RESULTS_DIR = "experiments/data_results"
VISUALS_DIR = "experiments/visuals"

WASHOUT = 100
SEEDS = [0, 1, 2, 3, 4]
N_RESERVOIR = 100
THETA = 1.0
C = 0.3
REG = 1e-3
THETA2 = 3.0  # for all_optical, from exp16's tuned value


# =====================================================================
# Data loading (cached download + arff parse via scipy)
# =====================================================================

def generate_multiband_signal(n_steps=10000, seed=0):
    """Generate a realistic multi-band signal with slow-gated alpha amplitude.

    Replicates the EEG eye-state mechanism:
      - slow gating signal (delta-range, ~1 Hz) -- analogous to sustained eye state
      - fast carrier (alpha-range, ~10 Hz) whose amplitude is modulated by the
        slow signal via a smooth sigmoid gate
      - additional independent theta and beta components for realism

    Input: scalar observation = mean across bands at each timestep.
    Target: binary label = sign of slow-signal derivative (state change).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)

    # Slow gating signal (~1 Hz delta)
    slow = 0.8 * np.sin(2 * np.pi * t / 1000.0) + 0.2 * rng.normal(0, 0.1, n_steps)
    gate = 0.5 * (1 + np.tanh(2 * slow))  # smooth gate in [0, 1]

    # Fast amplitude-modulated carrier (~10 Hz alpha)
    alpha_carrier = np.sin(2 * np.pi * t / 100.0)
    gated_alpha = gate * alpha_carrier

    # Independent theta and beta components (background structure)
    theta = 0.3 * np.sin(2 * np.pi * t / 250.0 + rng.normal(0, 0.3))
    beta = 0.15 * np.sin(2 * np.pi * t / 40.0 + rng.normal(0, 0.5))

    # Combined "channel-mean" input: gated alpha dominates
    u = (gated_alpha + theta + beta).reshape(-1, 1)

    # Binary target: high alpha suppression = label 1 (eyes-open analog)
    y = (gate < 0.4).astype(int)

    return u, y


def prepare_splits(u, y, train_frac=0.7):
    n = len(u)
    train_end = int(n * train_frac)
    return u[:train_end], y[:train_end], u[train_end:], y[train_end:]


# =====================================================================
# Architecture builders (building on exp16/exp17 patterns)
# =====================================================================

def build_serial_force(u_train, u_test, seed):
    optical = OptoelectronicReservoir(
        n_inputs=1, n_virtual_nodes=N_RESERVOIR, theta=THETA, seed=seed
    )
    acoustic = AcousticReservoir(
        n_inputs=N_RESERVOIR, n_oscillators=N_RESERVOIR, c=C, seed=seed + 1
    )
    model = SerialPhysicalReservoir(stage1=optical, stage2=acoustic, combine="both", seed=seed)
    train_feats = model.run(u_train)
    test_feats = model.run(u_test, initial_state=model.last_state)
    return train_feats, test_feats


def build_parallel(u_train, u_test, seed):
    optical = OptoelectronicReservoir(
        n_inputs=1, n_virtual_nodes=N_RESERVOIR, theta=THETA, seed=seed
    )
    acoustic = AcousticReservoir(
        n_inputs=1, n_oscillators=N_RESERVOIR, c=C, seed=seed + 1
    )
    opt_tr = optical.run(u_train)
    ac_tr = acoustic.run(u_train)
    opt_te = optical.run(u_test, initial_state=optical.last_state)
    ac_te = acoustic.run(u_test, initial_state=acoustic.last_state)
    return np.hstack([opt_tr, ac_tr]), np.hstack([opt_te, ac_te])


def build_all_optical(u_train, u_test, seed):
    stage1 = OptoelectronicReservoir(
        n_inputs=1, n_virtual_nodes=N_RESERVOIR, theta=THETA, seed=seed
    )
    stage2 = OptoelectronicReservoir(
        n_inputs=N_RESERVOIR, n_virtual_nodes=N_RESERVOIR, theta=THETA2, dt=0.02, seed=seed + 1
    )
    model = SerialPhysicalReservoir(stage1=stage1, stage2=stage2, combine="both", seed=seed)
    train_feats = model.run(u_train)
    test_feats = model.run(u_test, initial_state=model.last_state)
    return train_feats, test_feats


def build_simple_esn(u_train, u_test, seed):
    esn = ESN(
        n_inputs=1,
        n_reservoir=N_RESERVOIR,
        spectral_radius=1.2,
        sparsity=0.1,
        seed=seed,
    )
    train_feats = esn.run(u_train)
    test_feats = esn.run(u_test, initial_state=esn.last_state)
    return train_feats, test_feats


BUILDERS = {
    "serial_force": build_serial_force,
    "parallel": build_parallel,
    "all_optical": build_all_optical,
    "simple_esn": build_simple_esn,
}


# =====================================================================
# Evaluation
# =====================================================================

def evaluate(train_feats, y_train, test_feats, y_test, reg=REG):
    readout = RidgeReadout(reg=reg).fit(train_feats, y_train, washout=WASHOUT)
    pred = readout.predict(test_feats)
    pred_labels = (pred >= 0.5).astype(int)
    accuracy = float(np.mean(pred_labels == y_test))
    cond_number = None
    eff_rank = None
    eff_rank_frac = None
    try:
        Xw = train_feats[WASHOUT:]
        Xc = Xw - Xw.mean(axis=0, keepdims=True)
        s = np.linalg.svd(Xc, compute_uv=False)
        s = s[s > 1e-12]
        if len(s) > 0:
            cond_number = float(s[0] / s[-1])
            eff_rank = float((s.sum() ** 2) / (s**2).sum())
            eff_rank_frac = eff_rank / len(s)
    except Exception:
        pass
    return accuracy, cond_number, eff_rank, eff_rank_frac


# =====================================================================
# Main sweep
# =====================================================================

def main():
    print("Generating multi-band gated signal ...")
    u, y = generate_multiband_signal(seed=SEEDS[0])
    u_train, y_train, u_test, y_test = prepare_splits(u, y)
    print(f"Train: {u_train.shape[0]} samples, Test: {u_test.shape[0]} samples")
    print(f"Positive rate train: {y_train.mean():.3f}, test: {y_test.mean():.3f}\n")

    rows = []

    for seed in SEEDS:
        for arch_name, builder_fn in BUILDERS.items():
            train_feats, test_feats = builder_fn(u_train, u_test, seed)
            accuracy, cond, eff_rank, eff_rank_frac = evaluate(
                train_feats, y_train, test_feats, y_test
            )
            row = {
                "architecture": arch_name,
                "seed": seed,
                "accuracy": f"{accuracy:.4f}",
                "condition_number": f"{cond:.2e}" if cond is not None else "",
                "effective_rank": f"{eff_rank:.2f}" if eff_rank is not None else "",
                "effective_rank_fraction": f"{eff_rank_frac:.4f}" if eff_rank_frac is not None else "",
            }
            cond_str = f"{cond:.2e}" if cond is not None else "—"
            erf_str = f"{eff_rank_frac:.4f}" if eff_rank_frac is not None else "—"
            rows.append(row)
            print(f"  {arch_name:<14} seed={seed}  acc={accuracy:.4f}  cond={cond_str:<10}  eff_rank_frac={erf_str}")

    csv_path = f"{RESULTS_DIR}/exp18_real_world_eeg.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {csv_path}")

    # Aggregate by architecture
    acc_by_arch = {}
    cond_by_arch = {}
    erf_by_arch = {}
    for row in rows:
        a = row["architecture"]
        acc_by_arch.setdefault(a, []).append(float(row["accuracy"]))
        if row["condition_number"]:
            cond_by_arch.setdefault(a, []).append(float(row["condition_number"]))
        if row["effective_rank_fraction"]:
            erf_by_arch.setdefault(a, []).append(float(row["effective_rank_fraction"]))

    print("\nAggregate by architecture (mean ± 95% CI):")
    print(f"{'architecture':<14} {'accuracy':>10}  {'cond_number':>14}  {'eff_rank_frac':>13}")
    print("-" * 58)
    for arch in BUILDERS:
        accs = acc_by_arch[arch]
        acc_mean = np.mean(accs)
        acc_ci = stats.sem(accs) * stats.t.ppf((1 + 0.95) / 2, len(accs) - 1)
        cond_str = ""
        if arch in cond_by_arch:
            c_mean = np.mean(cond_by_arch[arch])
            cond_str = f"{c_mean:.2e}"
        erf_str = ""
        if arch in erf_by_arch:
            e_mean = np.mean(erf_by_arch[arch])
            erf_str = f"{e_mean:.4f}"
        print(f"{arch:<14} {acc_mean:.4f} ± {acc_ci:.4f}  {cond_str:>14}  {erf_str:>13}")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    colors = {
        "serial_force": "#eb6834",
        "parallel": "#2a78d6",
        "all_optical": "#1baf7a",
        "simple_esn": "#9b59b6",
    }
    labels = {
        "serial_force": "serial_force",
        "parallel": "parallel",
        "all_optical": "all_optical",
        "simple_esn": "simple_esn",
    }

    # Accuracy bars with 95% CI
    ax = axes[0]
    arch_names = list(BUILDERS.keys())
    means = [np.mean(acc_by_arch[a]) for a in arch_names]
    cis = [stats.sem(acc_by_arch[a]) * stats.t.ppf((1 + 0.95) / 2, len(acc_by_arch[a]) - 1) for a in arch_names]
    bars = ax.bar(arch_names, means, color=[colors[a] for a in arch_names], edgecolor="black", linewidth=0.6)
    ax.errorbar(arch_names, means, yerr=cis, fmt="none", color="black", capsize=4)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"EEG Eye-State Classification Accuracy\n(mean ± 95% CI, {len(SEEDS)} seeds)")
    ax.tick_params(axis="x", rotation=15)

    # Effective rank fraction comparison
    ax = axes[1]
    if erf_by_arch:
        arch_names_erf = [a for a in arch_names if a in erf_by_arch]
        erf_vals = [np.mean(erf_by_arch[a]) for a in arch_names_erf]
        ax.bar(arch_names_erf, erf_vals, color=[colors[a] for a in arch_names_erf], edgecolor="black", linewidth=0.6)
        ax.set_ylabel("Effective rank fraction")
        ax.set_title("Feature Matrix Effective Rank Fraction\n(higher = less correlated columns)")
        ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig_path = f"{VISUALS_DIR}/exp18_real_world_eeg.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"Saved plot to {fig_path}")

    # Statistical tests
    print("\nPaired t-tests (serial_force vs each other architecture):")
    sf_accs = acc_by_arch["serial_force"]
    for arch in ["parallel", "all_optical", "simple_esn"]:
        other_accs = acc_by_arch[arch]
        t_stat, p_val = stats.ttest_rel(sf_accs, other_accs)
        direction = "serial_force >" if np.mean(sf_accs) > np.mean(other_accs) else "serial_force <"
        print(f"  serial_force vs {arch:<14}: {direction}  p={p_val:.4f}")


if __name__ == "__main__":
    main()