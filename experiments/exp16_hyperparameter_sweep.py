"""
Hyperparameter-optimized architecture comparison across task difficulty.

All previous experiments (exp10-15) used untuned default hyperparameters.
This experiment sweeps key hyperparameters for the top 3 architectures to
determine:

1. Does serial_force win with optimized hyperparameters, or did defaults
   accidentally favor it?
2. What hyperparameter regions maximize each architecture's performance?
3. Does the coupling advantage hold on harder tasks (Mackey-Glass, NARMA10)?
4. Does the advantage scale with reservoir size?

Architectures tested:
  - serial_force (optical→acoustic, both taps) - exp10-13's winner
  - parallel (uncoupled baseline)
  - all_optical_dual_delay (single-substrate baseline)

Hyperparameters swept:
  - optical_theta: [0.2, 1.0, 5.0] (response timescale)
  - acoustic_c: [0.3, 1.0, 3.0] (damping/memory timescale)
  - n_virtual/n_oscillators: [100, 300, 900] (reservoir size scaling)
  - readout_reg: [1e-6, 1e-3, 1e-1] (regularization strength)

Tasks:
  - dual_timescale (interaction=1.0, the task where coupling helped most)
  - mackey_glass (chaotic, long memory)
  - narma10 (nonlinear, ~10-step memory)

Statistical design:
  - 10 seeds per configuration (not 4)
  - Reports mean ± 95% confidence interval
"""

import argparse
import csv
import itertools

import numpy as np
from scipy import stats

from reservoir_lab.physical import (
    AcousticReservoir,
    OptoelectronicReservoir,
    SerialPhysicalReservoir,
)
from reservoir_lab.readout import RidgeReadout

RESULTS_DIR = "experiments/data_results"
VISUALS_DIR = "experiments/visuals"

# Hyperparameter grid
OPTICAL_THETAS = [0.2, 1.0, 5.0]
ACOUSTIC_CS = [0.3, 1.0, 3.0]
RESERVOIR_SIZES = [100, 300, 900]
READOUT_REGS = [1e-6, 1e-3, 1e-1]
N_SEEDS = 10
SEEDS = list(range(N_SEEDS))


# =====================================================================
# Task generators (reused from exp10, exp13, exp09)
# =====================================================================

def dual_timescale_task(n_steps=1400, interaction=1.0, noise_std=0.35, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)
    slow = 0.6 * np.sin(2 * np.pi * t / 180.0)
    fast_raw = 0.3 * np.sin(2 * np.pi * t / 12.0 + 0.6 * np.sin(2 * np.pi * t / 33.0))
    noise = rng.normal(0, noise_std, n_steps)
    gate = 0.5 * (1 + np.tanh(4 * slow))
    fast_target = fast_raw * ((1 - interaction) + interaction * gate)
    u = (slow + fast_raw + noise).reshape(-1, 1)
    y = np.stack([slow, fast_target], axis=1)
    split = int(n_steps * 0.7)
    return u[:split], y[:split], u[split:], y[split:]


def mackey_glass_task(length=6000, train_size=4000, tau=17, seed=42):
    beta, gamma, n = 0.2, 0.1, 10
    rng = np.random.default_rng(seed)
    data = np.ones(length + tau + 1) * 1.2
    for t in range(tau, length + tau):
        delayed = data[t - tau]
        data[t + 1] = data[t] + beta * delayed / (1 + delayed**n) - gamma * data[t]
    data = data[tau:].reshape(-1, 1)
    train, test = data[:train_size], data[train_size:]
    return train[:-1], train[1:], test[:-1], test[1:]


def narma10_task(length=5000, train_size=4000, seed=42):
    rng = np.random.default_rng(seed)
    u = rng.random((length, 1)) * 0.5
    y = np.zeros((length, 1))
    for t in range(10, length - 1):
        y[t + 1] = (
            0.3 * y[t]
            + 0.05 * y[t] * np.sum(y[t - 10:t])
            + 1.5 * u[t - 9] * u[t]
            + 0.1
        )
    train_inputs, train_targets = u[:train_size], y[:train_size]
    test_inputs, test_targets = u[train_size:], y[train_size:]
    return train_inputs, train_targets, test_inputs, test_targets


TASKS = {
    "dual_timescale": lambda seed: dual_timescale_task(seed=seed),
    "mackey_glass": lambda seed: mackey_glass_task(seed=seed),
    "narma10": lambda seed: narma10_task(seed=seed),
}


# =====================================================================
# Architecture builders
# =====================================================================

def build_serial_force(u_train, u_test, seed, theta, c, n_reservoir, reg):
    optical = OptoelectronicReservoir(
        n_inputs=1, n_virtual_nodes=n_reservoir, theta=theta, seed=seed
    )
    acoustic = AcousticReservoir(
        n_inputs=n_reservoir, n_oscillators=n_reservoir, c=c, seed=seed + 1
    )
    model = SerialPhysicalReservoir(stage1=optical, stage2=acoustic, combine="both", seed=seed)
    train_feats = model.run(u_train)
    test_feats = model.run(u_test, initial_state=model.last_state)
    readout = RidgeReadout(reg=reg).fit(train_feats, np.zeros((train_feats.shape[0], 1)), washout=50)
    return train_feats, test_feats


def build_parallel(u_train, u_test, seed, theta, c, n_reservoir, reg):
    optical = OptoelectronicReservoir(
        n_inputs=1, n_virtual_nodes=n_reservoir, theta=theta, seed=seed
    )
    acoustic = AcousticReservoir(
        n_inputs=1, n_oscillators=n_reservoir, c=c, seed=seed + 1
    )
    opt_tr = optical.run(u_train)
    ac_tr = acoustic.run(u_train)
    opt_te = optical.run(u_test, initial_state=optical.last_state)
    ac_te = acoustic.run(u_test, initial_state=acoustic.last_state)
    return np.hstack([opt_tr, ac_tr]), np.hstack([opt_te, ac_te])


def build_all_optical(u_train, u_test, seed, theta, c, n_reservoir, reg):
    stage1 = OptoelectronicReservoir(
        n_inputs=1, n_virtual_nodes=n_reservoir, theta=theta, seed=seed
    )
    stage2 = OptoelectronicReservoir(
        n_inputs=n_reservoir, n_virtual_nodes=n_reservoir, theta=c * 10, seed=seed + 1
    )
    model = SerialPhysicalReservoir(stage1=stage1, stage2=stage2, combine="both", seed=seed)
    train_feats = model.run(u_train)
    test_feats = model.run(u_test, initial_state=model.last_state)
    return train_feats, test_feats


ARCHITECTURES = {
    "serial_force": build_serial_force,
    "parallel": build_parallel,
    "all_optical": build_all_optical,
}


# =====================================================================
# Evaluation with NRMSE metric
# =====================================================================

def nrmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2, axis=0)) / np.std(true, axis=0)


def evaluate_config(builder, u_train, y_train, u_test, y_test, seed, theta, c, n_res, reg):
    train_feats, test_feats = builder(u_train, u_test, seed, theta, c, n_res, reg)
    readout = RidgeReadout(reg=reg).fit(train_feats, y_train, washout=100)
    pred = readout.predict(test_feats)
    return nrmse(pred, y_test)


# =====================================================================
# Main sweep
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for reservoir computing architectures")
    parser.add_argument("--quick", action="store_true", help="Quick mode: fewer seeds and hyperparameters (~15-20 min)")
    args = parser.parse_args()

    if args.quick:
        # Quick mode: reduced grid for fast exploration
        optical_thetas = [0.2, 1.0]
        acoustic_cs = [0.3, 1.0]
        reservoir_sizes = [100, 300]
        readout_regs = [1e-6, 1e-3]
        n_seeds = 5
        seeds_list = list(range(n_seeds))
        mode_name = "QUICK"
    else:
        # Full mode: complete grid
        optical_thetas = OPTICAL_THETAS
        acoustic_cs = ACOUSTIC_CS
        reservoir_sizes = RESERVOIR_SIZES
        readout_regs = READOUT_REGS
        n_seeds = N_SEEDS
        seeds_list = SEEDS
        mode_name = "FULL"

    param_combos = list(itertools.product(
        optical_thetas, acoustic_cs, reservoir_sizes, readout_regs
    ))
    total_runs = len(param_combos) * len(ARCHITECTURES) * len(TASKS) * n_seeds
    print(f"Starting hyperparameter sweep [{mode_name} mode]: {total_runs} total configurations")
    print(f"  Architectures: {len(ARCHITECTURES)}")
    print(f"  Tasks: {len(TASKS)}")
    print(f"  Parameter combos: {len(param_combos)}")
    print(f"  Seeds: {n_seeds}")
    print()

    results = []
    config_id = 0

    for theta, c, n_res, reg in param_combos:
        for arch_name, builder in ARCHITECTURES.items():
            for task_name, task_fn in TASKS.items():
                seed_results = []
                for seed in seeds_list:
                    u_train, y_train, u_test, y_test = task_fn(seed=seed)
                    try:
                        nrmse_vals = evaluate_config(
                            builder, u_train, y_train, u_test, y_test,
                            seed, theta, c, n_res, reg
                        )
                        # Use slow component NRMSE as primary metric
                        seed_results.append(nrmse_vals[0])
                    except (np.linalg.LinAlgError, ValueError) as e:
                        seed_results.append(None)
                
                # Compute statistics
                valid_results = [r for r in seed_results if r is not None]
                if valid_results:
                    mean_val = np.mean(valid_results)
                    std_val = np.std(valid_results)
                    sem_val = std_val / np.sqrt(len(valid_results))
                    ci_95 = stats.t.interval(0.95, len(valid_results) - 1, loc=mean_val, scale=sem_val)
                    ci_width = (ci_95[1] - ci_95[0]) / 2
                else:
                    mean_val = std_val = ci_width = np.nan

                results.append({
                    "config_id": config_id,
                    "architecture": arch_name,
                    "task": task_name,
                    "optical_theta": theta,
                    "acoustic_c": c,
                    "n_reservoir": n_res,
                    "readout_reg": reg,
                    "nrmse_mean": f"{mean_val:.4f}" if not np.isnan(mean_val) else "FAILED",
                    "nrmse_std": f"{std_val:.4f}" if not np.isnan(std_val) else "FAILED",
                    "ci_95": f"{ci_width:.4f}" if not np.isnan(ci_width) else "FAILED",
                    "n_successful": len(valid_results),
                    "n_total": n_seeds,
                })
                
                config_id += 1

                if config_id % 20 == 0:
                    print(f"  Progress: {config_id}/{len(param_combos) * len(ARCHITECTURES) * len(TASKS)} configs")

    # Export full results
    fieldnames = [
        "config_id", "architecture", "task", "optical_theta", "acoustic_c",
        "n_reservoir", "readout_reg", "nrmse_mean", "nrmse_std", "ci_95",
        "n_successful", "n_total"
    ]
    with open(f"{RESULTS_DIR}/exp16_hyperparameter_sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {RESULTS_DIR}/exp16_hyperparameter_sweep.csv")
    print(f"Total configurations tested: {config_id}")

    # Summary by architecture and task
    print("\n" + "=" * 80)
    print("SUMMARY: Best configuration per architecture × task")
    print("=" * 80)
    for task_name in TASKS:
        print(f"\nTask: {task_name}")
        print(f"{'architecture':<15} {'best_nrmse':<12} {'best_params':<40} {'95% CI'}")
        print("-" * 80)
        for arch_name in ARCHITECTURES:
            task_results = [
                r for r in results
                if r["architecture"] == arch_name and r["task"] == task_name and r["nrmse_mean"] != "FAILED"
            ]
            if task_results:
                best = min(task_results, key=lambda r: float(r["nrmse_mean"]))
                params = f"θ={best['optical_theta']}, c={best['acoustic_c']}, n={best['n_reservoir']}, reg={best['readout_reg']}"
                print(f"{arch_name:<15} {best['nrmse_mean']:<12} {params:<40} ±{best['ci_95']}")
            else:
                print(f"{arch_name:<15} {'FAILED':<12}")


if __name__ == "__main__":
    main()