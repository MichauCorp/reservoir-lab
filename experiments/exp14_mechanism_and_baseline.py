"""
Three follow-ups to exp13's headline finding (force-coupling beat
parallel, parametric, and mutual coupling specifically when the task
needs cross-timescale interaction and timescales are separated):

1. MECHANISM: is the advantage really because force-coupling injects
   gate-relevant information into the acoustic state itself, decodable
   by a plain linear map -- rather than an artifact of readout capacity
   or luck? Fit a small linear decoder from acoustic states ALONE (no
   optical, no full-task readout) to the true gate signal, for both
   acoustic driven directly by raw input and acoustic force-driven by
   optical's output, and compare how well each recovers it.

2. GENERALIZATION: does the advantage hold on a second, structurally
   different interaction task? exp13 used a smooth tanh-based amplitude
   gate. This adds a pulse-gated task where the fast component only
   appears in narrow windows around the slow signal's peaks -- sparse
   and near-discontinuous, not just a rescaled version of the same
   interaction.

3. BASELINE: does mixing substrates (optical + acoustic) actually beat
   an all-optical alternative built the exact same way -- two optical
   stages with different theta (timescale), one force-driving the
   other -- instead of `parallel`, which was never the real competitor?
"""

import csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reservoir_lab.physical import (
    AcousticReservoir,
    OptoelectronicReservoir,
    SerialPhysicalReservoir,
)
from reservoir_lab.readout import RidgeReadout

RESULTS_DIR = "experiments/data_results"
VISUALS_DIR = "experiments/visuals"

N_STEPS = 1400
WASHOUT = 120
TRAIN_FRAC = 0.7
N_VIRTUAL = 90
N_OSCILLATORS = 90
SEEDS = [1, 2, 3, 4, 5, 6]
SEPARATED = {"optical_theta": 0.2, "acoustic_c": 0.3}


# =====================================================================
# Task families
# =====================================================================

def _base_signal(n_steps, seed, noise_std):
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)
    slow = 0.6 * np.sin(2 * np.pi * t / 180.0)
    fast_raw = 0.3 * np.sin(2 * np.pi * t / 12.0 + 0.6 * np.sin(2 * np.pi * t / 33.0))
    noise = rng.normal(0, noise_std, n_steps)
    return t, slow, fast_raw, noise


def tanh_gate_task(n_steps, seed=0, noise_std=0.35):
    """Smooth amplitude gate: fast component scaled continuously by
    tanh(slow) -- exp13's task."""
    _, slow, fast_raw, noise = _base_signal(n_steps, seed, noise_std)
    gate = 0.5 * (1 + np.tanh(4 * slow))
    fast_target = fast_raw * gate
    u = (slow + fast_raw + noise).reshape(-1, 1)
    y = np.stack([slow, fast_target], axis=1)
    split = int(n_steps * TRAIN_FRAC)
    return u[:split], y[:split], u[split:], y[split:], gate


def pulse_gate_task(n_steps, seed=0, noise_std=0.35, pulse_width=8):
    """Sparse pulse gate: fast component only appears in narrow windows
    around slow's peaks -- structurally different (near-discontinuous,
    mostly zero) from the smooth tanh gate above."""
    _, slow, fast_raw, noise = _base_signal(n_steps, seed, noise_std)

    dslow = np.diff(slow, prepend=slow[0])
    is_peak = (dslow[:-1] > 0) & (dslow[1:] <= 0)
    peak_idx = np.where(is_peak)[0] + 1

    gate = np.zeros(n_steps)
    for p in peak_idx:
        lo, hi = max(0, p - pulse_width // 2), min(n_steps, p + pulse_width // 2)
        gate[lo:hi] = 1.0

    fast_target = fast_raw * gate
    u = (slow + fast_raw + noise).reshape(-1, 1)
    y = np.stack([slow, fast_target], axis=1)
    split = int(n_steps * TRAIN_FRAC)
    return u[:split], y[:split], u[split:], y[split:], gate


TASKS = {"tanh_gate": tanh_gate_task, "pulse_gate": pulse_gate_task}


def nrmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2, axis=0)) / np.std(true, axis=0)


# =====================================================================
# 1. Mechanism: linear decodability of the gate from acoustic states alone
# =====================================================================

def acoustic_direct_states(u_train, u_test, seed):
    ac = AcousticReservoir(n_inputs=1, n_oscillators=N_OSCILLATORS, c=SEPARATED["acoustic_c"], seed=seed + 1)
    tr = ac.run(u_train)
    te = ac.run(u_test, initial_state=ac.last_state)
    return tr, te


def acoustic_force_coupled_states(u_train, u_test, seed):
    optical = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=N_VIRTUAL, theta=SEPARATED["optical_theta"], seed=seed)
    acoustic = AcousticReservoir(n_inputs=N_VIRTUAL, n_oscillators=N_OSCILLATORS, c=SEPARATED["acoustic_c"], seed=seed + 1)
    opt_tr = optical.run(u_train)
    ac_tr = acoustic.run(opt_tr)
    opt_te = optical.run(u_test, initial_state=optical.last_state)
    ac_te = acoustic.run(opt_te, initial_state=acoustic.last_state)
    return ac_tr, ac_te


def decode_gate(states_train, gate_train, states_test, gate_test):
    readout = RidgeReadout(reg=1e-3).fit(states_train, gate_train.reshape(-1, 1), washout=WASHOUT)
    pred = readout.predict(states_test)
    return nrmse(pred, gate_test.reshape(-1, 1))[0], pred


def mechanism_analysis():
    print("=" * 74)
    print("1. MECHANISM: is the gate signal linearly decodable from")
    print("   acoustic states ALONE (no optical, no full-task readout)?")
    print("=" * 74)

    seed = 1
    u_train, _y_train, u_test, _y_test, gate = tanh_gate_task(N_STEPS, seed=seed)
    split = int(N_STEPS * TRAIN_FRAC)
    gate_train, gate_test = gate[:split], gate[split:]

    direct_tr, direct_te = acoustic_direct_states(u_train, u_test, seed)
    coupled_tr, coupled_te = acoustic_force_coupled_states(u_train, u_test, seed)

    nrmse_direct, pred_direct = decode_gate(direct_tr, gate_train, direct_te, gate_test)
    nrmse_coupled, pred_coupled = decode_gate(coupled_tr, gate_train, coupled_te, gate_test)

    print(f"gate decodability NRMSE, acoustic_direct:        {nrmse_direct:.4f}")
    print(f"gate decodability NRMSE, acoustic_force_coupled: {nrmse_coupled:.4f}")
    print("(lower = the gate is more linearly recoverable from that reservoir's state alone)\n")

    fig, ax = plt.subplots(figsize=(9, 3.5))
    window = slice(0, 300)
    ax.plot(gate_test[window], label="true gate", color="black", linewidth=1.5)
    ax.plot(pred_direct[window], label=f"decoded from acoustic_direct (NRMSE={nrmse_direct:.3f})", alpha=0.8)
    ax.plot(pred_coupled[window], label=f"decoded from acoustic_force_coupled (NRMSE={nrmse_coupled:.3f})", alpha=0.8)
    ax.set_title("Linear decodability of the true gate signal from acoustic state alone")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{VISUALS_DIR}/exp14_gate_decodability.png", dpi=130)
    plt.close(fig)
    print(f"Saved plot to {VISUALS_DIR}/exp14_gate_decodability.png\n")


# =====================================================================
# 2 & 3. Generalization across tasks + all-optical baseline
# =====================================================================

def build_parallel(u_train, u_test, seed):
    optical = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=N_VIRTUAL, theta=SEPARATED["optical_theta"], seed=seed)
    acoustic = AcousticReservoir(n_inputs=1, n_oscillators=N_OSCILLATORS, c=SEPARATED["acoustic_c"], seed=seed + 1)
    opt_tr, ac_tr = optical.run(u_train), acoustic.run(u_train)
    opt_te = optical.run(u_test, initial_state=optical.last_state)
    ac_te = acoustic.run(u_test, initial_state=acoustic.last_state)
    return np.hstack([opt_tr, ac_tr]), np.hstack([opt_te, ac_te])


def build_serial_force_optical_acoustic(u_train, u_test, seed):
    stage1 = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=N_VIRTUAL, theta=SEPARATED["optical_theta"], seed=seed)
    stage2 = AcousticReservoir(n_inputs=N_VIRTUAL, n_oscillators=N_OSCILLATORS, c=SEPARATED["acoustic_c"], seed=seed + 1)
    model = SerialPhysicalReservoir(stage1, stage2, combine="both", seed=seed)
    tr = model.run(u_train)
    te = model.run(u_test, initial_state=model.last_state)
    return tr, te


def build_all_optical_dual_delay(u_train, u_test, seed, theta2=4.0, dt2=None):
    """Same pattern as serial_force (force-coupled, both taps read out),
    but BOTH stages are OptoelectronicReservoir -- one fast (theta
    matching the optical stage elsewhere), one deliberately slow (theta
    much larger) -- instead of mixing optical + acoustic substrates.
    This is the real competitive baseline: same architecture, single
    substrate.

    dt2 defaults to None, which lets OptoelectronicReservoir fall back to
    theta2/10 -- but that couples the delay window size (tau = n_nodes *
    dt) to theta2, confounding "slower response" with "longer delay
    window". tune_all_optical_theta2() below fixes dt2 explicitly so only
    the response-time parameter varies.
    """
    stage1 = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=N_VIRTUAL, theta=SEPARATED["optical_theta"], seed=seed)
    stage2 = OptoelectronicReservoir(
        n_inputs=N_VIRTUAL, n_virtual_nodes=N_OSCILLATORS, theta=theta2, dt=dt2, seed=seed + 1
    )
    model = SerialPhysicalReservoir(stage1, stage2, combine="both", seed=seed)
    tr = model.run(u_train)
    te = model.run(u_test, initial_state=model.last_state)
    return tr, te


def tune_all_optical_theta2():
    """Sweep stage2's theta with dt fixed (decoupled from theta, so only
    response-time -- not delay-window size -- changes), on the tanh_gate
    task where the coupling advantage was found. 3 seeds for speed; the
    winner gets re-evaluated with the full seed set in main()."""
    print("=" * 74)
    print("TUNING: all-optical dual-delay stage2 theta (dt fixed at 0.02,")
    print("        decoupling response-time from delay-window size)")
    print("=" * 74)

    tune_seeds = [1, 2, 3]
    candidates = [0.3, 0.5, 0.8, 1.2, 2.0, 3.0, 4.0, 6.0]
    results = {}

    print(f"{'theta2':<10}{'fast NRMSE (mean +- std)'}")
    print("-" * 40)
    for theta2 in candidates:
        cell = []
        for seed in tune_seeds:
            u_train, y_train, u_test, y_test, _ = tanh_gate_task(N_STEPS, seed=seed)
            result = evaluate(
                lambda utr, ute, sd, t2=theta2: build_all_optical_dual_delay(utr, ute, sd, theta2=t2, dt2=0.02),
                u_train, y_train, u_test, y_test, seed,
            )
            cell.append(result[1])
        results[theta2] = (np.mean(cell), np.std(cell))
        print(f"{theta2:<10}{results[theta2][0]:.4f} +- {results[theta2][1]:.4f}")

    best_theta2 = min(results, key=lambda k: results[k][0])
    print(f"\nBest theta2 = {best_theta2} (fast NRMSE {results[best_theta2][0]:.4f})\n")
    return best_theta2


def build_arch_dict(tuned_theta2):
    return {
        "parallel": build_parallel,
        "serial_force (optical+acoustic)": build_serial_force_optical_acoustic,
        "all_optical_dual_delay (tuned)": lambda utr, ute, sd: build_all_optical_dual_delay(
            utr, ute, sd, theta2=tuned_theta2, dt2=0.02
        ),
    }


def evaluate(builder, u_train, y_train, u_test, y_test, seed):
    train_feats, test_feats = builder(u_train, u_test, seed)
    readout = RidgeReadout(reg=1e-3).fit(train_feats, y_train, washout=WASHOUT)
    pred = readout.predict(test_feats)
    return nrmse(pred, y_test)


def generalization_and_baseline(tuned_theta2):
    print("=" * 74)
    print("2 & 3. GENERALIZATION across task families + tuned all-optical baseline")
    print("=" * 74)
    print(f"{'task':<14}{'architecture':<34}{'fast NRMSE (mean +- std)'}")
    print("-" * 78)

    archs = build_arch_dict(tuned_theta2)
    for task_name, task_fn in TASKS.items():
        for arch_name, builder in archs.items():
            cell = []
            for seed in SEEDS:
                u_train, y_train, u_test, y_test, _ = task_fn(N_STEPS, seed=seed)
                try:
                    result = evaluate(builder, u_train, y_train, u_test, y_test, seed)
                    cell.append(result[1])
                except (np.linalg.LinAlgError, ValueError) as e:
                    print(f"  FAILED: {task_name}, {arch_name}, seed={seed}: {e}")
            if cell:
                print(f"{task_name:<14}{arch_name:<34}{np.mean(cell):.4f} +- {np.std(cell):.4f}")
        print()


def export_mechanism_results(nrmse_direct, nrmse_coupled):
    """Export mechanism analysis results to CSV."""
    with open(f"{RESULTS_DIR}/exp14_mechanism_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "nrmse"])
        writer.writerow(["acoustic_direct", f"{nrmse_direct:.6f}"])
        writer.writerow(["acoustic_force_coupled", f"{nrmse_coupled:.6f}"])
    print(f"Results saved to {RESULTS_DIR}/exp14_mechanism_results.csv\n")


def export_generalization_results(tuned_theta2):
    """Export generalization and baseline results to CSV."""
    archs = build_arch_dict(tuned_theta2)
    all_results = []
    for task_name, task_fn in TASKS.items():
        for arch_name, builder in archs.items():
            cell = []
            for seed in SEEDS:
                u_train, y_train, u_test, y_test, _ = task_fn(N_STEPS, seed=seed)
                try:
                    result = evaluate(builder, u_train, y_train, u_test, y_test, seed)
                    cell.append(result[1])
                except (np.linalg.LinAlgError, ValueError) as e:
                    print(f"  FAILED: {task_name}, {arch_name}, seed={seed}: {e}")
            if cell:
                all_results.append({
                    "task": task_name,
                    "architecture": arch_name,
                    "fast_nrmse_mean": f"{np.mean(cell):.4f}",
                    "fast_nrmse_std": f"{np.std(cell):.4f}",
                })
    
    with open(f"{RESULTS_DIR}/exp14_generalization_baseline.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "architecture", "fast_nrmse_mean", "fast_nrmse_std"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Results saved to {RESULTS_DIR}/exp14_generalization_baseline.csv\n")


if __name__ == "__main__":
    mechanism_analysis()
    tuned_theta2 = tune_all_optical_theta2()
    export_generalization_results(tuned_theta2)
    generalization_and_baseline(tuned_theta2)
