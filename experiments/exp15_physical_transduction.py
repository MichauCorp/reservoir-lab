"""
Models a physically bandwidth-limited transducer on the optical -> acoustic
coupling path, informed by real device numbers rather than the idealized
instantaneous coupling used in exp10-14:

  - real optoelectronic reservoirs operate at 12-60 GHz (sub-nanosecond
    per virtual node)
  - a real piezo-optomechanical transducer achieves ~25 MHz bandwidth
    (Nature Communications, 2024)
  - practical mechanical/acoustic reservoir memory reaches millisecond-
    scale delays

With ~N_VIRTUAL virtual nodes per outer timestep, those numbers put the
transducer's own time constant at roughly 5-27 OUTER TIMESTEPS, not
sub-timestep -- comparable to or LONGER than the fast signal component's
own period (12 timesteps) in our task. A realistic transducer's bandwidth
limit alone, before acoustic's own dynamics do anything, is already in the
range that could smear out the fast content the whole mechanism depends on.

Critically, this penalty is asymmetric: it applies to CROSS-DOMAIN
(optical -> acoustic) coupling, which requires a lossy energy-domain
conversion (light -> mechanical motion). It does NOT apply the same way to
the all_optical_dual_delay baseline's own optical -> optical coupling,
which can stay in one domain (e.g. a beamsplitter or waveguide coupler) at
the reservoir's native bandwidth. exp14 compared an idealized hybrid
against an idealized all-optical baseline; this experiment applies the
transducer penalty only where a real device would actually need one, and
checks whether that asymmetry changes which architecture wins.
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
SEEDS = [1, 2, 3, 4]
SEPARATED = {"optical_theta": 0.2, "acoustic_c": 0.3}

# Transducer time constants to sweep, in units of OUTER timesteps.
# 0        = idealized instantaneous coupling (exp10-14's assumption)
# 5-27     = realistic range, from the real-device numbers above
# 50, 100  = pessimistic / worst-case
TAU_VALUES = [0, 2, 5, 10, 15, 27, 50, 100]


def tanh_gate_task(n_steps, seed=0, noise_std=0.35):
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)
    slow = 0.6 * np.sin(2 * np.pi * t / 180.0)
    fast_raw = 0.3 * np.sin(2 * np.pi * t / 12.0 + 0.6 * np.sin(2 * np.pi * t / 33.0))
    noise = rng.normal(0, noise_std, n_steps)
    gate = 0.5 * (1 + np.tanh(4 * slow))
    fast_target = fast_raw * gate
    u = (slow + fast_raw + noise).reshape(-1, 1)
    y = np.stack([slow, fast_target], axis=1)
    split = int(n_steps * TRAIN_FRAC)
    return u[:split], y[:split], u[split:], y[split:], gate


def nrmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2, axis=0)) / np.std(true, axis=0)


# =====================================================================
# The transducer: a single-pole low-pass filter across the OUTER-
# timestep axis, applied independently to each of optical's virtual-node
# channels, with state carried across the train/test boundary the same
# way every other stage's `last_state` is.
# =====================================================================

def transduce(states, tau_timesteps, initial=None):
    """states: (T, J). tau_timesteps <= 0 returns states unchanged
    (idealized, instantaneous coupling)."""
    if tau_timesteps <= 1e-9:
        return states.copy(), states[-1].copy()

    alpha = 1.0 - np.exp(-1.0 / tau_timesteps)
    out = np.zeros_like(states)
    prev = states[0].copy() if initial is None else np.array(initial, dtype=float)
    for t in range(len(states)):
        prev = prev + alpha * (states[t] - prev)
        out[t] = prev
    return out, prev


# =====================================================================
# 1. Mechanism, revisited: does gate-decodability survive transduction?
# =====================================================================

def acoustic_direct_states(u_train, u_test, seed):
    ac = AcousticReservoir(n_inputs=1, n_oscillators=N_OSCILLATORS, c=SEPARATED["acoustic_c"], seed=seed + 1)
    tr = ac.run(u_train)
    te = ac.run(u_test, initial_state=ac.last_state)
    return tr, te


def acoustic_force_coupled_states(u_train, u_test, seed, tau_timesteps):
    optical = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=N_VIRTUAL, theta=SEPARATED["optical_theta"], seed=seed)
    acoustic = AcousticReservoir(n_inputs=N_VIRTUAL, n_oscillators=N_OSCILLATORS, c=SEPARATED["acoustic_c"], seed=seed + 1)

    opt_tr = optical.run(u_train)
    opt_tr_transduced, filt_state = transduce(opt_tr, tau_timesteps)
    ac_tr = acoustic.run(opt_tr_transduced)

    opt_te = optical.run(u_test, initial_state=optical.last_state)
    opt_te_transduced, _ = transduce(opt_te, tau_timesteps, initial=filt_state)
    ac_te = acoustic.run(opt_te_transduced, initial_state=acoustic.last_state)

    return ac_tr, ac_te


def decode_gate(states_train, gate_train, states_test, gate_test):
    readout = RidgeReadout(reg=1e-3).fit(states_train, gate_train.reshape(-1, 1), washout=WASHOUT)
    pred = readout.predict(states_test)
    return nrmse(pred, gate_test.reshape(-1, 1))[0]


def mechanism_vs_transduction():
    print("=" * 74)
    print("1. MECHANISM vs TRANSDUCTION BANDWIDTH: does gate-decodability")
    print("   survive a physically realistic transducer time constant?")
    print("=" * 74)

    seed = 1
    u_train, _, u_test, _, gate = tanh_gate_task(N_STEPS, seed=seed)
    split = int(N_STEPS * TRAIN_FRAC)
    gate_train, gate_test = gate[:split], gate[split:]

    direct_tr, direct_te = acoustic_direct_states(u_train, u_test, seed)
    nrmse_direct = decode_gate(direct_tr, gate_train, direct_te, gate_test)

    print(f"{'tau (outer timesteps)':<26}{'gate decodability NRMSE'}")
    print("-" * 52)
    print(f"{'acoustic_direct (n/a)':<26}{nrmse_direct:.4f}")

    results = {}
    for tau in TAU_VALUES:
        coupled_tr, coupled_te = acoustic_force_coupled_states(u_train, u_test, seed, tau)
        nrmse_coupled = decode_gate(coupled_tr, gate_train, coupled_te, gate_test)
        results[tau] = nrmse_coupled
        label = f"tau={tau}" + (" (idealized)" if tau == 0 else "")
        print(f"{label:<26}{nrmse_coupled:.4f}")
    print()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(nrmse_direct, color="gray", linestyle="--", label="acoustic_direct (no transduction needed)")
    ax.plot(list(results.keys()), list(results.values()), marker="o", color="#eb6834", label="acoustic_force_coupled")
    ax.axvspan(5, 27, alpha=0.15, color="orange", label="realistic range (real device numbers)")
    ax.set_xlabel("transducer time constant (outer timesteps)")
    ax.set_ylabel("gate decodability NRMSE (lower = better)")
    ax.set_title("Does force-coupling's information advantage survive realistic transduction bandwidth?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{VISUALS_DIR}/exp15_transduction_mechanism.png", dpi=130)
    plt.close(fig)
    print(f"Saved plot to {VISUALS_DIR}/exp15_transduction_mechanism.png\n")

    # Export mechanism results
    with open(f"{RESULTS_DIR}/exp15_mechanism_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tau_timesteps", "nrmse_gate_decodability"])
        writer.writerow(["acoustic_direct", f"{nrmse_direct:.6f}"])
        for tau, nrmse_val in results.items():
            writer.writerow([tau, f"{nrmse_val:.6f}"])
    print(f"Results saved to {RESULTS_DIR}/exp15_mechanism_results.csv\n")


# =====================================================================
# 2. Full task comparison across the same tau sweep, all_optical_dual_delay
#    left unpenalized (same-domain coupling doesn't need a lossy transducer)
# =====================================================================

def build_parallel(u_train, u_test, seed):
    optical = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=N_VIRTUAL, theta=SEPARATED["optical_theta"], seed=seed)
    acoustic = AcousticReservoir(n_inputs=1, n_oscillators=N_OSCILLATORS, c=SEPARATED["acoustic_c"], seed=seed + 1)
    opt_tr, ac_tr = optical.run(u_train), acoustic.run(u_train)
    opt_te = optical.run(u_test, initial_state=optical.last_state)
    ac_te = acoustic.run(u_test, initial_state=acoustic.last_state)
    return np.hstack([opt_tr, ac_tr]), np.hstack([opt_te, ac_te])


def build_serial_force_transduced(u_train, u_test, seed, tau_timesteps):
    optical = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=N_VIRTUAL, theta=SEPARATED["optical_theta"], seed=seed)
    acoustic = AcousticReservoir(n_inputs=N_VIRTUAL, n_oscillators=N_OSCILLATORS, c=SEPARATED["acoustic_c"], seed=seed + 1)

    opt_tr = optical.run(u_train)
    opt_tr_transduced, filt_state = transduce(opt_tr, tau_timesteps)
    ac_tr = acoustic.run(opt_tr_transduced)

    opt_te = optical.run(u_test, initial_state=optical.last_state)
    opt_te_transduced, _ = transduce(opt_te, tau_timesteps, initial=filt_state)
    ac_te = acoustic.run(opt_te_transduced, initial_state=acoustic.last_state)

    return np.hstack([opt_tr, ac_tr]), np.hstack([opt_te, ac_te])


def build_all_optical_dual_delay(u_train, u_test, seed, theta2=3.0, dt2=0.02):
    """Unpenalized: optical -> optical stays in one domain, no lossy
    transducer required in a real device."""
    stage1 = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=N_VIRTUAL, theta=SEPARATED["optical_theta"], seed=seed)
    stage2 = OptoelectronicReservoir(n_inputs=N_VIRTUAL, n_virtual_nodes=N_OSCILLATORS, theta=theta2, dt=dt2, seed=seed + 1)
    model = SerialPhysicalReservoir(stage1, stage2, combine="both", seed=seed)
    tr = model.run(u_train)
    te = model.run(u_test, initial_state=model.last_state)
    return tr, te


def evaluate(builder, u_train, y_train, u_test, y_test):
    train_feats, test_feats = builder(u_train, u_test)
    readout = RidgeReadout(reg=1e-3).fit(train_feats, y_train, washout=WASHOUT)
    pred = readout.predict(test_feats)
    return nrmse(pred, y_test)


def full_comparison_across_tau():
    print("=" * 74)
    print("2. FULL TASK COMPARISON across the same transducer sweep")
    print("   (all_optical_dual_delay is NOT penalized -- same-domain coupling)")
    print("=" * 74)
    print(f"{'tau':<8}{'parallel':<24}{'serial_force (transduced)':<28}{'all_optical_dual_delay'}")
    print("-" * 88)

    parallel_by_seed = {}
    all_optical_by_seed = {}
    for seed in SEEDS:
        u_train, y_train, u_test, y_test, _ = tanh_gate_task(N_STEPS, seed=seed)
        parallel_by_seed[seed] = evaluate(lambda utr, ute, sd=seed: build_parallel(utr, ute, sd), u_train, y_train, u_test, y_test)[1]
        all_optical_by_seed[seed] = evaluate(
            lambda utr, ute, sd=seed: build_all_optical_dual_delay(utr, ute, sd), u_train, y_train, u_test, y_test
        )[1]

    parallel_mean, parallel_std = np.mean(list(parallel_by_seed.values())), np.std(list(parallel_by_seed.values()))
    all_optical_mean, all_optical_std = np.mean(list(all_optical_by_seed.values())), np.std(list(all_optical_by_seed.values()))

    tau_results = {}
    for tau in TAU_VALUES:
        cell = []
        for seed in SEEDS:
            u_train, y_train, u_test, y_test, _ = tanh_gate_task(N_STEPS, seed=seed)
            result = evaluate(
                lambda utr, ute, sd=seed, t=tau: build_serial_force_transduced(utr, ute, sd, t),
                u_train, y_train, u_test, y_test,
            )
            cell.append(result[1])
        tau_results[tau] = (np.mean(cell), np.std(cell))

        label = f"{tau}" + (" (ideal)" if tau == 0 else "")
        print(f"{label:<8}{parallel_mean:.4f}+-{parallel_std:.4f}       "
              f"{tau_results[tau][0]:.4f}+-{tau_results[tau][1]:.4f}              "
              f"{all_optical_mean:.4f}+-{all_optical_std:.4f}")
    print()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    taus = list(tau_results.keys())
    means = [tau_results[t][0] for t in taus]
    stds = [tau_results[t][1] for t in taus]
    ax.errorbar(taus, means, yerr=stds, marker="o", color="#eb6834", label="serial_force (transduced)", capsize=3)
    ax.axhline(parallel_mean, color="#2a78d6", linestyle="--", label="parallel (no coupling)")
    ax.axhline(all_optical_mean, color="#1baf7a", linestyle="--", label="all_optical_dual_delay (unpenalized)")
    ax.axvspan(5, 27, alpha=0.15, color="orange", label="realistic transducer range")
    ax.set_xlabel("transducer time constant (outer timesteps)")
    ax.set_ylabel("fast-target NRMSE (lower = better)")
    ax.set_title("Which architecture wins as transduction bandwidth becomes realistic?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{VISUALS_DIR}/exp15_architecture_vs_tau.png", dpi=130)
    plt.close(fig)
    print(f"Saved plot to {VISUALS_DIR}/exp15_architecture_vs_tau.png\n")

    # Export full comparison results
    with open(f"{RESULTS_DIR}/exp15_full_comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tau_timesteps", "parallel_mean", "parallel_std", "all_optical_mean", "all_optical_std",
                         "serial_force_mean", "serial_force_std"])
        for tau in TAU_VALUES:
            mean_val, std_val = tau_results[tau]
            writer.writerow([tau, f"{parallel_mean:.6f}", f"{parallel_std:.6f}",
                           f"{all_optical_mean:.6f}", f"{all_optical_std:.6f}",
                           f"{mean_val:.6f}", f"{std_val:.6f}"])
    print(f"Results saved to {RESULTS_DIR}/exp15_full_comparison.csv\n")


if __name__ == "__main__":
    mechanism_vs_transduction()
    full_comparison_across_tau()