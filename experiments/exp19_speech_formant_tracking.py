"""
Speech formant tracking: multi-step regression on a source-filter signal.

Tests whether the coupling advantage extends from classification (exp18) to
continuous trajectory prediction. The signal is synthesized from a simplified
source-filter model of speech:

  - Glottal pulse train (periodic, ~10 ms pitch period = 100 Hz)
  - Two formant resonances (vocal tract filters),with formant-1 amplitude
    smoothly gated by a slow voicing-state trajectory (0.5 Hz transitions)
  - Voicing state smoothly transitions between voiced (formant present) and
    unvoiced (formant suppressed) — a classic slow-gates-fast amplitude
    modulation, biophysically analogous to the EEG alpha-suppression mechanism

Task: predict the formant-1 amplitude envelope (slow signal) from the raw
waveform. This is a multi-step-ahead regression problem: the readout must
reconstruct the slowly-varying gating trajectory while ignoring the fast
pitch-period oscillations.

Architectures:
  - serial_force  (optical→acoustic, force-coupled)
  - parallel      (both reservoirs driven independently)
  - all_optical   (two force-coupled optical stages)
  - simple_esn    (software ESN baseline)

Design: 5 seeds, NRMSE metric, condition-number+effective-rank analysis on
feature matrices.
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
THETA2 = 3.0
PITCH_PERIOD = 10  # samples (~100 Hz at dt=0.1)
N_STEPS = 8000


# =====================================================================
# Source-filter speech synthesis
# =====================================================================

def _glottal_pulse(pitch_period):
    """Simple Rosenberg glottal pulse model."""
    t = np.linspace(0, 1, pitch_period)
    pulse = 0.5 * (1 - np.cos(2 * np.pi * t))
    return pulse


def generate_speech_signal(n_steps=N_STEPS, seed=0):
    """Generate a speech-realistic signal with slow-gated formant amplitude.

    Returns:
        u: waveform (n_steps, 1)
        envelope: target formant-1 amplitude envelope (n_steps, 1)
    """
    rng = np.random.default_rng(seed)
    pitch_period = PITCH_PERIOD
    pulse = _glottal_pulse(pitch_period)

    # Slow voicing-state trajectory (0.5 Hz)
    t = np.arange(n_steps)
    slow = 0.6 * np.sin(2 * np.pi * t / 2000.0) + 0.4 * rng.normal(0, 0.05, n_steps)
    gate = 0.5 * (1 + np.tanh(1.5 * slow))  # smooth gate [0,1]

    # Formant-1 resonance (~700 Hz) via simple exponential decay filter
    formant_freq = 0.35  # normalized frequency
    alpha = np.exp(-2 * np.pi * formant_freq)
    formant = np.zeros(n_steps + 2)
    for i in range(n_steps):
        idx = i % pitch_period
        excitation = pulse[idx] * gate[i]
        formant[i + 2] = alpha**2 * formant[i] + 2 * alpha * excitation

    # Trim and normalize
    formant = formant[2:n_steps + 2].copy()
    formant /= (np.max(np.abs(formant)) + 1e-8)

    # Observed waveform = glottal excitation directly (pre-formation)
    # In a real vocoder, the output is the post-formant signal; here we give
    # the model the raw excitation and ask it to predict the envelope
    u = np.zeros((n_steps, 1))
    for i in range(n_steps):
        u[i, 0] = pulse[i % pitch_period] * (0.5 + 0.5 * rng.normal())

    envelope = formant.reshape(-1, 1)
    return u, envelope


def prepare_splits(u, y, train_frac=0.7):
    n = len(u)
    train_end = int(n * train_frac)
    return u[:train_end], y[:train_end], u[train_end:], y[train_end:]


# =====================================================================
# Architecture builders
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

def nrmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)) / (np.std(true) + 1e-8))


def evaluate(train_feats, y_train, test_feats, y_test, reg=REG):
    readout = RidgeReadout(reg=reg).fit(train_feats, y_train, washout=WASHOUT)
    pred = readout.predict(test_feats)
    error = nrmse(pred, y_test)
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
    return error, cond_number, eff_rank, eff_rank_frac


# =====================================================================
# Main sweep
# =====================================================================

def main():
    print("Generating speech-realistic source-filter signal ...")
    u, envelope = generate_speech_signal(seed=SEEDS[0])
    u_train, y_train, u_test, y_test = prepare_splits(u, envelope)
    print(f"Train: {u_train.shape[0]} samples, Test: {u_test.shape[0]} samples")
    print(f"Envelope std train: {y_train.std():.4f}, test: {y_test.std():.4f}\n")

    rows = []

    for seed in SEEDS:
        for arch_name, builder_fn in BUILDERS.items():
            train_feats, test_feats = builder_fn(u_train, u_test, seed)
            error, cond, eff_rank, eff_rank_frac = evaluate(
                train_feats, y_train, test_feats, y_test
            )
            row = {
                "architecture": arch_name,
                "seed": seed,
                "nrmse": f"{error:.4f}",
                "condition_number": f"{cond:.2e}" if cond is not None else "",
                "effective_rank": f"{eff_rank:.2f}" if eff_rank is not None else "",
                "effective_rank_fraction": f"{eff_rank_frac:.4f}" if eff_rank_frac is not None else "",
            }
            rows.append(row)
            cond_str = f"{cond:.2e}" if cond is not None else "—"
            erf_str = f"{eff_rank_frac:.4f}" if eff_rank_frac is not None else "—"
            print(f"  {arch_name:<14} seed={seed}  nrmse={error:.4f}  cond={cond_str:<10}  eff_rank_frac={erf_str}")

    csv_path = f"{RESULTS_DIR}/exp19_speech_formant_tracking.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {csv_path}")

    # Aggregate
    nrmse_by_arch = {}
    cond_by_arch = {}
    erf_by_arch = {}
    for row in rows:
        a = row["architecture"]
        nrmse_by_arch.setdefault(a, []).append(float(row["nrmse"]))
        if row["condition_number"]:
            cond_by_arch.setdefault(a, []).append(float(row["condition_number"]))
        if row["effective_rank_fraction"]:
            erf_by_arch.setdefault(a, []).append(float(row["effective_rank_fraction"]))

    print("\nAggregate by architecture (mean ± 95% CI):")
    print(f"{'architecture':<14} {'nrmse':>8}  {'cond_number':>14}  {'eff_rank_frac':>13}")
    print("-" * 58)
    for arch in BUILDERS:
        errs = nrmse_by_arch[arch]
        err_mean = np.mean(errs)
        err_ci = stats.sem(errs) * stats.t.ppf((1 + 0.95) / 2, len(errs) - 1)
        cond_str = ""
        if arch in cond_by_arch:
            c_mean = np.mean(cond_by_arch[arch])
            cond_str = f"{c_mean:.2e}"
        erf_str = ""
        if arch in erf_by_arch:
            e_mean = np.mean(erf_by_arch[arch])
            erf_str = f"{e_mean:.4f}"
        print(f"{arch:<14} {err_mean:.4f} ± {err_ci:.4f}  {cond_str:>14}  {erf_str:>13}")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    colors = {
        "serial_force": "#eb6834",
        "parallel": "#2a78d6",
        "all_optical": "#1baf7a",
        "simple_esn": "#9b59b6",
    }

    # NRMSE bars (lower is better)
    ax = axes[0]
    arch_names = list(BUILDERS.keys())
    means = [np.mean(nrmse_by_arch[a]) for a in arch_names]
    cis = [stats.sem(nrmse_by_arch[a]) * stats.t.ppf((1 + 0.95) / 2, len(nrmse_by_arch[a]) - 1) for a in arch_names]
    bars = ax.bar(arch_names, means, color=[colors[a] for a in arch_names], edgecolor="black", linewidth=0.6)
    ax.errorbar(arch_names, means, yerr=cis, fmt="none", color="black", capsize=4)
    ax.set_ylabel("NRMSE (lower is better)")
    ax.set_title(f"Speech Formant Tracking NRMSE\n(mean ± 95% CI, {len(SEEDS)} seeds)")
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
    fig_path = f"{VISUALS_DIR}/exp19_speech_formant_tracking.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"Saved plot to {fig_path}")

    # Statistical tests: serial_force vs others (lower NRMSE = better)
    print("\nPaired t-tests (serial_force vs each other architecture, lower NRMSE wins):")
    sf_errs = nrmse_by_arch["serial_force"]
    for arch in ["parallel", "all_optical", "simple_esn"]:
        other_errs = nrmse_by_arch[arch]
        t_stat, p_val = stats.ttest_rel(sf_errs, other_errs)
        direction = "serial_force <" if np.mean(sf_errs) < np.mean(other_errs) else "serial_force >"
        print(f"  serial_force vs {arch:<14}: {direction}  p={p_val:.4f}")


if __name__ == "__main__":
    main()