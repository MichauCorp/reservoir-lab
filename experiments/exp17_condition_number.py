"""
Why is serial_force more sensitive to regularization than parallel or
all_optical? (exp16 found catastrophic NRMSE degradation for serial_force
specifically at reg=1e-6, while parallel and all_optical stayed
comparatively stable at the same regularization.)

Hypothesis: force-coupling drives every acoustic oscillator from the same
upstream optical signal (via a single shared driving-force projection),
which should make the acoustic state's channels more mutually correlated
than parallel's independently-driven acoustic. Highly correlated feature
columns make X^T X (the matrix ridge regression inverts) closer to
singular -- exactly what makes the ridge solution unstable when the
regularization term, which is the only thing keeping X^T X + reg*I
invertible, is too small.

This computes the singular value spectrum, condition number, and
effective rank (participation ratio) of each architecture's feature
matrix directly, rather than only inferring the cause from downstream
NRMSE instability.
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

RESULTS_DIR = "experiments/data_results"
VISUALS_DIR = "experiments/visuals"
WASHOUT = 100
SEED = 1


# =====================================================================
# Task generator (reused from exp16)
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


# =====================================================================
# Architecture builders (reused from exp16)
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


def build_all_optical(u_train, u_test, seed, theta, c, n_reservoir, reg, theta2=3.0):
    """Two optical stages with decoupled response-time and delay-window.
    dt=0.02 is fixed (matching exp14's fix) so theta2 controls only the
    response timescale, not the delay-window size."""
    stage1 = OptoelectronicReservoir(
        n_inputs=1, n_virtual_nodes=n_reservoir, theta=theta, seed=seed
    )
    stage2 = OptoelectronicReservoir(
        n_inputs=n_reservoir, n_virtual_nodes=n_reservoir, theta=theta2, dt=0.02, seed=seed + 1
    )
    model = SerialPhysicalReservoir(stage1=stage1, stage2=stage2, combine="both", seed=seed)
    train_feats = model.run(u_train)
    test_feats = model.run(u_test, initial_state=model.last_state)
    return train_feats, test_feats


# =====================================================================
# Condition-number analysis
# =====================================================================

# Includes theta=1.0, c=0.3 -- the exact combo where serial_force
# catastrophically failed at reg=1e-6 in exp16.
CONFIGS = [
    {"theta": 0.2, "c": 0.3, "n": 100},
    {"theta": 1.0, "c": 0.3, "n": 100},
    {"theta": 0.2, "c": 1.0, "n": 100},
    {"theta": 1.0, "c": 1.0, "n": 100},
]


def effective_rank(singular_values):
    """Participation-ratio effective rank: (sum s)^2 / sum(s^2).
    Equals n_features for perfectly uncorrelated, equal-magnitude columns;
    shrinks toward 1 as columns become redundant/correlated."""
    s = singular_values
    return float((s.sum() ** 2) / (s**2).sum())


def analyze_matrix(X, washout=WASHOUT):
    Xw = X[washout:]
    Xc = Xw - Xw.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    s = s[s > 1e-12]  # drop numerically-zero singular values before ratio
    cond = float(s[0] / s[-1])
    eff_rank = effective_rank(s)
    return s, cond, eff_rank


def main():
    results = []
    spectra = {}  # populated for the first config only, for the plot

    for i, cfg in enumerate(CONFIGS):
        u_train, _y_train, u_test, _y_test = dual_timescale_task(seed=SEED)

        builders = {
            "serial_force": build_serial_force(u_train, u_test, SEED, cfg["theta"], cfg["c"], cfg["n"], reg=1e-3)[0],
            "parallel": build_parallel(u_train, u_test, SEED, cfg["theta"], cfg["c"], cfg["n"], reg=1e-3)[0],
            "all_optical": build_all_optical(u_train, u_test, SEED, cfg["theta"], cfg["c"], cfg["n"], reg=1e-3, theta2=3.0)[0],
        }

        for arch_name, Xtr in builders.items():
            s, cond, eff_rank = analyze_matrix(Xtr)
            n_features = Xtr.shape[1]
            results.append({
                "architecture": arch_name,
                "theta": cfg["theta"],
                "c": cfg["c"],
                "n_reservoir": cfg["n"],
                "n_features": n_features,
                "condition_number": f"{cond:.2e}",
                "effective_rank": f"{eff_rank:.2f}",
                "effective_rank_fraction": f"{eff_rank / n_features:.4f}",
            })
            if i == 1:  # theta=1.0, c=0.3 -- the failure case -- for the plot
                spectra[arch_name] = s

    with open(f"{RESULTS_DIR}/exp17_condition_number.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"{'architecture':<14}{'theta':<7}{'c':<6}{'cond_number':<14}{'eff_rank':<11}{'eff_rank_frac'}")
    print("-" * 66)
    for r in results:
        print(f"{r['architecture']:<14}{r['theta']:<7}{r['c']:<6}{r['condition_number']:<14}{r['effective_rank']:<11}{r['effective_rank_fraction']}")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"serial_force": "#eb6834", "parallel": "#2a78d6", "all_optical": "#1baf7a"}
    for name, s in spectra.items():
        ax.plot(np.arange(1, len(s) + 1), s / s[0], label=name, color=colors[name])
    ax.set_yscale("log")
    ax.set_xlabel("singular value index")
    ax.set_ylabel("singular value (normalized to largest)")
    ax.set_title("Singular value spectra at theta=1.0, c=0.3 (serial_force's failure case)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{VISUALS_DIR}/exp17_singular_spectra.png", dpi=130)
    plt.close(fig)
    print(f"\nSaved plot to {VISUALS_DIR}/exp17_singular_spectra.png")


if __name__ == "__main__":
    main()