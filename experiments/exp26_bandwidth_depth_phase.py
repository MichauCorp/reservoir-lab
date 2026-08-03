"""
exp26: 2D phase diagram -- fast-component bandwidth x gate modulation depth.

exp24 fixed the gate at full saturation and swept continuous spectral
flatness of the fast component: serial_force won at every point tested,
even far beyond real ECG's measured flatness (0.0064) -- ruling out
bandwidth alone as the explanation for the real-ECG crossover.

exp25 fixed bandwidth at the real-ECG-matched value (alpha=0.15) and swept
gate modulation depth instead: the size-matched ESN won at and around real
ECG's measured modulation depth (~0.35, phase-binned exactly like exp18's
verify_pac), with serial_force only taking over at near-saturating depth --
making depth a plausible explanation.

Neither varied both axes together. This experiment sweeps both into one
2D phase diagram, reusing the same two metrics exp24/exp25 used so the axes
stay directly comparable to real measurements:

  x-axis: spectral flatness of the fast component (exp24's metric)
  y-axis: measured gate modulation depth (exp25's metric)

Real ECG is overlaid as a point (flatness 0.0064, depth ~0.35). The
discriminating question:

  - If the serial_force/ESN crossover boundary is horizontal in depth
    (bandwidth-independent), depth is confirmed as THE controlling
    variable and exp24/exp25 are reconciled: exp24 never saw a crossover
    because its depth was saturating everywhere.
  - If the boundary is diagonal, bandwidth and depth interact -- a result
    neither single-axis experiment could have seen.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.exp24_bandwidth_mapping import (
    blended_wave,
    real_ecg_fast_band_flatness,
    spectral_flatness,
)
from experiments.exp25_modulation_depth import (
    measured_modulation_depth,
    real_ecg_modulation_depth,
)
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

# Bandwidth axis: alpha=0.0 is a pure sinusoid, alpha=1.0 is exp20's sharp
# broadband pulse train, alpha=0.15 matches real ECG's measured spectral
# flatness in exp24's blend parametrization.
ALPHAS = [0.0, 0.15, 0.3, 0.5, 1.0]

# Depth axis: depth=1.0 is the saturating tanh gate every prior synthetic
# task used; depth=0.18 measures at real ECG's modulation depth (~0.35).
DEPTHS = [0.1, 0.18, 0.3, 0.5, 1.0]


def cell_edges(centers):
    """Extend 1-D cell-center coordinates into cell-edge coordinates for
    pcolormesh, handling the non-uniform measured grids correctly."""
    centers = np.asarray(centers, dtype=float)
    mids = (centers[:-1] + centers[1:]) / 2
    return np.concatenate([[2 * centers[0] - mids[0]], mids, [2 * centers[-1] - mids[-1]]])


def dual_timescale_task_2d(n_steps, alpha, depth, seed=0, noise_std=0.35):
    """alpha controls fast-component bandwidth (exp24's blend of sinusoid
    and broadband pulse); depth scales the gate's deviation from its 0.5
    baseline (exp25's parametrization, depth=1.0 = saturating tanh gate)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)

    slow = 0.6 * np.sin(2 * np.pi * t / 180.0)
    gate_full = 0.5 * (1 + np.tanh(4 * slow))
    gate = 0.5 + depth * (gate_full - 0.5)

    fast_phase = 2 * np.pi * t / 12.0 + 0.6 * np.sin(2 * np.pi * t / 33.0)
    fast_unit = blended_wave(fast_phase, alpha)
    fast_raw = 0.3 * fast_unit
    noise = rng.normal(0, noise_std, n_steps)

    fast_target = fast_raw * gate
    u = (slow + fast_raw + noise).reshape(-1, 1)
    y = np.stack([slow, fast_target], axis=1)

    split = int(n_steps * TRAIN_FRAC)
    return u[:split], y[:split], u[split:], y[split:], slow, fast_target, fast_raw


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


def main():
    print("=" * 78)
    print("exp26: 2D phase diagram -- bandwidth x modulation depth")
    print("=" * 78)
    print(f"Grid: {len(ALPHAS)} alphas x {len(DEPTHS)} depths x {len(ARCHS)} archs "
          f"x {len(SEEDS)} seeds = {len(ALPHAS) * len(DEPTHS) * len(ARCHS) * len(SEEDS)} runs\n")

    results = []
    flatness_by_alpha = {}
    measured_by_depth = {}
    for alpha in ALPHAS:
        for depth in DEPTHS:
            for seed in SEEDS:
                u_train, y_train, u_test, y_test, slow, fast_target, fast_raw = dual_timescale_task_2d(
                    N_STEPS, alpha, depth, seed=seed
                )
                if alpha not in flatness_by_alpha:
                    flatness_by_alpha[alpha] = spectral_flatness(fast_raw)
                if depth not in measured_by_depth:
                    measured_by_depth[depth] = measured_modulation_depth(slow, fast_target)

                for arch_name, builder in ARCHS.items():
                    result = evaluate(builder, u_train, y_train, u_test, y_test, seed)
                    results.append({
                        "alpha": alpha, "flatness": flatness_by_alpha[alpha],
                        "depth_param": depth, "measured_depth": measured_by_depth[depth],
                        "seed": seed, "architecture": arch_name,
                        "nrmse_slow": result[0], "nrmse_fast": result[1],
                    })

    with open(f"{RESULTS_DIR}/exp26_bandwidth_depth_phase.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "flatness", "depth_param", "measured_depth",
                                               "seed", "architecture", "nrmse_slow", "nrmse_fast"])
        writer.writeheader()
        writer.writerows(results)

    # Real ECG reference point, using exp24/exp25's own measurement functions.
    ecg_flatness = real_ecg_fast_band_flatness()
    ecg_depth = real_ecg_modulation_depth()
    print(f"Real ECG: fast-band spectral flatness={ecg_flatness:.4f}, "
          f"measured modulation depth={ecg_depth:.4f}\n")

    # ---- summarized tables -------------------------------------------------
    summary = {}
    for alpha in ALPHAS:
        for depth in DEPTHS:
            for arch_name in ARCHS:
                vals = [r["nrmse_fast"] for r in results
                        if r["alpha"] == alpha and r["depth_param"] == depth and r["architecture"] == arch_name]
                summary[(alpha, depth, arch_name)] = np.mean(vals)
            sf = summary[(alpha, depth, "serial_force")]
            esn = summary[(alpha, depth, "esn_matched")]
            wins = sum(1 for sf_v, esn_v in zip(
                [r["nrmse_fast"] for r in results
                 if r["alpha"] == alpha and r["depth_param"] == depth and r["architecture"] == "serial_force"],
                [r["nrmse_fast"] for r in results
                 if r["alpha"] == alpha and r["depth_param"] == depth and r["architecture"] == "esn_matched"],
            ) if sf_v < esn_v)
            summary[(alpha, depth, "winner")] = ("serial_force" if sf < esn else "esn_matched")
            summary[(alpha, depth, "sf_wins")] = wins

    print(f"{'alpha':<7}{'flatness':<11}{'depth_param':<13}{'measured':<10}{'winner':<14}{'sf seed-wins':<14}"
          f"{'fast NRMSE sf':<16}{'fast NRMSE esn'}")
    print("-" * 100)
    for alpha in ALPHAS:
        for depth in DEPTHS:
            sf = summary[(alpha, depth, "serial_force")]
            esn = summary[(alpha, depth, "esn_matched")]
            print(f"{alpha:<7}{flatness_by_alpha[alpha]:<11.4f}{depth:<13}{measured_by_depth[depth]:<10.4f}"
                  f"{summary[(alpha, depth, 'winner')]:<14}{summary[(alpha, depth, 'sf_wins')]:<14}/6"
                  f"{sf:<16.4f}{esn:<16.4f}")
        print()

    # ---- plot --------------------------------------------------------------
    depth_vals = [measured_by_depth[d] for d in DEPTHS]
    flat_vals = [flatness_by_alpha[a] for a in ALPHAS]
    z_diff = np.array([[summary[(a, d, "esn_matched")] - summary[(a, d, "serial_force")]
                        for d in DEPTHS] for a in ALPHAS])
    vmax = max(abs(z_diff.min()), abs(z_diff.max()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    X, Y = np.meshgrid(cell_edges(depth_vals), cell_edges(flat_vals))
    pc = ax1.pcolormesh(X, Y, z_diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="flat")
    cb = fig.colorbar(pc, ax=ax1)
    cb.set_label("fast NRMSE diff (esn - serial_force)   [>0: serial_force wins]")
    for i, a in enumerate(ALPHAS):
        for j, d in enumerate(DEPTHS):
            w = summary[(a, d, "winner")]
            txt = "sf" if w == "serial_force" else "esn"
            color = "white" if abs(z_diff[i, j]) > 0.6 * vmax else "black"
            ax1.text(depth_vals[j], flat_vals[i], txt, ha="center", va="center",
                     fontsize=8, color=color, fontweight="bold")
    ax1.plot(ecg_depth, ecg_flatness, "*", color="black", markersize=17,
             label=f"real ECG ({ecg_flatness:.3f}, {ecg_depth:.2f})")
    ax1.set_xlabel("measured gate modulation depth")
    ax1.set_ylabel("fast-component spectral flatness")
    ax1.set_title("exp26: winner phase diagram, serial_force vs size-matched ESN")
    ax1.legend(fontsize=9, loc="upper left")

    cmap_a = plt.cm.viridis(np.linspace(0, 1, len(ALPHAS)))
    for k, a in enumerate(ALPHAS):
        sf_means = [summary[(a, d, "serial_force")] for d in DEPTHS]
        esn_means = [summary[(a, d, "esn_matched")] for d in DEPTHS]
        ax2.plot(depth_vals, sf_means, "-o", color=cmap_a[k], label=f"serial_force alpha={a}")
        ax2.plot(depth_vals, esn_means, "--o", color=cmap_a[k], alpha=0.55, label=f"esn_matched alpha={a}")
    ax2.axvline(ecg_depth, color="black", linestyle="--", alpha=0.7,
                label=f"real ECG depth ({ecg_depth:.2f})")
    ax2.set_xlabel("measured gate modulation depth")
    ax2.set_ylabel("fast-target NRMSE (lower = better)")
    ax2.set_title("cross-sections: fast NRMSE vs depth, per bandwidth")
    ax2.legend(fontsize=7, ncol=2, loc="lower right")

    fig.tight_layout()
    fig.savefig(f"{VISUALS_DIR}/exp26_bandwidth_depth_phase.png", dpi=130)
    plt.close(fig)
    print(f"Saved plot to {VISUALS_DIR}/exp26_bandwidth_depth_phase.png")

    # ---- conclusion --------------------------------------------------------
    print("\n" + "=" * 78)
    print("CONCLUSION")
    print("=" * 78)
    switch_by_alpha = {}
    for alpha in ALPHAS:
        seq = [summary[(alpha, d, "winner")] for d in DEPTHS]
        print(f"  alpha={alpha} (flatness={flatness_by_alpha[alpha]:.4f}): winners by measured depth "
              + ", ".join(f"{measured_by_depth[d]:.3f}:{w}" for d, w in zip(DEPTHS, seq)))
        switch = None
        for i in range(1, len(seq)):
            if seq[i - 1] == "esn_matched" and seq[i] == "serial_force":
                switch = measured_by_depth[DEPTHS[i]]
                break
        switch_by_alpha[alpha] = switch

    present = {a: s for a, s in switch_by_alpha.items() if s is not None}
    if present:
        lo, hi = min(present.values()), max(present.values())
        print(f"\nCrossover (ESN -> serial_force) at measured depth: "
              f"{', '.join(f'alpha={a}: {s:.3f}' for a, s in present.items())}")
        print(f"Switch-depth span across alphas: {hi - lo:.3f}")
        if hi - lo < 0.4:
            print("-> Boundary is roughly HORIZONTAL in depth: depth is confirmed as THE")
            print("   controlling variable; exp24/25 are reconciled (exp24 never saw a")
            print("   crossover because its depth was saturating everywhere).")
        else:
            print("-> Boundary shifts with bandwidth: bandwidth and depth INTERACT -- a result")
            print("   neither exp24 nor exp25 alone could have shown.")
    else:
        print("\nNo ESN->serial_force switch found at any alpha in the depths tested.")

    print(f"\nReal ECG point: flatness={ecg_flatness:.4f}, depth={ecg_depth:.4f}")
    # Nearest grid cells to the real ECG point.
    near_alpha = min(ALPHAS, key=lambda a: abs(flatness_by_alpha[a] - ecg_flatness))
    near_depth = min(DEPTHS, key=lambda d: abs(measured_by_depth[d] - ecg_depth))
    near_winner = summary[(near_alpha, near_depth, "winner")]
    print(f"Nearest grid cell: alpha={near_alpha} (flatness={flatness_by_alpha[near_alpha]:.4f}), "
          f"depth={near_depth} (measured={measured_by_depth[near_depth]:.4f}) -> {near_winner} wins")
    if near_winner == "esn_matched":
        print("Real ECG falls in the ESN-favoring region -- consistent with exp18/exp23's")
        print("real-data crossover and with exp25's depth explanation.")
    else:
        print("Real ECG falls in the serial_force-favoring region -- depth alone does not")
        print("explain the real-data crossover.")


if __name__ == "__main__":
    main()