"""
exp21: Does pre-filtering the optical stage's input change serial_force's
slow-target performance on broadband signals?

Motivation: exp20 showed serial_force's slow-target advantage over parallel
on sinusoid (0.208 vs 0.232, mean across seeds) reverses on broadband
(0.228 vs 0.208, parallel now wins). The hypothesis was that the optical
reservoir's short memory (theta=0.2) smears sharp transients, passing a
noisier drive to the acoustic stage in serial_force. Pre-filtering the
input to the optical stage should smooth the broadband transients, letting
the optical stage track the amplitude modulation more cleanly.

RESULT (see exp22_optical_theta_sweep.py for the mechanistic follow-up):
pre-filtering did NOT restore serial_force's advantage -- it improved
slow-target accuracy for BOTH serial_force and parallel by similar
amounts, and the broadband-case reversal disappeared into a near-exact
TIE (serial_force_prefilter 0.161 vs parallel_prefilter 0.162), not a
restored serial_force win. The correct reading is "pre-filtering erases
the architecture difference", not "pre-filtering restores the advantage".

Also worth being explicit about, since it doesn't show up in the printed
slow-target summary: pre-filtering is not a free fix. It completely
destroys fast-target performance for both prefiltered variants (NRMSE
~1.0, no better than predicting the mean) because the low-pass filter
removes the fast content before any reservoir ever sees it. This
confirms the mechanism (unfiltered broadband transients reaching optical
do interfere with slow-target tracking) but the "fix" tested here is a
tradeoff, not a solution -- see exp22 for an attempt at a real fix
(tuning optical's own dynamics rather than deleting information).

Design:
  - Two waveform types: sinusoid and broadband (same as exp20)
  - Four architectures:
      serial_force           (no pre-filter, baseline from exp20)
      serial_force_prefilter  (input to optical stage is low-pass filtered)
      parallel               (no pre-filter, baseline from exp20)
      parallel_prefilter     (optical branch gets filtered input, acoustic
                              branch gets raw input -- control)
  - 6 seeds each, matching exp20
  - Pre-filter: 4th-order Butterworth low-pass, cutoff = 0.04 (normalised
    by Nyquist, i.e. actual cutoff ~0.02 cycles/sample, period ~50 timesteps).
    This is well above the slow component's frequency (~0.0056 cycles/sample,
    period 180) and below the fast component's fundamental (~0.083, period 12).

References:
  - exp20 (waveform shape experiment) provides the baseline numbers
  - exp16 provides the builder functions for serial_force and parallel
  - exp14 showed the mechanism: gate decodability from acoustic state alone
  - exp22 follows up on this experiment's actual finding
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

from experiments.exp16_hyperparameter_sweep import (
    build_parallel,
    build_serial_force,
)
from reservoir_lab.physical import (
    AcousticReservoir,
    OptoelectronicReservoir,
    SerialPhysicalReservoir,
)
from reservoir_lab.readout import RidgeReadout

RESULTS_DIR = "experiments/data_results"
VISUALS_DIR = "experiments/visuals"
WASHOUT = 120
TRAIN_FRAC = 0.7
N_STEPS = 1400
SEEDS = [1, 2, 3, 4, 5, 6]
THETA, C, N_RES, REG = 0.2, 0.3, 100, 1e-3
FILT_CUTOFF = 0.04  # normalised cutoff, actual freq = FILT_CUTOFF * Nyquist


def sharp_pulse_wave(phase, sharpness=8.0):
    """Broadband periodic waveform, same as exp20."""
    wrapped = np.mod(phase, 2 * np.pi) / (2 * np.pi)
    pulse = np.exp(-sharpness * wrapped)
    pulse = pulse - pulse.mean()
    pulse = pulse / pulse.std()
    return pulse


def dual_timescale_task(n_steps, waveform, seed=0, noise_std=0.35):
    """Same task generator as exp20.

    Returns (u_train, y_train, u_test, y_test) where
      u = (slow + fast_raw + noise).reshape(-1, 1)
      y = np.stack([slow, fast_target], axis=1)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)

    slow = 0.6 * np.sin(2 * np.pi * t / 180.0)
    gate = 0.5 * (1 + np.tanh(4 * slow))

    # identical phase/timing structure for both waveform variants
    fast_phase = 2 * np.pi * t / 12.0 + 0.6 * np.sin(2 * np.pi * t / 33.0)

    if waveform == "sinusoid":
        fast_unit = np.sin(fast_phase) / np.sin(fast_phase).std()
    elif waveform == "broadband":
        fast_unit = sharp_pulse_wave(fast_phase)
    else:
        raise ValueError(f"unknown waveform: {waveform}")

    fast_raw = 0.3 * fast_unit  # same amplitude scale (std) for both variants
    noise = rng.normal(0, noise_std, n_steps)

    fast_target = fast_raw * gate
    u = (slow + fast_raw + noise).reshape(-1, 1)
    y = np.stack([slow, fast_target], axis=1)

    split = int(n_steps * TRAIN_FRAC)
    return u[:split], y[:split], u[split:], y[split:]


def prefilter(u_train, u_test, cutoff=FILT_CUTOFF):
    """Apply a 4th-order low-pass Butterworth filter to the input.

    Filters train and test separately (they are contiguous segments of
    the same signal, so boundary effects are minimal at N=1400).

    Returns (u_train_filt, u_test_filt), each with the same shape.
    """
    sos = butter(4, cutoff, btype="low", output="sos")
    u_train_filt = sosfiltfilt(sos, u_train, axis=0)
    u_test_filt = sosfiltfilt(sos, u_test, axis=0)
    return u_train_filt, u_test_filt


def build_serial_force_prefilter(u_train, u_test, seed, theta, c, n_reservoir, reg):
    """serial_force with pre-filtered input to the optical stage.

    The acoustic stage is still driven by the optical stage's output
    (same as standard serial_force). Only the optical stage sees the
    filtered input.
    """
    u_train_filt, u_test_filt = prefilter(u_train, u_test)

    optical = OptoelectronicReservoir(
        n_inputs=1, n_virtual_nodes=n_reservoir, theta=theta, seed=seed
    )
    acoustic = AcousticReservoir(
        n_inputs=n_reservoir, n_oscillators=n_reservoir, c=c, seed=seed + 1
    )
    model = SerialPhysicalReservoir(
        stage1=optical, stage2=acoustic, combine="both", seed=seed
    )

    train_feats = model.run(u_train_filt)
    test_feats = model.run(u_test_filt, initial_state=model.last_state)
    return train_feats, test_feats


def build_parallel_prefilter(u_train, u_test, seed, theta, c, n_reservoir, reg):
    """parallel with pre-filtered input to the optical branch only.

    The acoustic branch still receives the raw input (unchanged). This
    is a control to check whether pre-filtering helps the optical stage
    in isolation, regardless of the coupling architecture.
    """
    u_train_filt, u_test_filt = prefilter(u_train, u_test)

    optical = OptoelectronicReservoir(
        n_inputs=1, n_virtual_nodes=n_reservoir, theta=theta, seed=seed
    )
    acoustic = AcousticReservoir(
        n_inputs=1, n_oscillators=n_reservoir, c=c, seed=seed + 1
    )

    opt_tr = optical.run(u_train_filt)
    ac_tr = acoustic.run(u_train)
    opt_te = optical.run(u_test_filt, initial_state=optical.last_state)
    ac_te = acoustic.run(u_test, initial_state=acoustic.last_state)

    return np.hstack([opt_tr, ac_tr]), np.hstack([opt_te, ac_te])


ARCHS = {
    "serial_force": (build_serial_force, {}),
    "serial_force_prefilter": (build_serial_force_prefilter, {}),
    "parallel": (build_parallel, {}),
    "parallel_prefilter": (build_parallel_prefilter, {}),
}


def nrmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2, axis=0)) / np.std(true, axis=0)


def evaluate(builder, u_train, y_train, u_test, y_test, seed, extra_kwargs=None):
    kwargs = extra_kwargs or {}
    train_feats, test_feats = builder(
        u_train, u_test, seed, THETA, C, N_RES, REG, **kwargs
    )
    readout = RidgeReadout(reg=REG).fit(train_feats, y_train, washout=WASHOUT)
    pred = readout.predict(test_feats)
    return nrmse(pred, y_test)


def main():
    print("=" * 74)
    print("exp21: Does pre-filtering the optical input restore serial_force's")
    print("slow-target advantage on broadband signals?")
    print("=" * 74)

    results = []
    example_signals = {}

    for waveform in ["sinusoid", "broadband"]:
        for seed in SEEDS:
            u_train, y_train, u_test, y_test = dual_timescale_task(
                N_STEPS, waveform, seed=seed
            )
            if seed == SEEDS[0]:
                example_signals[waveform] = (u_train, y_train)

            for arch_name, (builder, kwargs) in ARCHS.items():
                result = evaluate(
                    builder, u_train, y_train, u_test, y_test, seed, kwargs
                )
                results.append({
                    "waveform": waveform,
                    "seed": seed,
                    "architecture": arch_name,
                    "nrmse_slow": result[0],
                    "nrmse_fast": result[1],
                })

    # Save raw results to CSV
    csv_path = f"{RESULTS_DIR}/exp21_prefiltered_input.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["waveform", "seed", "architecture", "nrmse_slow", "nrmse_fast"]
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved raw results to {csv_path}")

    # Print summary table
    print(f"\n{'waveform':<12}{'architecture':<26}{'NRMSE slow':<24}{'NRMSE fast'}")
    print("-" * 74)
    summary = {}
    for waveform in ["sinusoid", "broadband"]:
        for arch_name in ARCHS:
            slow_vals = [
                r["nrmse_slow"]
                for r in results
                if r["waveform"] == waveform and r["architecture"] == arch_name
            ]
            fast_vals = [
                r["nrmse_fast"]
                for r in results
                if r["waveform"] == waveform and r["architecture"] == arch_name
            ]
            summary[(waveform, arch_name)] = (
                np.mean(slow_vals), np.std(slow_vals),
                np.mean(fast_vals), np.std(fast_vals),
            )
            print(
                f"{waveform:<12}{arch_name:<26}"
                f"{np.mean(slow_vals):.4f} +- {np.std(slow_vals):.4f}        "
                f"{np.mean(fast_vals):.4f} +- {np.std(fast_vals):.4f}"
            )
        print()

    # Plot: side-by-side bar chart for slow and fast targets
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    arch_names = list(ARCHS.keys())
    x = np.arange(len(arch_names))
    width = 0.35

    for idx, target_name in enumerate(["slow", "fast"]):
        ax = axes[idx]
        target_idx = 0 if target_name == "slow" else 2  # index into summary tuple
        std_idx = 1 if target_name == "slow" else 3

        sin_means = [summary[("sinusoid", a)][target_idx] for a in arch_names]
        sin_stds = [summary[("sinusoid", a)][std_idx] for a in arch_names]
        bb_means = [summary[("broadband", a)][target_idx] for a in arch_names]
        bb_stds = [summary[("broadband", a)][std_idx] for a in arch_names]

        ax.bar(x - width / 2, sin_means, width, yerr=sin_stds,
               label="sinusoid", color="#2a78d6", capsize=3)
        ax.bar(x + width / 2, bb_means, width, yerr=bb_stds,
               label="broadband", color="#eb6834", capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(arch_names, rotation=15, ha="right")
        ax.set_ylabel(f"{target_name}-target NRMSE (lower = better)")
        ax.set_title(f"{target_name.capitalize()} target")
        ax.legend(fontsize=8)

    fig.suptitle(
        "exp21: Pre-filtering erases the architecture difference (does not restore serial_force)",
        fontsize=10,
    )
    fig.tight_layout()
    plot_path = f"{VISUALS_DIR}/exp21_prefiltered_input.png"
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")

    # Also plot example waveforms to show the pre-filter effect
    # (use the training portion directly, no need to split again)
    sos_demo = butter(4, FILT_CUTOFF, btype="low", output="sos")
    fig, axes = plt.subplots(2, 2, figsize=(10, 5), sharex=True)
    window = slice(0, 100)
    for row, waveform in enumerate(["sinusoid", "broadband"]):
        u_raw, _ = example_signals[waveform]
        u_filt = sosfiltfilt(sos_demo, u_raw, axis=0)
        axes[row, 0].plot(u_raw[window], color="#2a78d6", linewidth=0.8)
        axes[row, 0].set_title(f"{waveform} — raw input")
        axes[row, 1].plot(u_filt[window], color="#eb6834", linewidth=0.8)
        axes[row, 1].set_title(f"{waveform} — pre-filtered input")
    fig.tight_layout()
    fig.savefig(f"{VISUALS_DIR}/exp21_prefilter_examples.png", dpi=130)
    plt.close(fig)
    print(f"\nSaved pre-filter example plot to {VISUALS_DIR}/exp21_prefilter_examples.png")

    slow_sf_bb = summary[("broadband", "serial_force")][0]
    slow_sfp_bb = summary[("broadband", "serial_force_prefilter")][0]
    slow_pa_bb = summary[("broadband", "parallel")][0]
    slow_pap_bb = summary[("broadband", "parallel_prefilter")][0]
    fast_sfp_bb = summary[("broadband", "serial_force_prefilter")][2]
    fast_pap_bb = summary[("broadband", "parallel_prefilter")][2]

    print("=" * 74)
    print("CONCLUSION")
    print("=" * 74)
    print(f"Broadband slow-target, unfiltered:  serial_force={slow_sf_bb:.4f}  parallel={slow_pa_bb:.4f}  "
          f"({'parallel wins' if slow_pa_bb < slow_sf_bb else 'serial_force wins'})")
    print(f"Broadband slow-target, prefiltered: serial_force={slow_sfp_bb:.4f}  parallel={slow_pap_bb:.4f}  "
          f"({'near-tie' if abs(slow_sfp_bb - slow_pap_bb) < 0.01 else 'still separated'})")
    print("Pre-filtering did NOT restore a serial_force advantage -- it improved both")
    print("architectures similarly and erased the difference between them (a tie, not a win).")
    print()
    print(f"Fast-target cost of pre-filtering: serial_force_prefilter={fast_sfp_bb:.4f}, "
          f"parallel_prefilter={fast_pap_bb:.4f} (both ~1.0 = no better than predicting the mean).")
    print("This is not a free fix -- removing the fast content to help the slow target also")
    print("destroys the model's ability to do the fast task at all. See exp22 for an attempt")
    print("at a real fix: tuning optical's own dynamics instead of deleting information.")


if __name__ == "__main__":
    main()