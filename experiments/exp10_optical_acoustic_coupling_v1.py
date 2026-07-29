"""
input -> optical -> acoustic -> readout

Every physical reservoir in this project so far (exp08, exp09) has been
benchmarked on its own. This experiment asks the question those didn't:
does chaining OptoelectronicReservoir (fast, nonlinear, short memory by
construction -- see physical/optical.py) into AcousticReservoir (slow,
long memory via lightly-damped oscillators -- see physical/acoustic.py)
let a single readout recover BOTH a fast and a slow component of a noisy
signal at once, better than either stage alone?

Task: a noisy signal containing a slow and a fast sinusoidal component.
Two targets, recovered simultaneously by one linear readout:
  y_slow(t): needs long-window averaging / noise rejection
  y_fast(t): needs short-memory, higher-bandwidth tracking

Four configurations, same reservoir "budget" (n_reservoir per stage) so
the comparison isn't just "more neurons wins":
  - Optical only            (no acoustic stage)
  - Acoustic only           (acoustic driven directly by the raw input)
  - Optical -> Acoustic     (serial, readout sees only the acoustic stage
                             -- the literal input->optical->acoustic->readout
                             pipeline)
  - Optical + Acoustic      (serial, readout sees BOTH stages concatenated
                             -- ablation: does funnelling everything through
                             the acoustic stage alone lose information the
                             optical stage had?)

As in exp09: these are untuned default hyperparameters for each stage.
This is a first pass to see if the coupling hypothesis is worth chasing,
not a tuned final comparison (a fairer test would sweep theta/eta for the
optical stage and damping/coupling for the acoustic stage).
"""

import csv
import numpy as np

from reservoir_lab.readout import RidgeReadout
from reservoir_lab.physical import OptoelectronicReservoir, AcousticReservoir, SerialPhysicalReservoir

RESULTS_DIR = "experiments/data_results"
VISUALS_DIR = "experiments/visuals"


N_VIRTUAL = 150     # optical stage size
N_OSCILLATORS = 150  # acoustic stage size
WASHOUT = 150
TRAIN_FRAC = 0.7
NOISE_LEVELS = [0.15, 0.4, 0.8]
SEED = 42


# =====================================================================
# Task: noisy signal with a slow and a fast component
# =====================================================================

def dual_timescale_task(n_steps, noise_std, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)
    slow = 0.6 * np.sin(2 * np.pi * t / 180.0)
    fast = 0.3 * np.sin(2 * np.pi * t / 12.0 + 0.7 * np.sin(2 * np.pi * t / 37.0))
    noise = rng.normal(0, noise_std, n_steps)

    u = (slow + fast + noise).reshape(-1, 1)
    y = np.stack([slow, fast], axis=1)

    split = int(n_steps * TRAIN_FRAC)
    return u[:split], y[:split], u[split:], y[split:]


# =====================================================================
# Reservoir factories, one per configuration
# =====================================================================

def make_optical_only(seed):
    return OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=N_VIRTUAL, seed=seed)


def make_acoustic_only(seed):
    return AcousticReservoir(n_inputs=1, n_oscillators=N_OSCILLATORS, seed=seed)


def make_serial(seed, combine):
    stage1 = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=N_VIRTUAL, seed=seed)
    stage2 = AcousticReservoir(n_inputs=N_VIRTUAL, n_oscillators=N_OSCILLATORS, seed=seed + 1)
    return SerialPhysicalReservoir(stage1, stage2, combine=combine, seed=seed)


CONFIGS = {
    "Optical only":                lambda seed: make_optical_only(seed),
    "Acoustic only (raw input)":   lambda seed: make_acoustic_only(seed),
    "Optical -> Acoustic (serial)": lambda seed: make_serial(seed, combine="final"),
    "Optical + Acoustic (both taps)": lambda seed: make_serial(seed, combine="both"),
}


# =====================================================================
# Evaluation, same continue-from-last_state pattern as exp08/exp09
# =====================================================================

def nrmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2, axis=0)) / np.std(true, axis=0)


def evaluate(reservoir, train_inputs, train_targets, test_inputs, test_targets, washout=WASHOUT):
    states = reservoir.run(train_inputs)
    readout = RidgeReadout(reg=1e-3).fit(states, train_targets, washout=washout)

    test_states = reservoir.run(test_inputs, initial_state=reservoir.last_state)
    pred = readout.predict(test_states)

    return nrmse(pred, test_targets)


def main():
    all_results = []
    for noise_std in NOISE_LEVELS:
        print(f"\n=== noise_std = {noise_std} ===")
        train_inputs, train_targets, test_inputs, test_targets = dual_timescale_task(
            n_steps=2500, noise_std=noise_std, seed=SEED
        )

        print(f"{'configuration':<34}{'NRMSE slow':<14}{'NRMSE fast'}")
        print("-" * 60)
        for name, factory in CONFIGS.items():
            reservoir = factory(SEED)
            result = evaluate(reservoir, train_inputs, train_targets, test_inputs, test_targets)
            print(f"{name:<34}{result[0]:<14.4f}{result[1]:.4f}")
            all_results.append({
                "noise_std": noise_std,
                "configuration": name,
                "nrmse_slow": f"{result[0]:.4f}",
                "nrmse_fast": f"{result[1]:.4f}",
            })

    # Export results
    with open(f"{RESULTS_DIR}/exp10_optical_acoustic_coupling.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["noise_std", "configuration", "nrmse_slow", "nrmse_fast"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to {RESULTS_DIR}/exp10_optical_acoustic_coupling.csv")

    print()
    print("Note: same n_reservoir budget per stage (150) across configs, but")
    print("'both taps' has 300 readout features total vs 150 for the others --")
    print("that's an inherent asymmetry of giving the readout access to more")
    print("of the pipeline, not a free variable to equalize away.")


if __name__ == "__main__":
    main()