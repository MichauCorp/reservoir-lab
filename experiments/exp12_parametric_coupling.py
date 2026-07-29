"""
Does real parametric coupling (optical modulates acoustic damping) do
anything the earlier input-driven coupling (exp10, exp11) didn't?

exp11 found that feeding the optical stage's output into the acoustic
stage as an *input* barely changed anything -- the readout couldn't
tell that branch apart from acoustic driven directly by raw input.
ParametricallyCoupledReservoir instead lets the optical stage modulate
the acoustic lattice's damping coefficient in real time, which is the
one coupling mechanism theoretically capable of producing genuine fast
x slow interaction features (see physical/parametric_coupling.py).

Configurations compared, same task and reservoir sizes as exp10/exp11:
  - optical only
  - acoustic_direct only
  - acoustic_input_coupled       (exp10/11's coupling: optical -> acoustic input)
  - acoustic_param_coupled       (new: optical -> acoustic damping)
  - optical + acoustic_direct    (best combo found in exp11)
  - optical + acoustic_param_coupled
"""

import csv
import numpy as np

from reservoir_lab.physical import (
    AcousticReservoir,
    OptoelectronicReservoir,
    ParametricallyCoupledReservoir,
)
from reservoir_lab.readout import RidgeReadout

RESULTS_DIR = "experiments/data_results"
VISUALS_DIR = "experiments/visuals"

N_VIRTUAL = 150
N_OSCILLATORS = 150
WASHOUT = 150
TRAIN_FRAC = 0.7
NOISE_LEVELS = [0.15, 0.4, 0.8]
SEED = 42


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


def nrmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2, axis=0)) / np.std(true, axis=0)


def check_finite(name, states):
    if not np.all(np.isfinite(states)):
        raise FloatingPointError(
            f"{name} produced non-finite states -- damping modulation likely "
            f"went unstable. Lower c_mod_gain or raise the damping floor."
        )


def evaluate(states_train, states_test, train_targets, test_targets):
    readout = RidgeReadout(reg=1e-3).fit(states_train, train_targets, washout=WASHOUT)
    pred = readout.predict(states_test)
    return nrmse(pred, test_targets)


def main():
    all_results = []
    for noise_std in NOISE_LEVELS:
        print(f"\n=== noise_std = {noise_std} ===")
        train_inputs, train_targets, test_inputs, test_targets = dual_timescale_task(
            n_steps=2500, noise_std=noise_std, seed=SEED
        )

        optical = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=N_VIRTUAL, seed=SEED)
        acoustic_direct = AcousticReservoir(n_inputs=1, n_oscillators=N_OSCILLATORS, seed=SEED + 1)
        param_modulator = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=N_VIRTUAL, seed=SEED + 2)
        acoustic_param = ParametricallyCoupledReservoir(
            modulator=param_modulator, n_inputs=1, n_oscillators=N_OSCILLATORS, seed=SEED + 3
        )

        opt_train = optical.run(train_inputs)
        opt_test = optical.run(test_inputs, initial_state=optical.last_state)
        check_finite("optical", opt_train)

        acd_train = acoustic_direct.run(train_inputs)
        acd_test = acoustic_direct.run(test_inputs, initial_state=acoustic_direct.last_state)
        check_finite("acoustic_direct", acd_train)

        acp_train = acoustic_param.run(train_inputs)
        acp_test = acoustic_param.run(test_inputs, initial_state=acoustic_param.last_state)
        check_finite("acoustic_param_coupled", acp_train)

        configs = {
            "optical only":                       (opt_train, opt_test),
            "acoustic_direct only":                (acd_train, acd_test),
            "acoustic_param_coupled only":          (acp_train, acp_test),
            "optical + acoustic_direct":            (np.hstack([opt_train, acd_train]), np.hstack([opt_test, acd_test])),
            "optical + acoustic_param_coupled":     (np.hstack([opt_train, acp_train]), np.hstack([opt_test, acp_test])),
        }

        print(f"{'configuration':<38}{'NRMSE slow':<14}{'NRMSE fast'}")
        print("-" * 64)
        for name, (states_train, states_test) in configs.items():
            result = evaluate(states_train, states_test, train_targets, test_targets)
            print(f"{name:<38}{result[0]:<14.4f}{result[1]:.4f}")
            all_results.append({
                "noise_std": noise_std,
                "configuration": name,
                "nrmse_slow": f"{result[0]:.4f}",
                "nrmse_fast": f"{result[1]:.4f}",
            })

    # Export results
    with open(f"{RESULTS_DIR}/exp12_parametric_coupling.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["noise_std", "configuration", "nrmse_slow", "nrmse_fast"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to {RESULTS_DIR}/exp12_parametric_coupling.csv")

    print()
    print("Compare 'optical + acoustic_param_coupled' against exp11's best,")
    print("'optical + acoustic_direct' -- if parametric coupling is adding a")
    print("genuine interaction feature, it should beat that baseline, not just")
    print("match it.")


if __name__ == "__main__":
    main()