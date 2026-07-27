"""
Architecture x task-interaction x timescale-separation sweep.

exp10-exp12 held the task fixed (recover two *independently* superimposed
sine waves) and varied only the coupling mechanism. That's a confound:
if the task has no genuine fast/slow interaction to exploit, no coupling
mechanism can look better than simple concatenation, regardless of how
good it is. This sweep varies the task's interaction level as a first-
class axis alongside architecture, so "coupling doesn't help" and "this
task doesn't need coupling" can no longer be mistaken for each other.

Axes:
  architecture         -- 5 ways of combining optical + acoustic (below)
  task interaction      -- 0.0 (fully additive, exp10-12's task),
                            0.5, 1.0 (fast target fully gated by slow phase)
  timescale separation  -- "separated" (optical fast / acoustic slow,
                            exp10-12's defaults) vs "similar" (both stages
                            tuned to comparable timescales) -- tests
                            whether any coupling benefit depends on the
                            two stages actually being different, as the
                            deep-ESN literature's own ablations found.
  seed                  -- 4 seeds per cell, mean +/- std reported.

Architectures:
  parallel             -- optical + acoustic, both driven directly by raw
                          input, independently. No coupling. The floor.
  serial_force         -- optical -> acoustic (driving force), readout
                          taps both stages. exp10's winning config.
  parametric           -- acoustic driven by raw input AND by optical's
                          damping modulation simultaneously. exp12.
  parallel_bilinear    -- same physical setup as `parallel`, but the
                          readout ALSO gets an explicit product feature
                          (optical activity x acoustic state vector).
                          Not physically coupled at all -- this is a
                          non-physical ceiling: if this doesn't win on
                          the high-interaction task either, the task
                          itself is the problem, not any physical
                          coupling mechanism's ability to realize it.
  mutual               -- true bidirectional coupling: optical modulates
                          acoustic's damping (as in parametric) AND
                          acoustic's *previous* state nudges optical's
                          operating point, with a one-step causal delay
                          to keep the simulation well-posed.
"""

import csv
import itertools

import numpy as np

from reservoir_lab.physical import (
    AcousticReservoir,
    MutualCoupledReservoir,
    OptoelectronicReservoir,
    ParametricallyCoupledReservoir,
    SerialPhysicalReservoir,
)
from reservoir_lab.readout import RidgeReadout

N_STEPS = 1400
WASHOUT = 120
TRAIN_FRAC = 0.7
N_VIRTUAL = 90
N_OSCILLATORS = 90
SEEDS = [1, 2, 3, 4]
INTERACTION_LEVELS = [0.0, 0.5, 1.0]
TIMESCALES = {
    "separated": {"optical_theta": 0.2, "acoustic_c": 0.3},
    "similar": {"optical_theta": 2.0, "acoustic_c": 3.0},
}


# =====================================================================
# Task: slow + fast, with fast optionally gated by slow's phase
# =====================================================================

def dual_timescale_task(n_steps, interaction, noise_std=0.35, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)
    slow = 0.6 * np.sin(2 * np.pi * t / 180.0)
    fast_raw = 0.3 * np.sin(2 * np.pi * t / 12.0 + 0.6 * np.sin(2 * np.pi * t / 33.0))
    noise = rng.normal(0, noise_std, n_steps)

    gate = 0.5 * (1 + np.tanh(4 * slow))          # in (0, 1), tracks slow's phase
    fast_target = fast_raw * ((1 - interaction) + interaction * gate)

    u = (slow + fast_raw + noise).reshape(-1, 1)   # what the reservoirs see
    y = np.stack([slow, fast_target], axis=1)      # what the readout must recover

    split = int(n_steps * TRAIN_FRAC)
    return u[:split], y[:split], u[split:], y[split:]


def nrmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2, axis=0)) / np.std(true, axis=0)


# =====================================================================
# Per-architecture feature builders. Each returns (train_feats, test_feats).
# =====================================================================

def build_parallel(u_train, u_test, seed, ts):
    optical = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=N_VIRTUAL, theta=ts["optical_theta"], seed=seed)
    acoustic = AcousticReservoir(n_inputs=1, n_oscillators=N_OSCILLATORS, c=ts["acoustic_c"], seed=seed + 1)

    opt_tr = optical.run(u_train)
    ac_tr = acoustic.run(u_train)
    opt_te = optical.run(u_test, initial_state=optical.last_state)
    ac_te = acoustic.run(u_test, initial_state=acoustic.last_state)

    return np.hstack([opt_tr, ac_tr]), np.hstack([opt_te, ac_te])


def build_serial_force(u_train, u_test, seed, ts):
    stage1 = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=N_VIRTUAL, theta=ts["optical_theta"], seed=seed)
    stage2 = AcousticReservoir(n_inputs=N_VIRTUAL, n_oscillators=N_OSCILLATORS, c=ts["acoustic_c"], seed=seed + 1)
    model = SerialPhysicalReservoir(stage1, stage2, combine="both", seed=seed)

    train_feats = model.run(u_train)
    test_feats = model.run(u_test, initial_state=model.last_state)
    return train_feats, test_feats


def build_parametric(u_train, u_test, seed, ts):
    modulator = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=N_VIRTUAL, theta=ts["optical_theta"], seed=seed)
    lattice = ParametricallyCoupledReservoir(
        modulator=modulator, n_inputs=1, n_oscillators=N_OSCILLATORS, c_base=ts["acoustic_c"], seed=seed + 1
    )

    ac_tr = lattice.run(u_train)
    opt_tr = modulator.run(u_train, initial_state=None)  # re-run standalone for readout tap
    ac_te = lattice.run(u_test, initial_state=lattice.last_state)
    opt_te = modulator.run(u_test, initial_state=modulator.last_state)

    return np.hstack([opt_tr, ac_tr]), np.hstack([opt_te, ac_te])


def build_parallel_bilinear(u_train, u_test, seed, ts):
    optical = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=N_VIRTUAL, theta=ts["optical_theta"], seed=seed)
    acoustic = AcousticReservoir(n_inputs=1, n_oscillators=N_OSCILLATORS, c=ts["acoustic_c"], seed=seed + 1)

    def feats(u, opt_init, ac_init):
        opt = optical.run(u, initial_state=opt_init)
        ac = acoustic.run(u, initial_state=ac_init)
        opt_activity = np.tanh(2 * opt.mean(axis=1) - 1).reshape(-1, 1)
        interaction = opt_activity * ac  # explicit product feature, shape (T, n_oscillators)
        return np.hstack([opt, ac, interaction])

    train_feats = feats(u_train, None, None)
    test_feats = feats(u_test, optical.last_state, acoustic.last_state)
    return train_feats, test_feats


def build_mutual(u_train, u_test, seed, ts):
    model = MutualCoupledReservoir(
        n_inputs=1, n_virtual_nodes=N_VIRTUAL, n_oscillators=N_OSCILLATORS,
        theta=ts["optical_theta"], c_base=ts["acoustic_c"], seed=seed,
    )
    train_feats = model.run(u_train)
    test_feats = model.run(u_test, initial_state=model.last_state)
    return train_feats, test_feats


ARCHITECTURES = {
    "parallel": build_parallel,
    "serial_force": build_serial_force,
    "parametric": build_parametric,
    "parallel_bilinear": build_parallel_bilinear,
    "mutual": build_mutual,
}


def evaluate(builder, u_train, y_train, u_test, y_test, seed, ts):
    train_feats, test_feats = builder(u_train, u_test, seed, ts)
    readout = RidgeReadout(reg=1e-3).fit(train_feats, y_train, washout=WASHOUT)
    pred = readout.predict(test_feats)
    return nrmse(pred, y_test)


def main():
    results = []
    combos = list(itertools.product(ARCHITECTURES, INTERACTION_LEVELS, TIMESCALES, SEEDS))
    print(f"Running {len(combos)} configurations...")

    for arch_name, interaction, ts_name, seed in combos:
        u_train, y_train, u_test, y_test = dual_timescale_task(N_STEPS, interaction, seed=seed)
        try:
            result = evaluate(ARCHITECTURES[arch_name], u_train, y_train, u_test, y_test, seed, TIMESCALES[ts_name])
            results.append({
                "architecture": arch_name, "interaction": interaction, "timescale": ts_name,
                "seed": seed, "nrmse_slow": result[0], "nrmse_fast": result[1],
            })
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"  FAILED: {arch_name}, interaction={interaction}, {ts_name}, seed={seed}: {e}")

    with open("experiments/exp13_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["architecture", "interaction", "timescale", "seed", "nrmse_slow", "nrmse_fast"])
        writer.writeheader()
        writer.writerows(results)
    print("Raw results written to experiments/exp13_results.csv\n")

    # Aggregate: mean +/- std across seeds, per (architecture, interaction, timescale)
    print(f"{'architecture':<20}{'interaction':<13}{'timescale':<12}{'fast NRMSE (mean+-std)'}")
    print("-" * 78)
    for ts_name in TIMESCALES:
        for interaction in INTERACTION_LEVELS:
            for arch_name in ARCHITECTURES:
                cell = [r["nrmse_fast"] for r in results
                        if r["architecture"] == arch_name and r["interaction"] == interaction and r["timescale"] == ts_name]
                if not cell:
                    continue
                print(f"{arch_name:<20}{interaction:<13}{ts_name:<12}{np.mean(cell):.4f} +- {np.std(cell):.4f}")
        print()


if __name__ == "__main__":
    main()