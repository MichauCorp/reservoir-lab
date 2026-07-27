"""
input -> {optical, acoustic (direct), acoustic (coupled to optical)} -> readout

exp10 asked whether a strict serial optical->acoustic pipeline could
recover both a slow and a fast target from noisy input, and found that
funnelling everything through one final readout on the acoustic stage
loses the fast-timescale information the optical stage had -- only
concatenating optical + coupled-acoustic states recovered it.

This experiment asks the natural follow-up: what's the theoretically
complete feature set, and how much does a regularized linear readout
actually lean on each branch? Three feature branches are available:

  1. optical           -- fast, nonlinear, driven by raw input
  2. acoustic_direct    -- slow memory, driven by raw input
  3. acoustic_coupled   -- slow memory, driven by the optical stage's output

Every non-empty combination of these three branches is evaluated, so
exp10's single- and two-branch results are reproduced here as a subset of
a complete 7-way comparison. On top of NRMSE, the full 3-branch readout's
own fitted weights are inspected directly (L2-norm share per branch) to
see which branch the regression actually relies on, rather than only
inferring it indirectly from NRMSE differences between subsets.
"""

import itertools

import numpy as np

from reservoir_lab.physical import AcousticReservoir, OptoelectronicReservoir
from reservoir_lab.readout import RidgeReadout

N_VIRTUAL = 150
N_OSCILLATORS = 150
WASHOUT = 150
TRAIN_FRAC = 0.7
NOISE_LEVELS = [0.15, 0.4, 0.8]
SEED = 42

BRANCHES = ["optical", "acoustic_direct", "acoustic_coupled"]


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


def build_branch_states(train_inputs, test_inputs, seed):
    """Run every reservoir once; reuse the resulting states across all subsets."""
    optical = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=N_VIRTUAL, seed=seed)
    acoustic_direct = AcousticReservoir(n_inputs=1, n_oscillators=N_OSCILLATORS, seed=seed + 1)
    acoustic_coupled = AcousticReservoir(n_inputs=N_VIRTUAL, n_oscillators=N_OSCILLATORS, seed=seed + 2)

    opt_train = optical.run(train_inputs)
    opt_test = optical.run(test_inputs, initial_state=optical.last_state)

    acd_train = acoustic_direct.run(train_inputs)
    acd_test = acoustic_direct.run(test_inputs, initial_state=acoustic_direct.last_state)

    acc_train = acoustic_coupled.run(opt_train)
    acc_test = acoustic_coupled.run(opt_test, initial_state=acoustic_coupled.last_state)

    return {
        "optical": (opt_train, opt_test),
        "acoustic_direct": (acd_train, acd_test),
        "acoustic_coupled": (acc_train, acc_test),
    }


def evaluate_subset(branch_states, subset, train_targets, test_targets):
    train_feats = np.hstack([branch_states[b][0] for b in subset])
    test_feats = np.hstack([branch_states[b][1] for b in subset])

    readout = RidgeReadout(reg=1e-3).fit(train_feats, train_targets, washout=WASHOUT)
    pred = readout.predict(test_feats)

    widths = [branch_states[b][0].shape[1] for b in subset]
    return nrmse(pred, test_targets), readout, widths


def branch_weight_share(readout, feature_widths, branch_names):
    """L2 norm of the fitted readout weights per branch, normalized to sum
    to 1 per output column -- 'how much does the readout lean on this
    branch', read directly off the trained weights."""
    Wout = readout.W_out
    norms = []
    start = 0
    for width in feature_widths:
        block = Wout[start:start + width]
        norms.append(np.linalg.norm(block, axis=0))
        start += width
    norms = np.array(norms)  # (n_branches, n_outputs)
    totals = norms.sum(axis=0, keepdims=True)
    totals[totals == 0] = 1.0
    return norms / totals


def main():
    subsets = []
    for r in range(1, len(BRANCHES) + 1):
        subsets.extend(itertools.combinations(BRANCHES, r))

    for noise_std in NOISE_LEVELS:
        print(f"\n=== noise_std = {noise_std} ===")
        train_inputs, train_targets, test_inputs, test_targets = dual_timescale_task(
            n_steps=2500, noise_std=noise_std, seed=SEED
        )
        branch_states = build_branch_states(train_inputs, test_inputs, SEED)

        print(f"{'feature branches':<48}{'NRMSE slow':<14}{'NRMSE fast'}")
        print("-" * 74)
        full_readout = None
        for subset in subsets:
            result, readout, widths = evaluate_subset(branch_states, subset, train_targets, test_targets)
            label = " + ".join(subset)
            print(f"{label:<48}{result[0]:<14.4f}{result[1]:.4f}")
            if len(subset) == len(BRANCHES):
                full_readout = (readout, widths, subset)

        readout, widths, subset = full_readout
        shares = branch_weight_share(readout, widths, subset)
        print()
        print("Readout weight share on the full 3-branch model (columns sum to 1):")
        print(f"{'branch':<20}{'slow':<10}{'fast'}")
        for name, row in zip(subset, shares):
            print(f"{name:<20}{row[0]:<10.3f}{row[1]:.3f}")

    print()
    print("Note: acoustic_direct and acoustic_coupled are separate instances of")
    print("the same AcousticReservoir class (different seeds) -- one driven by")
    print("raw input, one driven by the optical stage's states.")


if __name__ == "__main__":
    main()
