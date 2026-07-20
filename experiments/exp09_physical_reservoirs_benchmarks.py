"""
Benchmark comparison: software ESN vs. simulated optoelectronic reservoir
vs. simulated acoustic reservoir, across a battery of standard reservoir
computing tasks.

Tasks (increasing structural demand on the reservoir):
  1. Sine wave        - smooth, low-frequency, minimal memory needed
  2. NARMA10          - nonlinear, needs ~10 steps of memory
  3. Mackey-Glass      - chaotic, needs both memory and rich nonlinearity
  4. Memory capacity   - measures pure linear memory depth directly

Expectation going in: the ESN should generally come out on top, because
its reservoir matrix is an unconstrained random NxN system -- every one of
its N states can be an arbitrary nonlinear combination of the recurrent
history. The physical reservoirs are more constrained by construction:

  - Optoelectronic: all N "virtual node" states come from ONE scalar delay
    trajectory, so they're inherently more correlated with each other than
    N independent ESN neurons are -- less usable diversity per neuron.
  - Acoustic: coupling is local (nearest-neighbor only), so information
    about the input has to physically propagate across the lattice --
    a structural bottleneck the ESN's fully-connected matrix doesn't have.

That said, "generally best" is not "always best" -- these results depend
heavily on the (untuned, default) hyperparameters each reservoir is using.
A fairer fight would sweep spectral_radius for the ESN, eta/theta for the
optoelectronic model, and damping/coupling for the acoustic model, same as
suggested in exp09's follow-up. Treat this script as the baseline that
motivates that sweep, not the final word.
"""

import numpy as np

from reservoir_lab.reservoir import ESN
from reservoir_lab.readout import RidgeReadout
from reservoir_lab.physical import OptoelectronicReservoir, AcousticReservoir


# =====================================================================
# Reservoir factories -- one common signature (n_inputs, n_reservoir,
# seed) so every task/benchmark can build any reservoir type identically.
# =====================================================================

def make_esn(n_inputs, n_reservoir, seed):
    return ESN(
        n_inputs=n_inputs,
        n_reservoir=n_reservoir,
        spectral_radius=0.9,
        sparsity=0.1,
        seed=seed,
    )


def make_optical(n_inputs, n_reservoir, seed):
    return OptoelectronicReservoir(
        n_inputs=n_inputs,
        n_virtual_nodes=n_reservoir,
        seed=seed,
    )


def make_acoustic(n_inputs, n_reservoir, seed):
    return AcousticReservoir(
        n_inputs=n_inputs,
        n_oscillators=n_reservoir,
        seed=seed,
    )


RESERVOIR_TYPES = {
    "ESN": make_esn,
    "Optoelectronic": make_optical,
    "Acoustic": make_acoustic,
}


# =====================================================================
# Task data generators
# =====================================================================

def sine_task():
    t = np.linspace(0, 100, 3000)
    data = np.sin(t).reshape(-1, 1)

    split = 2000
    train, test = data[:split], data[split:]

    return train[:-1], train[1:], test[:-1], test[1:]


def narma10_task(length=5000, train_size=4000, seed=42):
    rng = np.random.default_rng(seed)

    u = rng.random((length, 1)) * 0.5
    y = np.zeros((length, 1))

    for t in range(10, length - 1):
        y[t + 1] = (
            0.3 * y[t]
            + 0.05 * y[t] * np.sum(y[t - 10:t])
            + 1.5 * u[t - 9] * u[t]
            + 0.1
        )

    train_inputs, train_targets = u[:train_size], y[:train_size]
    test_inputs, test_targets = u[train_size:], y[train_size:]

    return train_inputs, train_targets, test_inputs, test_targets


def mackey_glass_task(length=6000, train_size=4000, tau=17):
    beta, gamma, n = 0.2, 0.1, 10

    data = np.ones(length + tau + 1) * 1.2
    for t in range(tau, length + tau):
        delayed = data[t - tau]
        data[t + 1] = data[t] + beta * delayed / (1 + delayed**n) - gamma * data[t]

    data = data[tau:].reshape(-1, 1)

    train, test = data[:train_size], data[train_size:]

    return train[:-1], train[1:], test[:-1], test[1:]


TASKS = {
    "Sine": (sine_task, dict(n_reservoir=200, washout=50)),
    "NARMA10": (narma10_task, dict(n_reservoir=300, washout=100)),
    "Mackey-Glass": (mackey_glass_task, dict(n_reservoir=500, washout=100)),
}


# =====================================================================
# Prediction benchmark runner
# =====================================================================

def run_prediction_benchmark(reservoir_factory, n_inputs, n_reservoir, seed,
                              train_inputs, train_targets,
                              test_inputs, test_targets, washout):
    reservoir = reservoir_factory(n_inputs=n_inputs, n_reservoir=n_reservoir, seed=seed)

    states = reservoir.run(train_inputs)
    readout = RidgeReadout(reg=1e-6).fit(states, train_targets, washout=washout)

    # continue from wherever training left off, not a cold zero state
    test_states = reservoir.run(test_inputs, initial_state=reservoir.last_state)
    pred = readout.predict(test_states)

    return np.mean((pred - test_targets) ** 2)


# =====================================================================
# Memory capacity benchmark
#
# Uses the standard Jaeger (2001) definition: for each delay d, fit a
# linear readout that reconstructs input(t-d) from state(t), and score it
# with the squared correlation coefficient r^2 between reconstruction and
# ground truth (bounded in [0, 1], unlike raw MSE, so it's comparable
# across delays and across reservoir types). Total memory capacity is the
# sum of r^2 over all tested delays.
# =====================================================================

def r_squared(pred, true):
    pred, true = pred.flatten(), true.flatten()
    if np.std(pred) == 0 or np.std(true) == 0:
        return 0.0
    corr = np.corrcoef(pred, true)[0, 1]
    return corr ** 2


def memory_capacity(reservoir_factory, n_reservoir, seed,
                     timesteps=6000, washout=500,
                     delays=(1, 2, 5, 10, 20, 50, 100)):
    rng = np.random.default_rng(seed)
    data = rng.uniform(-1, 1, (timesteps, 1))

    reservoir = reservoir_factory(n_inputs=1, n_reservoir=n_reservoir, seed=seed)
    states = reservoir.run(data)

    states = states[washout:]
    data = data[washout:]

    total_mc = 0.0
    per_delay = {}

    for delay in delays:
        X = states[delay:]
        y = data[:-delay]

        split = int(len(X) * 0.7)
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        readout = RidgeReadout(reg=1e-6).fit(X_train, y_train)
        pred = readout.predict(X_test)

        score = r_squared(pred, y_test)
        per_delay[delay] = score
        total_mc += score

    return total_mc, per_delay


# =====================================================================
# Run everything
# =====================================================================

def main():
    seed = 42

    print("=" * 78)
    print("PREDICTION BENCHMARKS (test MSE, lower is better)")
    print("=" * 78)
    print(f"{'Task':<15}{'ESN':>16}{'Optoelectronic':>20}{'Acoustic':>16}")
    print("-" * 78)

    for task_name, (task_fn, cfg) in TASKS.items():
        train_inputs, train_targets, test_inputs, test_targets = task_fn()
        n_inputs = train_inputs.shape[1]

        row = [task_name]
        for res_name, factory in RESERVOIR_TYPES.items():
            mse = run_prediction_benchmark(
                factory, n_inputs, cfg["n_reservoir"], seed,
                train_inputs, train_targets, test_inputs, test_targets,
                cfg["washout"],
            )
            row.append(mse)

        print(f"{row[0]:<15}{row[1]:>16.6f}{row[2]:>20.6f}{row[3]:>16.6f}")

    print()
    print("=" * 78)
    print("MEMORY CAPACITY (total r^2 summed over delays [1,2,5,10,20,50,100],")
    print("higher is better -- theoretical max is roughly n_reservoir)")
    print("=" * 78)
    print(f"{'Reservoir':<20}{'Total MC':>15}")
    print("-" * 78)

    mc_results = {}
    for res_name, factory in RESERVOIR_TYPES.items():
        total_mc, per_delay = memory_capacity(factory, n_reservoir=200, seed=seed)
        mc_results[res_name] = (total_mc, per_delay)
        print(f"{res_name:<20}{total_mc:>15.3f}")

    print()
    print("Per-delay breakdown:")
    delays = list(next(iter(mc_results.values()))[1].keys())
    header = "Delay  " + "".join(f"{d:>10}" for d in delays)
    print(header)
    for res_name, (total_mc, per_delay) in mc_results.items():
        row = f"{res_name:<20}" + "".join(f"{per_delay[d]:>10.3f}" for d in delays)
        print(row)

    print()
    print("Note: all reservoirs used default/untuned hyperparameters and")
    print("matched n_reservoir counts per task. This shows relative behavior")
    print("under a naive equal-size comparison, not each architecture's best")
    print("achievable performance -- see exp10 (parameter sweep) for that.")


if __name__ == "__main__":
    main()