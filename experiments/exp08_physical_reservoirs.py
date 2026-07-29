"""
Compare the software ESN against the two simulated physical reservoirs
(optoelectronic delay-line, acoustic oscillator lattice) on the same task,
using the same next-step sine prediction task as exp02.

This is meant as a first sanity check that the physical reservoir
simulations behave like reservoirs at all (beat a trivial baseline, and
generalize to held-out data) before comparing them on harder tasks
(Mackey-Glass, NARMA10, memory capacity, etc).
"""

import csv
import numpy as np

from reservoir_lab.reservoir import ESN
from reservoir_lab.readout import RidgeReadout
from reservoir_lab.physical import OptoelectronicReservoir, AcousticReservoir

RESULTS_DIR = "experiments/data_results"
VISUALS_DIR = "experiments/visuals"


# 1. Data (same as exp02)

t = np.linspace(0, 100, 3000)
data = np.sin(t).reshape(-1, 1)

split = 2000
train_data, test_data = data[:split], data[split:]

train_inputs, train_targets = train_data[:-1], train_data[1:]
test_inputs, test_targets = test_data[:-1], test_data[1:]


def evaluate(name, reservoir, washout=100):
    states = reservoir.run(train_inputs)
    readout = RidgeReadout(reg=1e-6).fit(states, train_targets, washout=washout)

    # continue from the training state instead of resetting to zero
    test_states = reservoir.run(test_inputs, initial_state=reservoir.last_state)
    pred = readout.predict(test_states)

    mse = np.mean((pred - test_targets) ** 2)
    print(f"{name:>15s}  n_reservoir={reservoir.n_reservoir:<5d}  test MSE={mse:.6f}")
    return name, reservoir.n_reservoir, mse


print("Sine-wave generalization: software ESN vs. simulated physical reservoirs")
print("-" * 74)

results = [
    evaluate("ESN", ESN(n_inputs=1, n_reservoir=200, spectral_radius=0.9, sparsity=0.1)),
    evaluate("Optoelectronic", OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=200)),
    evaluate("Acoustic", AcousticReservoir(n_inputs=1, n_oscillators=200)),
]

# Export results
with open(f"{RESULTS_DIR}/exp08_physical_reservoirs.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "n_reservoir", "test_mse"])
    for name, n_res, mse in results:
        writer.writerow([name, n_res, f"{mse:.6f}"])
print(f"Results saved to {RESULTS_DIR}/exp08_physical_reservoirs.csv")
