import csv
import numpy as np # numerical arrays
from reservoir_lab.reservoir import ESN # the class defined above
from reservoir_lab.visualize import plot_results # visualization

RESULTS_DIR = "experiments/data_results"
VISUALS_DIR = "experiments/visuals"

t = np.linspace(0, 100, 2000)                           # 2000 evenly spaced time points from 0 to 100
data = np.sin(t).reshape(-1, 1)                         # sin(t) at each point; reshaped to column -> shape (2000, 1)

inputs, targets = data[:-1], data[1:]                   # inputs = data[0..1998], targets = data[1..1999]
                                                         # i.e. "given this value, predict the NEXT value"

esn = ESN(n_inputs=1, n_reservoir=200)                  # build a 200-neuron reservoir, 1 number in per timestep
states = esn.fit(inputs, targets)                       # train the readout to map reservoir states -> targets

pred = esn.predict(inputs)                              # generate predictions for the same inputs
mse = np.mean((pred - targets) ** 2)                    # average squared error across all 1999 predictions
plot_results(inputs, targets, pred, states, title="Baseline sine-wave prediction", save_path=f"{VISUALS_DIR}/exp01_baseline.png")
print(f"MSE: {mse:.6f}")

# Export results
with open(f"{RESULTS_DIR}/exp01_baseline.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    writer.writerow(["mse", f"{mse:.6f}"])
print(f"Results saved to {RESULTS_DIR}/exp01_baseline.csv")
