import csv
import numpy as np

from reservoir_lab.reservoir import ESN
from reservoir_lab.visualize import plot_results

RESULTS_DIR = "experiments/data_results"
VISUALS_DIR = "experiments/visuals"


# Generate NARMA10 data

def generate_narma10(length, seed=42):

    rng = np.random.default_rng(seed)

    u = rng.random((length, 1)) * 0.5
    y = np.zeros((length, 1))

    for t in range(10, length-1):

        y[t+1] = (
            0.3 * y[t]
            +
            0.05 * y[t] * np.sum(y[t-10:t])
            +
            1.5 * u[t-9] * u[t]
            +
            0.1
        )

    return u, y


# 1. Generate data

inputs, targets = generate_narma10(5000)


# 2. Split train/test

train_size = 4000

train_inputs = inputs[:train_size]
train_targets = targets[:train_size]

test_inputs = inputs[train_size:]
test_targets = targets[train_size:]


# 3. Create ESN

esn = ESN(
    n_inputs=1,
    n_reservoir=300,
    spectral_radius=0.9,
    sparsity=0.1
)


# 4. Train (discard the initial transient)

states = esn.fit(
    train_inputs,
    train_targets,
    washout=100
)


# 5. Predict unseen data — continue reservoir state from training

test_states = esn.run(test_inputs, initial_state=esn.last_state)
pred = test_states @ esn.W_out


# 6. Measure error

mse = np.mean(
    (pred - test_targets)**2
)

print(f"MSE: {mse:.6f}")

# Export results
with open(f"{RESULTS_DIR}/exp04_NARMA10.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    writer.writerow(["mse", f"{mse:.6f}"])
print(f"Results saved to {RESULTS_DIR}/exp04_NARMA10.csv")


# 7. Plot

plot_results(
    test_inputs,
    test_targets,
    pred,
    test_states,
    title="NARMA10 prediction",
    save_path=f"{VISUALS_DIR}/exp04_narma10.png"
)
