import numpy as np

from reservoir_lab.reservoir import ESN
from reservoir_lab.visualize import plot_results


# 1. Generate data
t = np.linspace(0, 100, 3000)

data = np.sin(t).reshape(-1, 1)


# 2. Split into train and test
split = 2000

train_data = data[:split]
test_data = data[split:]


# 3. Create next-step prediction task

train_inputs = train_data[:-1]
train_targets = train_data[1:]

test_inputs = test_data[:-1]
test_targets = test_data[1:]


# 4. Create ESN

esn = ESN(
    n_inputs=1,
    n_reservoir=200,
    spectral_radius=0.9,
    sparsity=0.1
)


# 5. Train readout only on training data

states = esn.fit(
    train_inputs,
    train_targets
)


# 6. Predict unseen data

pred = esn.predict(test_inputs)


# 7. Evaluate

mse = np.mean((pred - test_targets)**2)

print(f"Test MSE: {mse}")


# 8. Visualize

plot_results(
    test_inputs,
    test_targets,
    pred,
    states,
    title="Sine wave generalization",
    save_path="experiments/visuals/sine_generalization.png"
)