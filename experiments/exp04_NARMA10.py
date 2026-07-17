import numpy as np

from reservoir_lab.reservoir import ESN
from reservoir_lab.visualize import plot_results


# Generate NARMA10 data

def generate_narma10(length):

    u = np.random.rand(length, 1) * 0.5
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


# 4. Train

states = esn.fit(
    train_inputs,
    train_targets
)


# 5. Predict unseen data

pred = esn.predict(test_inputs)


# 6. Measure error

mse = np.mean(
    (pred - test_targets)**2
)

print("MSE:", mse)


# 7. Plot

plot_results(
    test_inputs,
    test_targets,
    pred,
    states,
    title="NARMA10 prediction",
    save_path="experiments/visuals/narma10.png"
)