import numpy as np

from reservoir_lab.reservoir import ESN
from reservoir_lab.visualize import plot_results


# Generate Mackey-Glass chaotic time series
def generate_mackey_glass(length, tau=17):

    beta = 0.2
    gamma = 0.1
    n = 10

    data = np.ones(length + tau + 1) * 1.2

    for t in range(tau, length + tau):
        delayed = data[t - tau]

        data[t + 1] = (
            data[t]
            +
            beta * delayed / (1 + delayed**n)
            -
            gamma * data[t]
        )

    return data[tau:].reshape(-1, 1)


# 1. Generate data
data = generate_mackey_glass(6000)


# 2. Split train/test
train_size = 4000

train = data[:train_size]
test = data[train_size:]


# 3. Create next-step prediction task

train_inputs = train[:-1]
train_targets = train[1:]

test_inputs = test[:-1]
test_targets = test[1:]


# 4. Create reservoir

esn = ESN(
    n_inputs=1,
    n_reservoir=500,
    spectral_radius=0.9,
    sparsity=0.05
)


# 5. Train readout (discard the initial transient)

states = esn.fit(
    train_inputs,
    train_targets,
    washout=100
)


# 6. Predict unseen data — continue reservoir state across the train/test boundary

test_states = esn.run(test_inputs, initial_state=esn.last_state)
pred = test_states @ esn.W_out


# 7. Calculate error

mse = np.mean(
    (pred - test_targets) ** 2
)

print("MSE:", mse)


# 8. Plot results

plot_results(
    test_inputs,
    test_targets,
    pred,
    test_states,
    title="Mackey-Glass prediction",
    save_path="experiments/visuals/mackey_glass.png"
)