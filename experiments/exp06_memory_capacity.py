import numpy as np

from reservoir_lab.reservoir import ESN


# -----------------------------
# Generate random input
# -----------------------------

np.random.seed(42)

timesteps = 10000

data = np.random.randint(
    0,
    2,
    timesteps
).reshape(-1, 1)


# -----------------------------
# Create reservoir
# -----------------------------

esn = ESN(
    n_inputs=1,
    n_reservoir=1000,
    spectral_radius=1.25,
    sparsity=0.1
)


# -----------------------------
# Run reservoir once
# -----------------------------

states = esn.run(data)


# -----------------------------
# Remove initial transient
# -----------------------------

washout = 500

states = states[washout:]
data = data[washout:]


# -----------------------------
# Test different memory delays
# -----------------------------

delays = [
    1,
    2,
    5,
    10,
    20,
    50,
    100
]


results = []


for delay in delays:

    print(f"Testing delay {delay}")


    # Current reservoir state
    X = states[delay:]


    # Information we want to recover
    y = data[:-delay]


    # Train/test split

    split = int(len(X) * 0.7)

    X_train = X[:split]
    y_train = y[:split]

    X_test = X[split:]
    y_test = y[split:]


    # Train linear readout manually

    reg = 1e-6

    W_out = np.linalg.solve(
        X_train.T @ X_train
        + reg * np.eye(X_train.shape[1]),

        X_train.T @ y_train
    )


    # Predict

    pred = X_test @ W_out


    mse = np.mean(
        (pred - y_test) ** 2
    )


    results.append(mse)


    print(
        f"delay={delay}, MSE={mse:.6f}"
    )


print()
print("===================")
print("Memory Results")
print("===================")


for delay, mse in zip(delays, results):

    print(
        f"Delay {delay}: {mse:.6f}"
    )