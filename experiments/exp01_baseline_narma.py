import numpy as np
from reservoir_lab.reservoir import ESN

# Simple sine-wave prediction as a smoke test
t = np.linspace(0, 100, 2000)
data = np.sin(t).reshape(-1, 1)

inputs, targets = data[:-1], data[1:]
esn = ESN(n_inputs=1, n_reservoir=200)
esn.fit(inputs, targets)
pred = esn.predict(inputs)

mse = np.mean((pred - targets) ** 2)
print(f"MSE: {mse:.6f}")