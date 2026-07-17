import numpy as np

class ESN:
    def __init__(self, n_inputs, n_reservoir, spectral_radius=0.9, sparsity=0.1, seed=42):
        rng = np.random.default_rng(seed)
        self.n_reservoir = n_reservoir

        # Reservoir weights: sparse, scaled to spectral radius
        W = rng.uniform(-1, 1, (n_reservoir, n_reservoir))
        W[rng.random((n_reservoir, n_reservoir)) > sparsity] = 0
        rho = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (spectral_radius / rho)

        self.W_in = rng.uniform(-1, 1, (n_reservoir, n_inputs))
        self.W_out = None

    def run(self, inputs):
        states = np.zeros((len(inputs), self.n_reservoir))
        state = np.zeros(self.n_reservoir)
        for t, u in enumerate(inputs):
            state = np.tanh(self.W_in @ u + self.W @ state)
            states[t] = state
        return states

    def fit(self, inputs, targets, reg=1e-6):
        states = self.run(inputs)
        self.W_out = np.linalg.solve(
            states.T @ states + reg * np.eye(self.n_reservoir),
            states.T @ targets
        )
        return states

    def predict(self, inputs):
        return self.run(inputs) @ self.W_out