import numpy as np                                          # numerical arrays and linear algebra

class ESN:
    def __init__(self, n_inputs, n_reservoir, spectral_radius=0.9, sparsity=0.1, seed=42):
        # n_inputs: how many numbers arrive per timestep (fixes W_in's shape, permanently)
        # n_reservoir: how many "neurons" the reservoir has (more = more memory capacity)
        # spectral_radius: target stability level for the reservoir (see below)
        # sparsity: fraction of reservoir connections that stay non-zero
        # seed: makes the "random" matrices reproducible across runs

        rng = np.random.default_rng(seed)                    # seeded random generator
        self.n_reservoir = n_reservoir                        # stored for later use in run()/fit()

        W = rng.uniform(-1, 1, (n_reservoir, n_reservoir))    # dense random reservoir matrix, values in [-1, 1]
        W[rng.random((n_reservoir, n_reservoir)) > sparsity] = 0   # zero out ~90% of entries -> sparse connectivity

        rho = np.max(np.abs(np.linalg.eigvals(W)))            # compute W's natural gain (spectral radius)
        self.W = W * (spectral_radius / rho)                  # rescale so gain exactly equals target -> stability

        self.W_in = rng.uniform(-1, 1, (n_reservoir, n_inputs))  # random fixed weights: input -> reservoir
        self.W_out = None                                     # not trained yet; filled in by fit()

    def run(self, inputs):
        # inputs: shape (n_timesteps, n_inputs) -- the data being fed through the reservoir
        states = np.zeros((len(inputs), self.n_reservoir))    # preallocate: one row per timestep, one column per neuron
        state = np.zeros(self.n_reservoir)                    # reservoir starts with zero activity (no memory yet)

        for t, u in enumerate(inputs):                        # u = the input vector at timestep t
            state = np.tanh(self.W_in @ u + self.W @ state)   # new state = squash(input contribution + memory contribution)
            states[t] = state                                 # record this timestep's full reservoir snapshot

        return states                                         # shape (n_timesteps, n_reservoir) -- reservoir's full history

    def fit(self, inputs, targets, reg=1e-6):
        # targets: shape (n_timesteps, n_outputs) -- what each reservoir state should map to
        states = self.run(inputs)                             # collect reservoir activity for every timestep

        self.W_out = np.linalg.solve(                         # solve ridge regression in closed form:
            states.T @ states + reg * np.eye(self.n_reservoir),  #   (S^T S + reg*I)
            states.T @ targets                                #   S^T targets
        )                                                      # result: W_out, shape (n_reservoir, n_outputs)
        return states                                         # returned in case caller wants to inspect it

    def predict(self, inputs):
        return self.run(inputs) @ self.W_out                  # re-run reservoir, then apply the trained readout