import numpy as np                                          # numerical arrays and linear algebra


class ESN:

    def __init__(self, n_inputs, n_reservoir, spectral_radius=0.9, sparsity=0.1, seed=42):
        # n_inputs: how many numbers arrive per timestep (fixes W_in's shape, permanently)
        # n_reservoir: how many "neurons" the reservoir has (more = more memory capacity)
        # spectral_radius: target stability level for the reservoir
        # sparsity: fraction of reservoir connections that stay non-zero
        # seed: makes the "random" matrices reproducible across runs

        rng = np.random.default_rng(seed)

        self.n_reservoir = n_reservoir

        # Random recurrent reservoir matrix
        W = rng.uniform(-1, 1, (n_reservoir, n_reservoir))

        # Remove connections to create sparsity
        W[rng.random((n_reservoir, n_reservoir)) > sparsity] = 0

        # Scale reservoir to desired spectral radius
        rho = np.max(np.abs(np.linalg.eigvals(W)))

        self.W = W * (spectral_radius / rho)

        # Random input -> reservoir weights
        self.W_in = rng.uniform(
            -1,
            1,
            (n_reservoir, n_inputs)
        )

        # Will be learned later
        self.W_out = None


    def run(self, inputs):
        """
        Run a sequence through the reservoir.

        inputs:
            shape = (timesteps, n_inputs)

        returns:
            states:
            shape = (timesteps, n_reservoir)
        """

        states = np.zeros(
            (len(inputs), self.n_reservoir)
        )

        state = np.zeros(
            self.n_reservoir
        )

        for t, u in enumerate(inputs):

            state = np.tanh(
                self.W_in @ u
                +
                self.W @ state
            )

            states[t] = state

        return states



    def fit(self, inputs, targets, reg=1e-6):
        """
        Original time-series prediction training.

        Each reservoir state maps to a target:

            state(t) -> target(t)

        """

        states = self.run(inputs)

        self.W_out = np.linalg.solve(

            states.T @ states
            +
            reg * np.eye(self.n_reservoir),

            states.T @ targets
        )

        return states



    def predict(self, inputs):
        """
        Original time-series prediction.

        Returns output for every timestep.
        """

        return self.run(inputs) @ self.W_out



    def fit_sequences(self, sequences, targets, reg=1e-6):
        """
        Train on complete sequences.

        Used for classification tasks.

        Each sequence:

            input sequence
                  |
                  v
              reservoir
                  |
                  v
          final reservoir state
                  |
                  v
              target label


        sequences:
            shape = (samples, timesteps, n_inputs)

        targets:
            shape = (samples, outputs)
        """

        states = []


        for sequence in sequences:

            reservoir_states = self.run(sequence)

            # Use the final reservoir state as the sequence representation
            states.append(
                reservoir_states[-1]
            )


        states = np.array(states)


        # Ridge regression:
        # reservoir state -> target

        self.W_out = np.linalg.solve(

            states.T @ states
            +
            reg * np.eye(self.n_reservoir),

            states.T @ targets
        )


        return states



    def predict_sequences(self, sequences):
        """
        Predict one output per sequence.

        Used for classification.

        sequences:
            shape = (samples, timesteps, n_inputs)

        returns:
            shape = (samples, outputs)
        """

        outputs = []


        for sequence in sequences:

            reservoir_states = self.run(sequence)

            final_state = reservoir_states[-1]

            output = final_state @ self.W_out

            outputs.append(output)


        return np.array(outputs)