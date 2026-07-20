import numpy as np


class RidgeReadout:
    """
    Linear readout trained with ridge regression, shared by every reservoir
    type in this project (software ESN, optical, acoustic, or eventually
    real hardware). A "reservoir" of any kind only has to produce a states
    array; this class turns states into predictions the same way regardless
    of what generated them.
    """

    def __init__(self, reg=1e-6):
        self.reg = reg
        self.W_out = None

    def fit(self, states, targets, washout=0):
        """
        states:  shape (timesteps, n_reservoir)
        targets: shape (timesteps, n_outputs)
        washout: initial timesteps to discard before regression (lets the
                 reservoir's transient settle before it's used for training)
        """
        train_states = states[washout:]
        train_targets = targets[washout:]

        n = train_states.shape[1]

        self.W_out = np.linalg.solve(
            train_states.T @ train_states + self.reg * np.eye(n),
            train_states.T @ train_targets
        )
        return self

    def predict(self, states):
        if self.W_out is None:
            raise RuntimeError("Readout has not been fit yet.")
        return states @ self.W_out

    def fit_sequences(self, sequence_states, targets, reg=None):
        """
        sequence_states: final state of each sequence, shape (samples, n_reservoir)
        targets:         shape (samples, outputs)
        Used for classification tasks, mirroring ESN.fit_sequences().
        """
        reg = self.reg if reg is None else reg
        n = sequence_states.shape[1]

        self.W_out = np.linalg.solve(
            sequence_states.T @ sequence_states + reg * np.eye(n),
            sequence_states.T @ targets
        )
        return self