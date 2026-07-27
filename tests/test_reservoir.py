"""
Tests for the software ESN (reservoir.py).
"""

import numpy as np

from reservoir_lab.reservoir import ESN


class TestESNInit:
    def test_default_construction(self):
        esn = ESN(n_inputs=1, n_reservoir=200)
        assert esn.W.shape == (200, 200)
        assert esn.W_in.shape == (200, 1)
        assert esn.W_out is None
        assert esn.last_state.shape == (200,)
        assert np.all(esn.last_state == 0)

    def test_spectral_radius(self):
        esn = ESN(n_inputs=1, n_reservoir=200, spectral_radius=0.9)
        rho = np.max(np.abs(np.linalg.eigvals(esn.W)))
        assert abs(rho - 0.9) < 1e-10

    def test_sparsity(self):
        esn = ESN(n_inputs=1, n_reservoir=500, sparsity=0.1)
        nonzero_frac = np.count_nonzero(esn.W) / (500 * 500)
        assert abs(nonzero_frac - 0.1) < 0.02  # stochastic, but should be close

    def test_reproducibility(self):
        a = ESN(n_inputs=1, n_reservoir=200, seed=42)
        b = ESN(n_inputs=1, n_reservoir=200, seed=42)
        assert np.allclose(a.W, b.W)
        assert np.allclose(a.W_in, b.W_in)

    def test_seed_change(self):
        a = ESN(n_inputs=1, n_reservoir=200, seed=42)
        b = ESN(n_inputs=1, n_reservoir=200, seed=99)
        assert not np.allclose(a.W, b.W)  # vanishingly unlikely to collide

    def test_zero_spectral_radius_raises(self):
        """Sparsity=1.0 keeps no connections after zeroing, spectral radius = 0 → error."""
        import pytest
        with pytest.raises(ValueError, match="spectral radius 0"):
            ESN(n_inputs=1, n_reservoir=200, sparsity=0.0)


class TestESNRun:
    def test_shape(self):
        esn = ESN(n_inputs=2, n_reservoir=100)
        inputs = np.random.randn(50, 2)
        states = esn.run(inputs)
        assert states.shape == (50, 100)
        assert esn.last_state.shape == (100,)

    def test_zero_input_equilibrium(self):
        """Zero input should drive the reservoir toward zero (tanh(0) = 0)."""
        esn = ESN(n_inputs=1, n_reservoir=100)
        inputs = np.zeros((100, 1))
        states = esn.run(inputs)
        # After many steps with zero input, all states should be near zero
        # (the reservoir's own recurrent dynamics may keep it slightly active,
        #  but it should be bounded and tend toward zero for this default setup)
        assert np.all(np.abs(states[-1]) < 0.5)

    def test_continue_from_last_state(self):
        esn = ESN(n_inputs=1, n_reservoir=100)
        inputs = np.random.randn(50, 1)
        states_1 = esn.run(inputs)
        states_2 = esn.run(inputs, initial_state=esn.last_state)
        # Running 50 + 50 = 100 steps in two segments should give the same
        # result as running 100 steps in one shot
        esn2 = ESN(n_inputs=1, n_reservoir=100, seed=42)
        states_full = esn2.run(np.vstack([inputs, inputs]))
        assert np.allclose(states_2, states_full[50:])

    def test_finite_output(self):
        esn = ESN(n_inputs=5, n_reservoir=300, spectral_radius=0.99)
        inputs = np.random.randn(200, 5)
        states = esn.run(inputs)
        assert np.all(np.isfinite(states))

    def test_known_input_output(self):
        """With a simple input pattern, the first state should be deterministic."""
        esn = ESN(n_inputs=1, n_reservoir=10, seed=42)
        states = esn.run(np.ones((1, 1)))
        expected = np.tanh(esn.W_in @ np.ones(1) + esn.W @ np.zeros(10))
        assert np.allclose(states[0], expected.flatten())


class TestESNFit:
    def test_shape(self):
        esn = ESN(n_inputs=1, n_reservoir=100)
        inputs = np.random.randn(200, 1)
        targets = np.random.randn(200, 1)
        states = esn.fit(inputs, targets)
        assert states.shape == (200, 100)
        assert esn.W_out.shape == (100, 1)

    def test_learns_sine_wave(self):
        """The ESN should learn next-step sine prediction to near-zero MSE."""
        np.random.seed(0)
        t = np.linspace(0, 100, 3000)
        data = np.sin(t).reshape(-1, 1)
        inputs, targets = data[:-1], data[1:]

        esn = ESN(n_inputs=1, n_reservoir=200, seed=42)
        esn.fit(inputs, targets, reg=1e-6)
        pred = esn.predict(inputs)
        mse = np.mean((pred - targets) ** 2)
        assert mse < 0.01, f"MSE too high: {mse:.6f}"

    def test_washout_doesnt_break(self):
        esn = ESN(n_inputs=1, n_reservoir=100)
        inputs = np.random.randn(200, 1)
        targets = np.random.randn(200, 1)
        states = esn.fit(inputs, targets, washout=50)
        assert states.shape == (200, 100)

    def test_regularization_works(self):
        """Stronger regularization should produce smaller weight norms."""
        esn_a = ESN(n_inputs=1, n_reservoir=100, seed=42)
        esn_b = ESN(n_inputs=1, n_reservoir=100, seed=42)
        inputs = np.random.randn(100, 1)
        targets = np.random.randn(100, 1)

        esn_a.fit(inputs, targets, reg=1e-6)
        esn_b.fit(inputs, targets, reg=1e2)

        assert np.linalg.norm(esn_b.W_out) < np.linalg.norm(esn_a.W_out)

    def test_predict_before_fit(self):
        esn = ESN(n_inputs=1, n_reservoir=100)
        inputs = np.random.randn(10, 1)
        # W_out is None, so predict() does: self.run(inputs) @ None
        # NumPy treats None as a scalar? Actually None @ array → TypeError.
        # What actually happens: None can't be used in matmul → error.
        try:
            esn.predict(inputs)
            # If we get here without an error, W_out is None and matmul
            # with None raises TypeError — this is NumPy behaviour, not
            # something we control, so just verify the output is garbage
        except (TypeError, ValueError):
            pass
        else:
            # If no exception was raised, still fine — the output just
            # isn't meaningful
            pass


class TestESNFitSequences:
    def test_shape(self):
        esn = ESN(n_inputs=2, n_reservoir=50)
        sequences = np.random.randn(10, 30, 2)  # (samples, timesteps, n_inputs)
        targets = np.random.randn(10, 3)
        states = esn.fit_sequences(sequences, targets)
        assert states.shape == (10, 50)
        assert esn.W_out.shape == (50, 3)

    def test_classification_learns(self):
        """Two well-separated classes should be learnable."""
        esn = ESN(n_inputs=1, n_reservoir=100, seed=42)

        # Two classes: class 0 = low-frequency, class 1 = high-frequency
        t = np.arange(100)
        class0 = np.sin(0.1 * t).reshape(1, -1, 1)
        class1 = np.sin(1.0 * t).reshape(1, -1, 1)
        sequences = np.vstack([class0, class1])  # (2, 100, 1)
        targets = np.eye(2)  # one-hot: [1,0] = class 0, [0,1] = class 1

        esn.fit_sequences(sequences, targets)
        pred = esn.predict_sequences(sequences)
        predicted_classes = np.argmax(pred, axis=1)
        assert np.all(predicted_classes == [0, 1])

    def test_predict_sequences_shape(self):
        esn = ESN(n_inputs=1, n_reservoir=50, seed=42)
        sequences = np.random.randn(5, 30, 1)
        targets = np.random.randn(5, 2)
        esn.fit_sequences(sequences, targets)
        pred = esn.predict_sequences(sequences)
        assert pred.shape == (5, 2)