"""
Tests for RidgeReadout (readout.py).
"""

import numpy as np

from reservoir_lab.readout import RidgeReadout


class TestRidgeReadoutFit:
    def test_shape_single_output(self):
        readout = RidgeReadout(reg=1e-6)
        states = np.random.randn(200, 50)
        targets = np.random.randn(200, 1)
        readout.fit(states, targets)
        assert readout.W_out.shape == (50, 1)

    def test_shape_multi_output(self):
        readout = RidgeReadout(reg=1e-6)
        states = np.random.randn(200, 50)
        targets = np.random.randn(200, 3)
        readout.fit(states, targets)
        assert readout.W_out.shape == (50, 3)

    def test_washout(self):
        readout = RidgeReadout(reg=1e-6)
        states = np.random.randn(200, 50)
        targets = np.random.randn(200, 1)

        readout.fit(states, targets, washout=50)
        # With washout=50, only the last 150 timesteps are used for training,
        # but the readout weight shape should be the same
        assert readout.W_out.shape == (50, 1)

    def test_washout_removes_initial_transient(self):
        """With washout, the first N states are ignored — verify by checking
        that the weight vector is different from no-washout."""
        readout_a = RidgeReadout(reg=1e-6)
        readout_b = RidgeReadout(reg=1e-6)
        states = np.random.randn(200, 50)
        targets = np.random.randn(200, 1)

        readout_a.fit(states, targets, washout=0)
        readout_b.fit(states, targets, washout=50)
        assert not np.allclose(readout_a.W_out, readout_b.W_out)

    def test_exact_recovery(self):
        """If targets are a known linear function of states, ridge regression
        with no noise and no regularization should recover it exactly."""
        readout = RidgeReadout(reg=1e-12)
        n = 1000
        states = np.random.randn(n, 20)
        true_W = np.random.randn(20, 2)
        targets = states @ true_W

        readout.fit(states, targets)
        pred = readout.predict(states)
        assert np.allclose(pred, targets, atol=1e-8)


class TestRidgeReadoutPredict:
    def test_predict_after_fit(self):
        readout = RidgeReadout(reg=1e-6)
        states = np.random.randn(100, 20)
        targets = np.random.randn(100, 1)
        readout.fit(states, targets)
        pred = readout.predict(states)
        assert pred.shape == (100, 1)
        assert np.all(np.isfinite(pred))

    def test_predict_before_fit_raises(self):
        readout = RidgeReadout(reg=1e-6)
        states = np.random.randn(10, 5)
        try:
            readout.predict(states)
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass


class TestRidgeReadoutFitSequences:
    def test_shape(self):
        readout = RidgeReadout(reg=1e-6)
        sequence_states = np.random.randn(20, 50)
        targets = np.random.randn(20, 3)
        readout.fit_sequences(sequence_states, targets)
        assert readout.W_out.shape == (50, 3)

    def test_exact_recovery(self):
        readout = RidgeReadout(reg=1e-12)
        states = np.random.randn(30, 10)
        true_W = np.random.randn(10, 2)
        targets = states @ true_W
        readout.fit_sequences(states, targets)
        pred = readout.predict(states)
        assert np.allclose(pred, targets, atol=1e-8)


class TestRidgeReadoutRegularization:
    def test_strong_reg_smaller_weights(self):
        a = RidgeReadout(reg=1e-6)
        b = RidgeReadout(reg=1e2)
        states = np.random.randn(100, 20)
        targets = np.random.randn(100, 2)
        a.fit(states, targets)
        b.fit(states, targets)
        assert np.linalg.norm(b.W_out) < np.linalg.norm(a.W_out)

    def test_default_reg_is_1e_6(self):
        readout = RidgeReadout()
        assert readout.reg == 1e-6

    def test_fit_sequences_custom_reg(self):
        readout = RidgeReadout(reg=1e-6)
        states = np.random.randn(20, 10)
        targets = np.random.randn(20, 1)
        readout.fit_sequences(states, targets, reg=1e2)
        # Should use the provided reg, not self.reg
        assert readout.W_out is not None