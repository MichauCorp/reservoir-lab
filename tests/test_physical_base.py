"""
Tests for the physical reservoir base interface and utilities (base.py).
"""

import numpy as np

from reservoir_lab.physical.base import make_mask, PhysicalReservoir


class TestMakeMask:
    def test_shape(self):
        mask = make_mask(100, 3, seed=42)
        assert mask.shape == (100, 3)

    def test_default_uniform_range(self):
        mask = make_mask(100, 1, seed=42)
        assert mask.min() >= -1.0
        assert mask.max() <= 1.0

    def test_discrete_levels(self):
        mask = make_mask(100, 1, levels=(-1, 1), seed=42)
        unique = np.unique(mask)
        assert set(unique).issubset({-1.0, 1.0})

    def test_reproducibility(self):
        a = make_mask(100, 3, seed=42)
        b = make_mask(100, 3, seed=42)
        assert np.allclose(a, b)

    def test_seed_change(self):
        a = make_mask(100, 3, seed=42)
        b = make_mask(100, 3, seed=99)
        assert not np.allclose(a, b)


class TestPhysicalReservoirABC:
    def test_cannot_instantiate_abstract(self):
        """PhysicalReservoir is an ABC — direct instantiation should fail."""
        try:
            PhysicalReservoir(n_reservoir=10)
            assert False, "Should not be able to instantiate ABC"
        except TypeError:
            pass

    def test_subclass_must_implement_run(self):
        """A subclass that doesn't implement run() should fail."""
        class Incomplete(PhysicalReservoir):
            pass
        try:
            Incomplete(n_reservoir=10)
            assert False, "Should not be able to instantiate without run()"
        except TypeError:
            pass

    def test_subclass_with_run_works(self):
        class Complete(PhysicalReservoir):
            def run(self, inputs, initial_state=None):
                n = len(inputs)
                return np.zeros((n, self.n_reservoir))
        r = Complete(n_reservoir=10, seed=42)
        assert r.n_reservoir == 10
        assert r.last_state.shape == (10,)
        states = r.run(np.zeros((5, 2)))
        assert states.shape == (5, 10)