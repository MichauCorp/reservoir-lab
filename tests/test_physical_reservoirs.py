"""
Tests for physical reservoir implementations (acoustic, optical, coupled, etc.).
"""

import numpy as np
import pytest

from reservoir_lab.physical.acoustic import AcousticReservoir
from reservoir_lab.physical.coupled import SerialPhysicalReservoir
from reservoir_lab.physical.hardware import HardwareReservoir
from reservoir_lab.physical.mutual import MutualCoupledReservoir
from reservoir_lab.physical.optical import OptoelectronicReservoir
from reservoir_lab.physical.parametric_coupling import ParametricallyCoupledReservoir


class TestAcousticReservoir:
    def test_default_construction(self):
        res = AcousticReservoir(n_inputs=2)
        assert res.n_oscillators == 100
        assert res.n_reservoir == 100
        assert res.W_in.shape == (100, 2)
        assert res.last_state.shape == (100,)
        assert np.all(res.last_state == 0)

    def test_custom_dimensions(self):
        res = AcousticReservoir(n_inputs=3, n_oscillators=50, n_driver_nodes=3, seed=42)
        assert res.n_oscillators == 50
        assert res.n_reservoir == 50
        assert res.W_in.shape == (50, 3)
        # Check that only driver nodes have non-zero input weights
        nonzero_rows = np.where(np.any(res.W_in != 0, axis=1))[0]
        assert len(nonzero_rows) == 3

    def test_run_shape(self):
        res = AcousticReservoir(n_inputs=2, n_oscillators=30, seed=42)
        inputs = np.random.randn(20, 2)
        states = res.run(inputs)
        assert states.shape == (20, 30)
        assert res.last_state.shape == (30,)

    def test_run_finite_output(self):
        res = AcousticReservoir(n_inputs=1, n_oscillators=50, seed=42)
        inputs = np.random.randn(100, 1)
        states = res.run(inputs)
        assert np.all(np.isfinite(states))

    def test_reproducibility(self):
        inputs = np.random.randn(30, 2)
        res_a = AcousticReservoir(n_inputs=2, n_oscillators=30, seed=42)
        res_b = AcousticReservoir(n_inputs=2, n_oscillators=30, seed=42)
        states_a = res_a.run(inputs)
        states_b = res_b.run(inputs)
        assert np.allclose(states_a, states_b)

    def test_seed_change(self):
        inputs = np.random.randn(30, 2)
        res_a = AcousticReservoir(n_inputs=2, n_oscillators=30, seed=42)
        res_b = AcousticReservoir(n_inputs=2, n_oscillators=30, seed=99)
        states_a = res_a.run(inputs)
        states_b = res_b.run(inputs)
        assert not np.allclose(states_a, states_b)

    def test_zero_input(self):
        """With zero input, the reservoir should remain near equilibrium."""
        res = AcousticReservoir(n_inputs=1, n_oscillators=50, seed=42)
        inputs = np.zeros((50, 1))
        states = res.run(inputs)
        # With no driving force, oscillators should stay small
        assert np.all(np.abs(states[-1]) < 1.0)

    def test_continue_from_last_state(self):
        res = AcousticReservoir(n_inputs=1, n_oscillators=50, seed=42)
        inputs = np.random.randn(30, 1)
        states_1 = res.run(inputs)
        states_2 = res.run(inputs, initial_state=res.last_state)
        
        res2 = AcousticReservoir(n_inputs=1, n_oscillators=50, seed=42)
        states_full = res2.run(np.vstack([inputs, inputs]))
        assert np.allclose(states_2, states_full[30:], atol=1e-10)

    def test_driver_node_effect(self):
        """Different inputs should produce different states."""
        res = AcousticReservoir(n_inputs=1, n_oscillators=50, seed=42)
        inputs_a = np.ones((20, 1))
        inputs_b = -np.ones((20, 1))
        states_a = res.run(inputs_a)
        states_b = res.run(inputs_b)
        assert not np.allclose(states_a, states_b)


class TestOptoelectronicReservoir:
    def test_default_construction(self):
        res = OptoelectronicReservoir(n_inputs=2)
        assert res.n_virtual_nodes == 100
        assert res.n_reservoir == 100
        assert res.mask.shape == (100, 2)
        assert res.last_state.shape == (100,)

    def test_custom_virtual_nodes(self):
        res = OptoelectronicReservoir(n_inputs=3, n_virtual_nodes=50, seed=42)
        assert res.n_virtual_nodes == 50
        assert res.n_reservoir == 50

    def test_run_shape(self):
        res = OptoelectronicReservoir(n_inputs=2, n_virtual_nodes=30, seed=42)
        inputs = np.random.randn(20, 2)
        states = res.run(inputs)
        assert states.shape == (20, 30)
        assert res.last_state.shape == (30,)

    def test_run_finite_output(self):
        res = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=50, seed=42)
        inputs = np.random.randn(100, 1)
        states = res.run(inputs)
        assert np.all(np.isfinite(states))

    def test_reproducibility(self):
        inputs = np.random.randn(20, 2)
        res_a = OptoelectronicReservoir(n_inputs=2, n_virtual_nodes=30, seed=42)
        res_b = OptoelectronicReservoir(n_inputs=2, n_virtual_nodes=30, seed=42)
        states_a = res_a.run(inputs)
        states_b = res_b.run(inputs)
        assert np.allclose(states_a, states_b)

    def test_seed_change(self):
        inputs = np.random.randn(20, 2)
        res_a = OptoelectronicReservoir(n_inputs=2, n_virtual_nodes=30, seed=42)
        res_b = OptoelectronicReservoir(n_inputs=2, n_virtual_nodes=30, seed=99)
        states_a = res_a.run(inputs)
        states_b = res_b.run(inputs)
        assert not np.allclose(states_a, states_b)

    def test_continue_from_last_state(self):
        res = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=30, seed=42)
        inputs = np.random.randn(20, 1)
        states_1 = res.run(inputs)
        states_2 = res.run(inputs, initial_state=res.last_state)
        
        res2 = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=30, seed=42)
        states_full = res2.run(np.vstack([inputs, inputs]))
        assert np.allclose(states_2, states_full[20:], atol=1e-10)

    def test_input_drives_state(self):
        """Different inputs should produce different states."""
        res = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=50, seed=42)
        inputs_a = np.ones((20, 1))
        inputs_b = -np.ones((20, 1))
        states_a = res.run(inputs_a)
        states_b = res.run(inputs_b)
        assert not np.allclose(states_a, states_b)


class TestMutualCoupledReservoir:
    def test_default_construction(self):
        res = MutualCoupledReservoir(n_inputs=2)
        assert res.n_virtual_nodes == 100
        assert res.n_oscillators == 100
        assert res.n_reservoir == 200  # combined

    def test_run_shape(self):
        res = MutualCoupledReservoir(n_inputs=2, n_virtual_nodes=30, n_oscillators=20, seed=42)
        inputs = np.random.randn(20, 2)
        states = res.run(inputs)
        assert states.shape == (20, 50)  # 30 + 20

    def test_run_finite_output(self):
        res = MutualCoupledReservoir(n_inputs=1, n_virtual_nodes=30, n_oscillators=20, seed=42)
        inputs = np.random.randn(50, 1)
        states = res.run(inputs)
        assert np.all(np.isfinite(states))

    def test_last_state_is_tuple(self):
        res = MutualCoupledReservoir(n_inputs=1, n_virtual_nodes=20, n_oscillators=15, seed=42)
        inputs = np.random.randn(10, 1)
        res.run(inputs)
        assert isinstance(res.last_state, tuple)
        assert len(res.last_state) == 2

    def test_continue_from_last_state(self):
        res = MutualCoupledReservoir(n_inputs=1, n_virtual_nodes=20, n_oscillators=15, seed=42)
        inputs = np.random.randn(15, 1)
        states_1 = res.run(inputs)
        states_2 = res.run(inputs, initial_state=res.last_state)
        
        res2 = MutualCoupledReservoir(n_inputs=1, n_virtual_nodes=20, n_oscillators=15, seed=42)
        states_full = res2.run(np.vstack([inputs, inputs]))
        assert np.allclose(states_2, states_full[15:], atol=1e-10)

    def test_feedback_gain_effect(self):
        """Higher feedback gain should produce different dynamics."""
        res_low = MutualCoupledReservoir(n_inputs=1, n_virtual_nodes=20, n_oscillators=15, 
                                         feedback_gain=0.0, seed=42)
        res_high = MutualCoupledReservoir(n_inputs=1, n_virtual_nodes=20, n_oscillators=15,
                                          feedback_gain=1.0, seed=42)
        inputs = np.random.randn(20, 1)
        states_low = res_low.run(inputs)
        states_high = res_high.run(inputs)
        assert not np.allclose(states_low, states_high)


class TestParametricallyCoupledReservoir:
    def test_construction_with_modulator(self):
        modulator = OptoelectronicReservoir(n_inputs=2, n_virtual_nodes=50, seed=42)
        res = ParametricallyCoupledReservoir(modulator=modulator, n_inputs=2)
        assert res.n_oscillators == 100
        assert res.n_reservoir == 100
        assert res.modulator is modulator

    def test_run_shape(self):
        modulator = OptoelectronicReservoir(n_inputs=2, n_virtual_nodes=30, seed=42)
        res = ParametricallyCoupledReservoir(modulator=modulator, n_inputs=2, 
                                             n_oscillators=20, seed=42)
        inputs = np.random.randn(20, 2)
        states = res.run(inputs)
        assert states.shape == (20, 20)

    def test_run_finite_output(self):
        modulator = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=30, seed=42)
        res = ParametricallyCoupledReservoir(modulator=modulator, n_inputs=1,
                                             n_oscillators=20, seed=42)
        inputs = np.random.randn(50, 1)
        states = res.run(inputs)
        assert np.all(np.isfinite(states))

    def test_last_state_is_tuple(self):
        modulator = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=20, seed=42)
        res = ParametricallyCoupledReservoir(modulator=modulator, n_inputs=1,
                                             n_oscillators=15, seed=42)
        inputs = np.random.randn(10, 1)
        res.run(inputs)
        assert isinstance(res.last_state, tuple)
        assert len(res.last_state) == 2

    def test_c_mod_gain_effect(self):
        """Different modulation gains should produce different behavior."""
        modulator = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=30, seed=42)
        
        res_no_mod = ParametricallyCoupledReservoir(
            modulator=modulator, n_inputs=1, n_oscillators=20,
            c_mod_gain=0.0, seed=42
        )
        res_mod = ParametricallyCoupledReservoir(
            modulator=modulator, n_inputs=1, n_oscillators=20,
            c_mod_gain=2.0, seed=42
        )
        inputs = np.random.randn(30, 1)
        states_no_mod = res_no_mod.run(inputs)
        states_mod = res_mod.run(inputs)
        assert not np.allclose(states_no_mod, states_mod)

    def test_continue_from_last_state(self):
        modulator = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=20, seed=42)
        res = ParametricallyCoupledReservoir(modulator=modulator, n_inputs=1,
                                             n_oscillators=15, seed=42)
        inputs = np.random.randn(15, 1)
        states_1 = res.run(inputs)
        states_2 = res.run(inputs, initial_state=res.last_state)
        
        modulator2 = OptoelectronicReservoir(n_inputs=1, n_virtual_nodes=20, seed=42)
        res2 = ParametricallyCoupledReservoir(modulator=modulator2, n_inputs=1,
                                              n_oscillators=15, seed=42)
        states_full = res2.run(np.vstack([inputs, inputs]))
        assert np.allclose(states_2, states_full[15:], atol=1e-10)


class TestSerialPhysicalReservoir:
    def test_construction(self):
        stage1 = OptoelectronicReservoir(n_inputs=2, n_virtual_nodes=50, seed=42)
        stage2 = AcousticReservoir(n_inputs=50, n_oscillators=30, seed=42)
        res = SerialPhysicalReservoir(stage1, stage2)
        assert res.stage1 is stage1
        assert res.stage2 is stage2
        assert res.n_reservoir == 30  # "final" mode

    def test_construction_both_mode(self):
        stage1 = OptoelectronicReservoir(n_inputs=2, n_virtual_nodes=50, seed=42)
        stage2 = AcousticReservoir(n_inputs=50, n_oscillators=30, seed=42)
        res = SerialPhysicalReservoir(stage1, stage2, combine="both")
        assert res.n_reservoir == 80  # 50 + 30

    def test_invalid_combine_raises(self):
        stage1 = OptoelectronicReservoir(n_inputs=2, n_virtual_nodes=50, seed=42)
        stage2 = AcousticReservoir(n_inputs=50, n_oscillators=30, seed=42)
        with pytest.raises(ValueError, match="combine must be"):
            SerialPhysicalReservoir(stage1, stage2, combine="invalid")

    def test_run_shape_final(self):
        stage1 = OptoelectronicReservoir(n_inputs=2, n_virtual_nodes=30, seed=42)
        stage2 = AcousticReservoir(n_inputs=30, n_oscillators=20, seed=42)
        res = SerialPhysicalReservoir(stage1, stage2, combine="final")
        inputs = np.random.randn(20, 2)
        states = res.run(inputs)
        assert states.shape == (20, 20)

    def test_run_shape_both(self):
        stage1 = OptoelectronicReservoir(n_inputs=2, n_virtual_nodes=30, seed=42)
        stage2 = AcousticReservoir(n_inputs=30, n_oscillators=20, seed=42)
        res = SerialPhysicalReservoir(stage1, stage2, combine="both")
        inputs = np.random.randn(20, 2)
        states = res.run(inputs)
        assert states.shape == (20, 50)  # 30 + 20

    def test_run_finite_output(self):
        stage1 = OptoelectronicReservoir(n_inputs=2, n_virtual_nodes=30, seed=42)
        stage2 = AcousticReservoir(n_inputs=30, n_oscillators=20, seed=42)
        res = SerialPhysicalReservoir(stage1, stage2)
        inputs = np.random.randn(50, 2)
        states = res.run(inputs)
        assert np.all(np.isfinite(states))

    def test_last_state_is_tuple(self):
        stage1 = OptoelectronicReservoir(n_inputs=2, n_virtual_nodes=20, seed=42)
        stage2 = AcousticReservoir(n_inputs=20, n_oscillators=15, seed=42)
        res = SerialPhysicalReservoir(stage1, stage2)
        inputs = np.random.randn(10, 2)
        res.run(inputs)
        assert isinstance(res.last_state, tuple)
        assert len(res.last_state) == 2

    def test_continue_from_last_state(self):
        stage1 = OptoelectronicReservoir(n_inputs=2, n_virtual_nodes=20, seed=42)
        stage2 = AcousticReservoir(n_inputs=20, n_oscillators=15, seed=42)
        res = SerialPhysicalReservoir(stage1, stage2)
        inputs = np.random.randn(15, 2)
        states_1 = res.run(inputs)
        states_2 = res.run(inputs, initial_state=res.last_state)
        
        stage1b = OptoelectronicReservoir(n_inputs=2, n_virtual_nodes=20, seed=42)
        stage2b = AcousticReservoir(n_inputs=20, n_oscillators=15, seed=42)
        res2 = SerialPhysicalReservoir(stage1b, stage2b)
        states_full = res2.run(np.vstack([inputs, inputs]))
        assert np.allclose(states_2, states_full[15:], atol=1e-10)


class TestHardwareReservoir:
    def test_construction(self):
        res = HardwareReservoir(n_inputs=3, n_reservoir=50, seed=42)
        assert res.n_reservoir == 50
        assert res.n_inputs == 3
        assert res.last_state.shape == (50,)

    def test_run_raises_not_implemented(self):
        res = HardwareReservoir(n_inputs=2, n_reservoir=20, seed=42)
        inputs = np.random.randn(10, 2)
        with pytest.raises(NotImplementedError):
            res.run(inputs)

    def test_drive_and_read_raises(self):
        res = HardwareReservoir(n_inputs=2, n_reservoir=20, seed=42)
        with pytest.raises(NotImplementedError):
            res._drive_and_read(np.ones(2))

    def test_close_does_not_raise(self):
        res = HardwareReservoir(n_inputs=2, n_reservoir=20, seed=42)
        res.close()  # Should not raise