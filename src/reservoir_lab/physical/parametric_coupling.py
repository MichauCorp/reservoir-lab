"""
Parametric (multiplicative) coupling between reservoirs.

Every coupling tried so far (SerialPhysicalReservoir, exp10/exp11) feeds
one reservoir's *output* in as another's *input* -- the acoustic lattice
gets driven by the optical stage's states the same way it would be
driven by raw input. That's still, mathematically, a fixed function of
one system handed to a second, separately-evolving system: additive
(concatenation) and input-driven coupling are both linear-in-two-
separately-evolving-systems operations, and exp11 found that kind of
coupling adds little over just running both reservoirs in parallel.

This module instead modulates a *physical parameter* of the acoustic
lattice -- its damping coefficient -- using the optical stage's
instantaneous state. Concretely: how fast energy leaves the acoustic
lattice depends on what the optical stage is doing right now, not on
what was fed into the lattice. This is qualitatively different: it can
produce genuine fast x slow interaction features (e.g. "a fast optical
event happening while the lattice is in an underdamped, ringing phase"
leaves a different trace than the same event during an overdamped
phase), which neither concatenation nor input-driven coupling can
represent.

Damping is clamped to stay non-negative (with a small positive floor)
throughout -- negative damping means the lattice pumps energy in
indefinitely and blows up, which is a real risk of parametric coupling
that input-driven coupling doesn't have.
"""

import numpy as np

from .base import PhysicalReservoir


class ParametricallyCoupledReservoir(PhysicalReservoir):
    """
    Acoustic oscillator lattice (same equations of motion as
    AcousticReservoir) whose damping coefficient is modulated at every
    timestep by an upstream PhysicalReservoir's instantaneous state
    (typically OptoelectronicReservoir), instead of being driven by it
    as an input.

    modulator : PhysicalReservoir
        Any reservoir producing states of shape
        (timesteps, modulator.n_reservoir). Its states are averaged and
        squashed to build a per-timestep damping modulation signal.
    c_base : float
        Baseline damping (matches AcousticReservoir's `c`).
    c_mod_gain : float
        How strongly the modulator's state shifts damping away from
        c_base. Damping is c_base * (1 + c_mod_gain * tanh(modulator
        signal)), clamped to a small positive floor.
    """

    def __init__(
        self,
        modulator,
        n_inputs,
        n_oscillators=100,
        n_driver_nodes=5,
        k=1.0,
        alpha=0.5,
        c_base=0.3,
        c_mod_gain=1.0,
        k_c=0.8,
        dt=0.05,
        substeps=5,
        input_gain=2.0,
        seed=42,
    ):
        super().__init__(n_reservoir=n_oscillators, seed=seed)

        self.modulator = modulator
        self.n_oscillators = n_oscillators
        self.k = k
        self.alpha = alpha
        self.c_base = c_base
        self.c_mod_gain = c_mod_gain
        self.k_c = k_c
        self.dt = dt
        self.substeps = substeps
        self.input_gain = input_gain

        driver_idx = self.rng.choice(
            n_oscillators, size=min(n_driver_nodes, n_oscillators), replace=False
        )
        self.W_in = np.zeros((n_oscillators, n_inputs))
        self.W_in[driver_idx] = self.rng.uniform(-1, 1, size=(len(driver_idx), n_inputs))

        self._x = np.zeros(n_oscillators)
        self._v = np.zeros(n_oscillators)

    def _coupling(self, x):
        left = np.roll(x, 1)
        right = np.roll(x, -1)
        left[0] = x[0]
        right[-1] = x[-1]
        return self.k_c * (left + right - 2 * x)

    def run(self, inputs, initial_state=None):
        """
        inputs: shape (timesteps, n_inputs) -- fed to BOTH the modulator
                and this lattice's driver nodes.
        initial_state: None, or (lattice_initial, modulator_initial),
                       mirroring SerialPhysicalReservoir's convention.
        returns: states, shape (timesteps, n_oscillators)
        """
        lat_init, mod_init = (None, None) if initial_state is None else initial_state

        mod_states = self.modulator.run(inputs, initial_state=mod_init)
        mod_signal = np.tanh(mod_states.mean(axis=1))  # bounded to [-1, 1]

        n_steps = len(inputs)
        states = np.zeros((n_steps, self.n_oscillators))

        x = self._x.copy() if lat_init is None else np.array(lat_init, dtype=float)
        v = self._v.copy()
        sub_dt = self.dt / self.substeps

        for t, u in enumerate(inputs):
            F_drive = self.input_gain * (self.W_in @ u)
            c_t = max(self.c_base * (1 + self.c_mod_gain * mod_signal[t]), 0.05 * self.c_base)

            for _ in range(self.substeps):
                force = (
                    -self.k * x
                    - self.alpha * x**3
                    - c_t * v
                    + self._coupling(x)
                    + F_drive
                )
                v = v + sub_dt * force
                x = x + sub_dt * v

            states[t] = x

        self._x, self._v = x, v
        self.last_state = (x.copy(), self.modulator.last_state)

        return states