"""
Simulated acoustic / mechanical reservoir.

Models a physical reservoir computing substrate as a lattice of coupled,
damped, nonlinear oscillators -- the kind of thing you'd build from a line
of masses and springs, a metamaterial plate, or a chain of coupled piezo
elements excited by an acoustic driver. Unlike the optical reservoir (one
element + time-multiplexed delay), this is a genuinely spatial reservoir:
many physical degrees of freedom (oscillator displacements), each one a
"reservoir neuron," coupled to their neighbors.

Equations of motion for oscillator i (a driven, damped Duffing oscillator,
coupled to its neighbors):

    m*x_i'' = -k*x_i - alpha*x_i^3 - c*x_i'
              + k_c*(x_(i-1) + x_(i+1) - 2*x_i)   [nearest-neighbor coupling]
              + F_i(t)                             [drive, input-coupled]

  x_i    : displacement of oscillator i (a reservoir neuron)
  k, alpha : linear and cubic (nonlinear) stiffness -- alpha is what gives
             you the nonlinearity a linear acoustic medium lacks
  c      : damping -- gives *fading* memory; without it, oscillations would
           ring forever and the reservoir would never "forget" old inputs
  k_c    : coupling strength between neighboring oscillators -- lets
           information about the input spread spatially across the lattice,
           standing in for elastic/acoustic coupling in a real medium
  F_i(t) : the input is only injected at a few "driver" nodes (like a real
           acoustic reservoir only has a couple of transducers), and has to
           propagate through the coupling to reach the rest of the lattice

Integrated with a semi-implicit (symplectic) Euler scheme, which is
adequately stable for damped oscillator systems without needing a heavier
integrator.

Swapping this for real hardware later means replacing run()'s inner loop
with: write F_i(t) to a DAC/driver, read x_i(t) back from an accelerometer/
microphone array, at whatever the hardware's native sample rate is.
"""

import numpy as np

from .base import PhysicalReservoir


class AcousticReservoir(PhysicalReservoir):

    def __init__(
        self,
        n_inputs,
        n_oscillators=100,
        n_driver_nodes=5,
        k=1.0,
        alpha=0.5,
        c=0.3,
        k_c=0.8,
        dt=0.05,
        substeps=5,
        input_gain=2.0,
        seed=42,
    ):
        super().__init__(n_reservoir=n_oscillators, seed=seed)

        self.n_oscillators = n_oscillators
        self.k = k
        self.alpha = alpha
        self.c = c
        self.k_c = k_c
        self.dt = dt
        self.substeps = substeps  # inner integration steps per input sample
        self.input_gain = input_gain

        # Which oscillators the input actually drives -- a real acoustic
        # reservoir only has a handful of transducers, not one per neuron.
        driver_idx = self.rng.choice(
            n_oscillators, size=min(n_driver_nodes, n_oscillators), replace=False
        )
        self.W_in = np.zeros((n_oscillators, n_inputs))
        self.W_in[driver_idx] = self.rng.uniform(
            -1, 1, size=(len(driver_idx), n_inputs)
        )

        self._x = np.zeros(n_oscillators)   # displacement
        self._v = np.zeros(n_oscillators)   # velocity

    def _coupling(self, x):
        # Nearest-neighbor coupling on a 1D chain, fixed (non-periodic)
        # boundaries -- swap for a 2D lattice/graph coupling to model a
        # plate or an irregular structure instead.
        left = np.roll(x, 1)
        right = np.roll(x, -1)
        left[0] = x[0]
        right[-1] = x[-1]
        return self.k_c * (left + right - 2 * x)

    def run(self, inputs, initial_state=None):
        """
        inputs: shape (timesteps, n_inputs)
        returns: states, shape (timesteps, n_oscillators) -- the
                 displacement of every oscillator, sampled once per input
                 timestep (after `substeps` inner integration steps).
        """
        n_steps = len(inputs)
        states = np.zeros((n_steps, self.n_oscillators))

        x = self._x.copy() if initial_state is None else np.array(initial_state, dtype=float)
        v = self._v.copy()
        sub_dt = self.dt / self.substeps

        for t, u in enumerate(inputs):
            F_drive = self.input_gain * (self.W_in @ u)

            for _ in range(self.substeps):
                force = (
                    -self.k * x
                    - self.alpha * x**3
                    - self.c * v
                    + self._coupling(x)
                    + F_drive
                )
                # semi-implicit Euler: update velocity first, then position
                # with the *new* velocity -- more stable than plain Euler
                # for oscillatory systems
                v = v + sub_dt * force
                x = x + sub_dt * v

            states[t] = x

        self._x, self._v = x, v
        self.last_state = x.copy()

        return states