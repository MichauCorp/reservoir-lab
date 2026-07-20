"""
Simulated optoelectronic delay-line reservoir.

Models the most common physical photonic reservoir architecture in the
literature (Appeltant et al. 2011; Larger et al. 2012; Paquot et al. 2012):
a single nonlinear opto-electronic element (e.g. a Mach-Zehnder modulator)
sitting in a feedback loop with a fixed time delay tau. The nonlinearity
plus the delay together give you both the "many effective neurons" (via
time multiplexing) and the fading memory that a spatial network of neurons
would otherwise have to provide.

Governing equation (normalized Ikeda-style delay differential equation):

    dx/dt = -x(t) / theta  +  eta * sin^2( x(t - tau) + gamma * J(t) + phi )

  x(t)   : the physical state (e.g. optical intensity after the MZM)
  tau    : feedback delay -- set equal to one input step's duration, so
           x(t - tau) is "the reservoir's own state one input-step ago" --
           this is where the recurrence/memory comes from
  theta  : response time of the physical element (a low-pass filter)
  eta    : feedback gain (bigger = more nonlinear excursions)
  gamma  : input gain
  phi    : operating point / bias of the modulator
  J(t)   : masked input waveform (piecewise constant across N virtual
           nodes within each delay window)

If you eventually get real hardware (a laser diode + modulator + fiber
delay line + photodetector, or an audio-frequency analogue built from an
op-amp oscillator), you'd replace the inner update with actual hardware I/O
and keep everything else (masking, virtual node sampling, readout)
identical -- that's the point of the PhysicalReservoir interface.
"""

import numpy as np

from .base import PhysicalReservoir, make_mask


class OptoelectronicReservoir(PhysicalReservoir):

    def __init__(
        self,
        n_inputs,
        n_virtual_nodes=100,
        theta=0.2,
        eta=0.5,
        gamma=0.3,
        phi=np.pi / 4,
        dt=None,
        mask_levels=(-1, 1),
        seed=42,
    ):
        super().__init__(n_reservoir=n_virtual_nodes, seed=seed)

        self.n_virtual_nodes = n_virtual_nodes
        self.theta = theta
        self.eta = eta
        self.gamma = gamma
        self.phi = phi

        # One virtual node per Euler integration step within a delay window.
        # dt is the integration step; the delay tau = n_virtual_nodes * dt.
        self.dt = dt if dt is not None else theta / 10

        self.mask = make_mask(
            n_virtual_nodes, n_inputs, levels=mask_levels, seed=seed
        )

        # Ring buffer holding the last n_virtual_nodes samples of physical
        # state x(t), standing in for the delay line itself.
        self._history = np.zeros(n_virtual_nodes)

    def _nonlinearity(self, x_delayed, J):
        return np.sin(x_delayed + self.gamma * J + self.phi) ** 2

    def run(self, inputs, initial_state=None):
        """
        inputs: shape (timesteps, n_inputs)
        returns: states, shape (timesteps, n_virtual_nodes)
        """
        n_steps = len(inputs)
        states = np.zeros((n_steps, self.n_virtual_nodes))

        history = (
            self._history.copy()
            if initial_state is None
            else np.array(initial_state, dtype=float)
        )

        for t, u in enumerate(inputs):
            # J(t): masked input, one value per virtual node, held for the
            # duration of this input sample -- the piecewise-constant
            # waveform driving the physical nonlinearity across the delay
            # window.
            J = self.mask @ u

            node_values = np.zeros(self.n_virtual_nodes)
            x = history[-1]

            for i in range(self.n_virtual_nodes):
                x_delayed = history[i]  # x(t - tau + i*dt)

                dx = (
                    -x / self.theta
                    + self.eta * self._nonlinearity(x_delayed, J[i])
                )
                x = x + self.dt * dx
                node_values[i] = x

            states[t] = node_values
            history = node_values.copy()

        self._history = history
        self.last_state = history.copy()

        return states