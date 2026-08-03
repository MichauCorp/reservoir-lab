"""
Mutual (bidirectional) optical <-> acoustic coupling.

Every coupling so far has been one-directional: optical influences
acoustic (as a driving force in SerialPhysicalReservoir, as a damping
modulator in ParametricallyCoupledReservoir), but acoustic never
influences optical back. The deep-ESN literature's own bidirectional
variants (see exp discussion) only add feedback through a narrow,
stabilized channel, specifically because naive bidirectional coupling
risks breaking the echo-state property. This class follows that same
caution:

  - Acoustic's damping is modulated by optical's state (as in
    ParametricallyCoupledReservoir).
  - Optical's operating point (phi) is nudged by acoustic's state, but
    ONLY using acoustic's state from the *previous* timestep -- a one-
    step causal delay, not an instantaneous simultaneous dependency.
    This avoids solving an implicit (algebraic loop) system and keeps
    the whole thing a straightforward forward simulation, at the cost
    of the feedback being one step "stale".
  - The feedback gain into optical is kept small by default, matching
    the "narrow channel" caution from the literature rather than a full
    dense bidirectional connection.

Both component update equations are inlined from OptoelectronicReservoir
and AcousticReservoir (same physics, same defaults) since true mutual
coupling requires interleaving their per-timestep updates rather than
running one to completion before starting the other.
"""

import numpy as np

from .base import PhysicalReservoir, make_mask

# Fixed optical integration step, independent of theta. Same rationale as
# OptoelectronicReservoir.DEFAULT_DT: theta/10 silently coupled the response
# timescale to the virtual-node delay-window size, confounding any theta
# sweep. For theta=0.2, theta/10 == 0.02, so behavior is numerically
# unchanged for the values used throughout this investigation.
DEFAULT_DT = 0.02


class MutualCoupledReservoir(PhysicalReservoir):
    def __init__(
        self,
        n_inputs,
        n_virtual_nodes=100,
        n_oscillators=100,
        n_driver_nodes=5,
        theta=0.2,
        eta=0.5,
        gamma=0.3,
        phi=np.pi / 4,
        optical_dt=None,
        k=1.0,
        alpha=0.5,
        c_base=0.3,
        c_mod_gain=1.0,
        k_c=0.8,
        acoustic_dt=0.05,
        substeps=5,
        input_gain=2.0,
        feedback_gain=0.3,
        mask_levels=(-1, 1),
        seed=42,
    ):
        # n_reservoir = combined width; this class always returns both
        # stages' states concatenated (there's no useful "final tap only"
        # reading for a genuinely mutual system).
        super().__init__(n_reservoir=n_virtual_nodes + n_oscillators, seed=seed)

        self.n_virtual_nodes = n_virtual_nodes
        self.n_oscillators = n_oscillators
        self.theta, self.eta, self.gamma, self.phi = theta, eta, gamma, phi
        self.optical_dt = optical_dt if optical_dt is not None else DEFAULT_DT
        self.mask = make_mask(n_virtual_nodes, n_inputs, levels=mask_levels, seed=seed)

        self.k, self.alpha, self.c_base, self.c_mod_gain, self.k_c = k, alpha, c_base, c_mod_gain, k_c
        self.acoustic_dt = acoustic_dt
        self.substeps = substeps
        self.input_gain = input_gain
        self.feedback_gain = feedback_gain

        driver_idx = self.rng.choice(n_oscillators, size=min(n_driver_nodes, n_oscillators), replace=False)
        self.W_in = np.zeros((n_oscillators, n_inputs))
        self.W_in[driver_idx] = self.rng.uniform(-1, 1, size=(len(driver_idx), n_inputs))

        self._opt_history = np.zeros(n_virtual_nodes)
        self._ac_x = np.zeros(n_oscillators)
        self._ac_v = np.zeros(n_oscillators)
        self._ac_energy_prev = 0.0

    def _coupling(self, x):
        left, right = np.roll(x, 1), np.roll(x, -1)
        left[0], right[-1] = x[0], x[-1]
        return self.k_c * (left + right - 2 * x)

    def run(self, inputs, initial_state=None):
        opt_init, ac_init = (None, None) if initial_state is None else initial_state

        opt_history = self._opt_history.copy() if opt_init is None else np.array(opt_init, dtype=float)
        ac_x = self._ac_x.copy() if ac_init is None else np.array(ac_init, dtype=float)
        ac_v = self._ac_v.copy()
        ac_energy_prev = self._ac_energy_prev if initial_state is not None else 0.0

        n_steps = len(inputs)
        opt_states = np.zeros((n_steps, self.n_virtual_nodes))
        ac_states = np.zeros((n_steps, self.n_oscillators))
        sub_dt = self.acoustic_dt / self.substeps

        for t, u in enumerate(inputs):
            # ---- optical step, nudged by LAST step's acoustic energy ----
            J = self.mask @ u
            phi_t = self.phi + self.feedback_gain * np.tanh(ac_energy_prev)

            node_values = np.zeros(self.n_virtual_nodes)
            x_opt = opt_history[-1]
            for i in range(self.n_virtual_nodes):
                x_delayed = opt_history[i]
                dx = -x_opt / self.theta + self.eta * np.sin(x_delayed + self.gamma * J[i] + phi_t) ** 2
                x_opt = x_opt + self.optical_dt * dx
                node_values[i] = x_opt
            opt_history = node_values.copy()
            opt_states[t] = node_values

            # ---- acoustic step: driven by raw input, damping modulated
            # by THIS step's (just-computed) optical state ----
            mod_signal = np.tanh(node_values.mean())
            c_t = max(self.c_base * (1 + self.c_mod_gain * mod_signal), 0.05 * self.c_base)
            F_drive = self.input_gain * (self.W_in @ u)

            for _ in range(self.substeps):
                force = -self.k * ac_x - self.alpha * ac_x**3 - c_t * ac_v + self._coupling(ac_x) + F_drive
                ac_v = ac_v + sub_dt * force
                ac_x = ac_x + sub_dt * ac_v
            ac_states[t] = ac_x
            ac_energy_prev = ac_x.mean()

        self._opt_history, self._ac_x, self._ac_v = opt_history, ac_x, ac_v
        self._ac_energy_prev = ac_energy_prev
        self.last_state = (opt_history.copy(), ac_x.copy())

        return np.hstack([opt_states, ac_states])