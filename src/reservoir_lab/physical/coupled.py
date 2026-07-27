"""
Serial composition of two PhysicalReservoir stages.

input -> stage1 -> stage2 -> readout

Both OptoelectronicReservoir and AcousticReservoir already share the same
PhysicalReservoir interface (`run(inputs, initial_state=None) -> states`,
plus `.last_state` for train/test continuity), so chaining them just means
feeding stage1's states array in as stage2's `inputs` at every timestep.
Wrapping that in a PhysicalReservoir subclass means it drops straight into
the same evaluate()/RidgeReadout pattern used everywhere else in this
project (see exp08, exp09) with no changes required elsewhere.
"""

import numpy as np

from .base import PhysicalReservoir


class SerialPhysicalReservoir(PhysicalReservoir):
    """
    Chains stage1 -> stage2. stage2 is driven by stage1's states rather
    than the raw input, so stage2 must be constructed with
    n_inputs == stage1.n_reservoir.

    combine:
        "final" (default) -- readout only ever sees stage2's states. This
            is the literal input -> stage1 -> stage2 -> readout pipeline,
            and it's the one to use if the claim under test is "does the
            *second* stage, on its own, benefit from the first stage's
            preprocessing".
        "both" -- readout sees [stage1_states, stage2_states]
            concatenated. Use this to check whether funnelling everything
            through stage2 alone throws away information stage1 had that
            a downstream readout could otherwise still use.
    """

    def __init__(self, stage1, stage2, combine="final", seed=42):
        if combine not in ("final", "both"):
            raise ValueError("combine must be 'final' or 'both'")

        n_reservoir = (
            stage2.n_reservoir if combine == "final"
            else stage1.n_reservoir + stage2.n_reservoir
        )
        super().__init__(n_reservoir=n_reservoir, seed=seed)

        self.stage1 = stage1
        self.stage2 = stage2
        self.combine = combine

    def run(self, inputs, initial_state=None):
        """
        inputs: shape (timesteps, n_inputs) -- fed to stage1 only.
        initial_state: None, or a (stage1_initial, stage2_initial) tuple,
                        mirroring each stage's own `.last_state` so a
                        SerialPhysicalReservoir can be resumed across a
                        train/test boundary the same way a single-stage
                        reservoir is (see evaluate() in exp08/exp09).
        returns: states, shape (timesteps, self.n_reservoir)
        """
        s1_init, s2_init = (None, None) if initial_state is None else initial_state

        states1 = self.stage1.run(inputs, initial_state=s1_init)
        states2 = self.stage2.run(states1, initial_state=s2_init)

        self.last_state = (self.stage1.last_state, self.stage2.last_state)

        if self.combine == "both":
            return np.hstack([states1, states2])
        return states2