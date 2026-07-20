"""
Shared interface and utilities for "physical" reservoirs.

Real optical/acoustic/mechanical reservoir computers almost universally use
a single (or very few) physical nonlinear elements with delayed feedback,
rather than thousands of discrete physical neurons -- that's the entire
point: nonlinearity + memory come from one cheap physical process instead of
a large matrix multiply. Time-division multiplexing turns that one nonlinear
node into many effective ("virtual") reservoir neurons.

Every physical reservoir in this package follows the same recipe:

  1. mask the input                (input encoding)
  2. drive one nonlinear physical process with it over time
  3. sample that process at N points per input step ("virtual nodes")
  4. treat those N samples as the reservoir state, same shape/role as the
     ESN's `states[t]` vector

This lets any physical reservoir be swapped in wherever ESN.run() is used,
train the same RidgeReadout, and get compared apples-to-apples against the
matrix-based ESN.
"""

import numpy as np
from abc import ABC, abstractmethod


class PhysicalReservoir(ABC):
    """
    Common interface for physical (or physically-inspired) reservoirs.

    n_reservoir here means "number of virtual nodes" -- the number of state
    values produced per input timestep, exactly analogous to ESN.n_reservoir,
    even though physically it may come from one element sampled repeatedly
    rather than many separate neurons.
    """

    def __init__(self, n_reservoir, seed=42):
        self.n_reservoir = n_reservoir
        self.rng = np.random.default_rng(seed)
        self.last_state = np.zeros(n_reservoir)

    @abstractmethod
    def run(self, inputs, initial_state=None):
        """
        inputs: shape (timesteps, n_inputs)
        returns: states, shape (timesteps, n_reservoir)
        Must set self.last_state to the final internal state, same
        convention as ESN.run(), so reservoirs are interchangeable across
        train/test boundaries.
        """
        raise NotImplementedError


def make_mask(n_virtual_nodes, n_inputs, levels=None, seed=42):
    """
    Build a fixed random input mask, shape (n_virtual_nodes, n_inputs).

    This is what turns a single scalar input sample into a piecewise
    "waveform" spanning the delay/settling window -- each virtual node sees
    the same input sample multiplied by a different fixed mask weight. It's
    the standard trick (Appeltant et al. 2011) that lets one physical
    nonlinear element behave like a reservoir of many neurons.

    levels: if given, mask values are drawn from this discrete set
            (e.g. [-1, 1] for a simple binary mask, common in optical
            setups where the mask is applied via an amplitude modulator).
            If None, mask values are drawn uniformly from [-1, 1].
    """
    rng = np.random.default_rng(seed)

    if levels is not None:
        return rng.choice(levels, size=(n_virtual_nodes, n_inputs)).astype(float)

    return rng.uniform(-1, 1, size=(n_virtual_nodes, n_inputs))