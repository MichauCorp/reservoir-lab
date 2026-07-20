"""
Template for a *real* physical reservoir, once you have actual hardware
(an optical delay line, a piezo/acoustic setup, a memristor crossbar, etc).

This intentionally does NOT run out of the box -- there's no way to know
your exact hardware I/O without your setup in front of us. It exists so
that hardware, once you have it, slots into the exact same interface as
OptoelectronicReservoir / AcousticReservoir / ESN, and every experiment
script, the Readout class, and plot_results() keep working unmodified.

Fill in `_drive_and_read()` with whatever your actual I/O layer is, e.g.:

  - Audio-based acoustic reservoir: `sounddevice` to play F(t) out one
    channel and record x(t) back in on others (mic array / accelerometers)
  - Optical/optoelectronic: a DAQ card (`nidaqmx`) or microcontroller over
    serial (`pyserial`) driving a laser/modulator, digitizing a
    photodetector's output
  - Memristor crossbar: an SMU or custom driver board, again usually over
    serial or a vendor SDK

The masking / virtual-node logic in OptoelectronicReservoir.run() is
reusable as-is if your hardware is also a delay-based single-node system;
this class assumes instead that your hardware directly exposes N readout
channels (one per reservoir neuron), which is the simpler case to wire up.
"""

import numpy as np

from .base import PhysicalReservoir


class HardwareReservoir(PhysicalReservoir):

    def __init__(self, n_inputs, n_reservoir, seed=42):
        super().__init__(n_reservoir=n_reservoir, seed=seed)
        self.n_inputs = n_inputs
        # TODO: open your device connection here, e.g.:
        # self.device = nidaqmx.Task()  /  serial.Serial("/dev/ttyUSB0", ...)

    def _drive_and_read(self, u):
        """
        u: shape (n_inputs,) -- one input sample.
        Must return: shape (n_reservoir,) -- the physical readout for this
        timestep (e.g. n_reservoir photodetector/accelerometer channels).

        Replace this with real hardware I/O. Left unimplemented on purpose.
        """
        raise NotImplementedError(
            "Wire this up to your actual hardware's write/read calls."
        )

    def run(self, inputs, initial_state=None):
        n_steps = len(inputs)
        states = np.zeros((n_steps, self.n_reservoir))

        for t, u in enumerate(inputs):
            states[t] = self._drive_and_read(u)

        self.last_state = states[-1].copy()
        return states

    def close(self):
        """Release the hardware connection. Call when you're done."""
        # TODO: self.device.close()
        pass