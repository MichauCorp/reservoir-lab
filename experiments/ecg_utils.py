"""
Shared ECG utilities for experiments using scipy's bundled ECG recording.

Provides:
  - verify_pac(): phase-amplitude coupling verification
  - build_ecg_task(): task construction from a configurable time window

Used by exp18 (real-signal validation) and exp27 (segment-scan for high
modulation depth).
"""

import numpy as np
from scipy.datasets import electrocardiogram
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, decimate, hilbert, sosfiltfilt

FS_NATIVE = 360.0
DECIMATE_FACTOR = 5
FS_DECIMATED = FS_NATIVE / DECIMATE_FACTOR  # 72 Hz

# Filters (matching exp18's exact choices)
_sos_low = butter(4, 2.0, btype="low", fs=FS_NATIVE, output="sos")
_sos_fast = butter(4, [8.0, 30.0], btype="band", fs=FS_NATIVE, output="sos")


def verify_pac(low, fast_band):
    """Confirm genuine phase-amplitude coupling in a recording segment.

    Bins the fast-band envelope by the slow component's instantaneous phase
    and returns (value_correlation, modulation_depth), matching exp18's
    methodology exactly.

    Parameters
    ----------
    low : array, shape (n_samples,)
        Slow (<2 Hz) component.
    fast_band : array, shape (n_samples,)
        Fast (8-30 Hz) bandpassed component.

    Returns
    -------
    value_corr : float
        Pearson correlation between slow and smoothed fast envelope.
    modulation_depth : float
        (max - min) / mean of envelope across phase bins.
    """
    envelope = np.abs(hilbert(fast_band))
    envelope_smooth = uniform_filter1d(envelope, size=int(FS_NATIVE * 0.3))
    value_corr = np.corrcoef(low, envelope_smooth)[0, 1]

    slow_phase = np.angle(hilbert(low - low.mean()))
    bins = np.linspace(-np.pi, np.pi, 19)
    idx = np.digitize(slow_phase, bins)
    means = np.array([envelope_smooth[idx == b].mean() for b in range(1, 19)])
    modulation_depth = (means.max() - means.min()) / means.mean()
    return value_corr, modulation_depth


def build_ecg_task(seed, start_seconds=0.0, segment_seconds=120,
                   noise_std=0.3, train_frac=0.7):
    """Build a supervised task from a window of the scipy ECG recording.

    Parameters
    ----------
    seed : int
        RNG seed for reproducible noise.
    start_seconds : float
        Start time of the window to extract (default 0.0 = first 120s,
        matching exp18's original behavior).
    segment_seconds : float
        Length of the window in seconds (default 120).
    noise_std : float
        Standard deviation of additive Gaussian noise (default 0.3).
    train_frac : float
        Fraction of data used for training (default 0.7).

    Returns
    -------
    u_train : ndarray, shape (n_train, 1)
        Noisy input (clean ECG + noise).
    y_train : ndarray, shape (n_train, 2)
        Targets: [slow_target, fast_target].
    u_test : ndarray, shape (n_test, 1)
        Noisy input for test split.
    y_test : ndarray, shape (n_test, 2)
        Targets for test split.
    value_corr : float
        Value correlation (PAC verification).
    modulation_depth : float
        Phase-binned modulation depth.
    """
    ecg = electrocardiogram()
    start_native = int(start_seconds * FS_NATIVE)
    n_native = int(segment_seconds * FS_NATIVE)
    seg = ecg[start_native:start_native + n_native]
    seg = (seg - seg.mean()) / seg.std()

    low_native = sosfiltfilt(_sos_low, seg)
    fast_native = sosfiltfilt(_sos_fast, seg)

    value_corr, mod_depth = verify_pac(low_native, fast_native)

    low = decimate(low_native, DECIMATE_FACTOR, zero_phase=True)
    fast = decimate(fast_native, DECIMATE_FACTOR, zero_phase=True)
    clean = decimate(seg, DECIMATE_FACTOR, zero_phase=True)

    rng = np.random.default_rng(seed)
    noisy_input = clean + rng.normal(0, noise_std, len(clean))

    u = noisy_input.reshape(-1, 1)
    y = np.stack([low, fast], axis=1)
    split = int(len(u) * train_frac)
    return u[:split], y[:split], u[split:], y[split:], value_corr, mod_depth