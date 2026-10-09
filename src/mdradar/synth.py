"""Ideal point-scatterer FMCW simulator.

Used ONLY to validate the signal-processing code against closed-form answers
(tests/, notebooks/01). No model is trained or evaluated on simulated data and
no simulated quantity appears in any reported result.
"""

from __future__ import annotations

import numpy as np

from .io import C, RecordingMeta

GLASGOW_META = RecordingMeta(
    carrier_hz=5.8e9, chirp_s=1e-3, samples_per_chirp=128, bandwidth_hz=400e6
)


def beat_signal(
    range_m: np.ndarray, meta: RecordingMeta = GLASGOW_META, amplitude: float = 1.0
) -> np.ndarray:
    """De-chirped return of one point scatterer.

    ``range_m[n]`` is the range at chirp n (stop-and-hop: constant within a
    chirp). With fast time t in [0, T):

        s[n, t] = A exp( j 2 pi ( (2 B R_n / (c T)) t + 2 f_c R_n / c ) )

    The first term places the target in range bin 2 B R / c; the second is the
    carrier phase whose chirp-to-chirp change is the Doppler shift 2 v / lambda.
    The sign of that phase (and hence of Doppler for an approaching target)
    is a receiver convention; here an *increasing* range gives positive Doppler.
    """
    r = np.asarray(range_m, dtype=np.float64)[:, None]
    n = meta.samples_per_chirp
    t = np.arange(n)[None, :] * (meta.chirp_s / n)
    phase = 2 * np.pi * (2 * meta.bandwidth_hz * r / (C * meta.chirp_s) * t + 2 * meta.carrier_hz * r / C)
    return (amplitude * np.exp(1j * phase)).astype(np.complex128)


def constant_velocity(r0: float, v: float, n_chirps: int, meta: RecordingMeta = GLASGOW_META):
    return r0 + v * np.arange(n_chirps) * meta.chirp_s


def oscillating(
    r0: float, amp_m: float, rate_hz: float, n_chirps: int, meta: RecordingMeta = GLASGOW_META
):
    """Sinusoidal micro-motion R(t) = r0 + a sin(2 pi f_m t); the peak Doppler
    is 2 (2 pi f_m a) / lambda."""
    t = np.arange(n_chirps) * meta.chirp_s
    return r0 + amp_m * np.sin(2 * np.pi * rate_hz * t)
