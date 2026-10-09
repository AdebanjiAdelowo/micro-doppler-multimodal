"""Handcrafted micro-Doppler features computed from a power spectrogram.

All features are functions of S[k, m] >= 0 (linear power, Doppler bin k at
frequency f_k, frame m). Definitions are in docs/METHODOLOGY.md, section 5.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def denoise(s: np.ndarray, k: float = 6.0) -> np.ndarray:
    """Suppress the noise floor: subtract it and zero cells below k times it.

    Most time-frequency cells of an indoor activity recording hold noise only.
    For complex Gaussian noise the power in a cell is exponentially
    distributed, whose mean is median / ln 2; that is the floor estimate
    (conservative when several range bins are summed, since the median then
    approaches the mean). Subtracting the floor alone leaves the upper tail of
    the noise, which still biases the Doppler centroid towards 0 Hz and
    inflates the bandwidth, so cells below ``k`` times the floor are zeroed
    (false-alarm probability exp(-k) = 0.25 % per noise-only cell for k = 6).
    The price is that components less than 10 log10(k) = 7.8 dB above the
    floor are discarded.
    """
    floor = np.median(s) / np.log(2.0)
    return np.where(s > k * floor, s - floor, 0.0)


def centroid_bandwidth(s: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame Doppler centroid f_c[m] and RMS bandwidth B[m]."""
    p = s / (s.sum(axis=0, keepdims=True) + _EPS)
    fc = (f[:, None] * p).sum(axis=0)
    bw = np.sqrt((((f[:, None] - fc[None, :]) ** 2) * p).sum(axis=0))
    return fc, bw


def envelopes(s: np.ndarray, f: np.ndarray, q: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
    """Lower/upper Doppler envelopes: frequencies below which a fraction q
    and 1-q of each frame's energy lies (percentile method)."""
    cdf = np.cumsum(s, axis=0)
    cdf = cdf / (cdf[-1:, :] + _EPS)
    lo = f[np.argmax(cdf >= q, axis=0)]
    hi = f[np.argmax(cdf >= 1.0 - q, axis=0)]
    return lo, hi


def _stats(x: np.ndarray, name: str) -> dict:
    return {
        f"{name}_mean": float(np.mean(x)),
        f"{name}_std": float(np.std(x)),
        f"{name}_min": float(np.min(x)),
        f"{name}_max": float(np.max(x)),
    }


def extract_features(s: np.ndarray, f: np.ndarray, frame_rate_hz: float) -> dict:
    """Feature dictionary for one spectrogram (n_doppler, n_frames)."""
    s = denoise(np.asarray(s, dtype=np.float64))
    total = s.sum() + _EPS
    fc, bw = centroid_bandwidth(s, f)
    lo, hi = envelopes(s, f)
    out: dict[str, float] = {}
    out |= _stats(fc, "centroid")
    out |= _stats(bw, "bandwidth")
    out |= _stats(lo, "env_lower")
    out |= _stats(hi, "env_upper")
    out |= _stats(hi - lo, "env_width")

    # Energy balance between Doppler signs and between slow/fast components.
    out["energy_pos_frac"] = float(s[f > 0].sum() / total)
    out["energy_high_frac"] = float(s[np.abs(f) > 0.4 * np.abs(f).max()].sum() / total)

    # Temporal energy profile e[m]: when and how concentrated the motion is.
    e = s.sum(axis=0)
    p_t = e / (e.sum() + _EPS)
    t = np.arange(e.size) / max(e.size - 1, 1)
    t_c = float((t * p_t).sum())
    out["time_centroid"] = t_c
    out["time_spread"] = float(np.sqrt((((t - t_c) ** 2) * p_t).sum()))
    out["energy_cv"] = float(e.std() / (e.mean() + _EPS))
    out["active_frac"] = float(np.mean(e > 0.1 * e.max()))

    # Entropy of the normalised time-frequency distribution.
    p = s / total
    out["tf_entropy"] = float(-(p * np.log(p + _EPS)).sum() / np.log(p.size))
    p_f = s.sum(axis=1) / total
    out["doppler_entropy"] = float(-(p_f * np.log(p_f + _EPS)).sum() / np.log(p_f.size))

    # Low-rank structure: energy captured by the leading singular values.
    sv = np.linalg.svd(np.sqrt(s), compute_uv=False)
    sv2 = sv**2 / (np.sum(sv**2) + _EPS)
    out["svd_energy_1"] = float(sv2[0])
    out["svd_energy_3"] = float(sv2[:3].sum())

    # Cadence: periodicity of the upper envelope (limb swing rate in gait).
    env = hi - hi.mean()
    spec = np.abs(np.fft.rfft(env * np.hanning(env.size))) ** 2
    freqs = np.fft.rfftfreq(env.size, d=1.0 / frame_rate_hz)
    band = (freqs >= 0.5) & (freqs <= 4.0)
    if band.any() and spec[1:].sum() > 0:
        k = np.argmax(np.where(band, spec, -1.0))
        out["cadence_hz"] = float(freqs[k])
        out["cadence_strength"] = float(spec[k] / (spec[1:].sum() + _EPS))
    else:
        out["cadence_hz"] = 0.0
        out["cadence_strength"] = 0.0
    return out


def feature_matrix(specs: np.ndarray, f: np.ndarray, frame_rate_hz: float):
    """Stack features of a batch of spectrograms; returns (X, names)."""
    rows = [extract_features(s, f, frame_rate_hz) for s in specs]
    names = list(rows[0])
    return np.array([[r[n] for n in names] for r in rows], dtype=np.float64), names
