"""Radar signal processing: range compression, clutter filtering, STFT.

Every operation here is defined mathematically in docs/METHODOLOGY.md and
checked numerically in tests/test_dsp.py. Conventions:

* fast time  = samples within one chirp (axis -1 of the beat signal);
* slow time  = chirp index, sampled at the PRF (axis 0 of the beat signal);
* Doppler    f_d = 2 v / lambda, positive for a target approaching the radar
  *under the sign convention of the receiver*; the sign convention of the
  Glasgow hardware is not documented, so we never interpret the sign as a
  direction of travel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy import signal

from .io import C


# --------------------------------------------------------------------------
# Range compression
# --------------------------------------------------------------------------
def range_profiles(beat: np.ndarray, window: str = "hann") -> np.ndarray:
    """Fast-time DFT of de-chirped FMCW data.

    For a point target at range R the beat signal of one chirp is a complex
    tone at f_b = 2 B R / (c T), so the DFT over fast time is a range profile.
    With N samples per chirp of duration T the DFT bin spacing is 1/T, hence
    bin k corresponds to R_k = k c / (2 B).

    Returns the N/2 non-negative-frequency bins, shape (n_chirps, N/2). The
    window is normalised to unit coherent gain so amplitudes are comparable
    across window choices.
    """
    n = beat.shape[-1]
    w = signal.get_window(window, n, fftbins=True)
    w = w / w.mean()
    spec = np.fft.fft(beat * w, axis=-1) / n
    return spec[..., : n // 2]


def range_axis(n_samples: int, bandwidth_hz: float) -> np.ndarray:
    """Range of each bin returned by :func:`range_profiles`, in metres."""
    return np.arange(n_samples // 2) * C / (2.0 * bandwidth_hz)


# --------------------------------------------------------------------------
# Clutter (MTI) filter
# --------------------------------------------------------------------------
def mti_highpass(
    x: np.ndarray,
    prf_hz: float,
    cutoff_hz: float = 3.75,
    order: int = 4,
    zero_phase: bool = False,
    axis: int = -1,
) -> np.ndarray:
    """Butterworth high-pass along slow time to suppress static clutter.

    Walls and furniture return energy at zero Doppler that is tens of dB above
    the human return. A high-pass with cutoff f_c removes |f_d| < f_c, i.e.
    radial speeds below f_c * lambda / 2 (0.097 m/s at 5.8 GHz for the default
    3.75 Hz, which equals the dataset authors' normalised cutoff 0.0075).

    The default is causal (as a streaming system would run it). The filter
    state is initialised to its steady state for a constant input equal to the
    first sample, which removes the start-up transient a large static clutter
    level would otherwise cause.
    """
    sos = signal.butter(order, cutoff_hz, btype="highpass", fs=prf_hz, output="sos")
    if zero_phase:
        return signal.sosfiltfilt(sos, x, axis=axis)
    x = np.asarray(x)
    first = np.take(x, [0], axis=axis)
    zi = signal.sosfilt_zi(sos)  # (n_sections, 2)
    shape = [1] * x.ndim
    shape[axis] = 2
    zi = zi.reshape(zi.shape[0], *shape) * first[None]
    return signal.sosfilt(sos, x, axis=axis, zi=zi)[0]


# --------------------------------------------------------------------------
# Short-time Fourier transform
# --------------------------------------------------------------------------
def _window(name: str, n: int) -> np.ndarray:
    return signal.get_window(name, n, fftbins=True)


def stft(
    x: np.ndarray, n_window: int, hop: int, n_fft: int | None = None, window: str = "hann"
) -> np.ndarray:
    """Two-sided STFT of a complex signal along the last axis.

        X[k, m] = sum_{n=0}^{N_w-1} x[n + m H] w[n] exp(-j 2 pi k n / N_fft)

    No padding at the edges: frame m covers samples [mH, mH + N_w). The
    frequency axis is fftshift-ed so that index 0 is -PRF/2 (see
    :func:`doppler_axis`). Output shape (..., n_fft, n_frames).
    """
    n_fft = n_fft or n_window
    if n_fft < n_window:
        raise ValueError("n_fft must be >= n_window")
    if x.shape[-1] < n_window:
        raise ValueError("signal shorter than one window")
    w = _window(window, n_window)
    n_frames = 1 + (x.shape[-1] - n_window) // hop
    idx = np.arange(n_window)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[..., idx] * w  # (..., n_frames, n_window)
    spec = np.fft.fftshift(np.fft.fft(frames, n=n_fft, axis=-1), axes=-1)
    return np.swapaxes(spec, -1, -2)


def stft_torch(
    x: torch.Tensor, n_window: int, hop: int, n_fft: int | None = None, window: str = "hann"
) -> torch.Tensor:
    """PyTorch implementation of :func:`stft` (batched, same conventions)."""
    n_fft = n_fft or n_window
    w = torch.as_tensor(_window(window, n_window), dtype=x.real.dtype, device=x.device)
    frames = x.unfold(-1, n_window, hop) * w  # (..., n_frames, n_window)
    spec = torch.fft.fftshift(torch.fft.fft(frames, n=n_fft, dim=-1), dim=-1)
    return spec.transpose(-1, -2)


def doppler_axis(n_fft: int, prf_hz: float) -> np.ndarray:
    """Doppler frequency (Hz) of each row of the fftshift-ed STFT."""
    return np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / prf_hz))


def doppler_to_velocity(f_d: np.ndarray, wavelength_m: float) -> np.ndarray:
    """Radial velocity v = f_d * lambda / 2 (monostatic radar)."""
    return np.asarray(f_d) * wavelength_m / 2.0


def frame_times(n_samples: int, n_window: int, hop: int, prf_hz: float) -> np.ndarray:
    """Centre time (s) of each STFT frame."""
    n_frames = 1 + (n_samples - n_window) // hop
    return (np.arange(n_frames) * hop + n_window / 2.0) / prf_hz


def enbw_bins(window: str, n_window: int) -> float:
    """Equivalent noise bandwidth of the window, in DFT bins of size PRF/N_w."""
    w = _window(window, n_window)
    return float(n_window * np.sum(w**2) / np.sum(w) ** 2)


def resolution(n_window: int, hop: int, prf_hz: float, window: str = "hann") -> dict:
    """Time/frequency resolution implied by an STFT parameterisation.

    * ``time_res_s``   = N_w / PRF, the support of the analysis window;
    * ``freq_res_hz``  = ENBW * PRF / N_w, the equivalent noise bandwidth;
    * their product is ENBW (>= 1), independent of N_w: resolution can only be
      traded between the two axes, never improved on both (Gabor limit).
    * ``frame_period_s`` = H / PRF only sets how densely the time axis is
      *sampled*; overlap does not change either resolution.
    """
    t = n_window / prf_hz
    f = enbw_bins(window, n_window) * prf_hz / n_window
    return {
        "time_res_s": t,
        "freq_res_hz": f,
        "tf_product": t * f,
        "frame_period_s": hop / prf_hz,
        "overlap": 1.0 - hop / n_window,
    }


# --------------------------------------------------------------------------
# Micro-Doppler spectrogram
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SpecConfig:
    """Parameters of the micro-Doppler spectrogram (slow-time samples)."""

    n_window: int = 200  # 0.2 s at PRF 1 kHz
    hop: int = 40  # 80 % overlap
    n_fft: int = 256  # zero-padded DFT (interpolates, adds no resolution)
    window: str = "hann"
    fd_max_hz: float = 250.0  # keep |f_d| <= fd_max (6.5 m/s at 5.8 GHz)
    dynamic_range_db: float = 45.0
    out_size: tuple[int, int] = (128, 128)  # (doppler, time) fed to models

    def to_dict(self) -> dict:
        return asdict(self)


def power_spectrogram(x, cfg: SpecConfig, prf_hz: float):
    """Range-integrated power spectrogram.

    ``x`` has shape (..., n_range_bins, n_slow) and holds MTI-filtered range
    profiles. The STFT is taken per range bin and the *powers* are summed over
    bins (non-coherent integration): a person spans several 0.375 m range bins
    and migrates between them, so phases across bins are not aligned.

    Returns ``(S, f_d)`` with S of shape (..., n_doppler, n_frames), linear
    power, cropped to |f_d| <= cfg.fd_max_hz. Works on NumPy arrays and on
    torch tensors.
    """
    f = doppler_axis(cfg.n_fft, prf_hz)
    keep = np.abs(f) <= cfg.fd_max_hz
    if isinstance(x, torch.Tensor):
        z = stft_torch(x, cfg.n_window, cfg.hop, cfg.n_fft, cfg.window)
        s = (z.real**2 + z.imag**2).sum(dim=-3)
        return s[..., torch.as_tensor(keep, device=s.device), :], f[keep]
    z = stft(x, cfg.n_window, cfg.hop, cfg.n_fft, cfg.window)
    s = (z.real**2 + z.imag**2).sum(axis=-3)
    return s[..., keep, :], f[keep]


def to_db_image(s, dynamic_range_db: float = 45.0):
    """Map linear power to [0, 1]: dB relative to the per-recording peak,
    clipped ``dynamic_range_db`` below it.

    Peak normalisation discards absolute received power, which depends on
    range and radar cross-section rather than on the activity.
    """
    if isinstance(s, torch.Tensor):
        peak = s.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
        db = 10.0 * torch.log10((s / peak).clamp_min(1e-30))
        return (db.clamp_min(-dynamic_range_db) + dynamic_range_db) / dynamic_range_db
    peak = np.maximum(s.max(axis=(-2, -1), keepdims=True), 1e-30)
    db = 10.0 * np.log10(np.maximum(s / peak, 1e-30))
    return (np.maximum(db, -dynamic_range_db) + dynamic_range_db) / dynamic_range_db


def model_input(x: torch.Tensor, cfg: SpecConfig, prf_hz: float) -> torch.Tensor:
    """Batched front end: (B, n_range, n_slow) complex -> (B, 1, H, W) float.

    The spectrogram is resampled to ``cfg.out_size`` with area/bilinear
    interpolation so that models see a fixed grid for every STFT setting.
    Resampling does not add resolution; it only makes ablations comparable.
    """
    s, _ = power_spectrogram(x, cfg, prf_hz)
    img = to_db_image(s, cfg.dynamic_range_db).unsqueeze(1).float()
    if tuple(img.shape[-2:]) != tuple(cfg.out_size):
        img = torch.nn.functional.interpolate(
            img, size=cfg.out_size, mode="bilinear", align_corners=False, antialias=True
        )
    return img
