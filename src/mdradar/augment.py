"""Signal-domain noise injection and physically justified augmentation.

Everything operates on complex slow-time signals of shape (..., n_slow), i.e.
*before* the STFT, so that each transform corresponds to something that can
happen during acquisition. Transforms that would change the label are listed
in docs/METHODOLOGY.md (section 7) and deliberately not implemented:
time reversal and Doppler-sign flipping both map "sitting down" onto
something resembling "standing up".
"""

from __future__ import annotations

import torch


def signal_power(x: torch.Tensor) -> torch.Tensor:
    """Mean power per recording over all trailing (range, slow-time) samples."""
    return (x.real**2 + x.imag**2).mean(dim=(-2, -1), keepdim=True)


def add_awgn(x: torch.Tensor, snr_db: float | torch.Tensor, generator=None) -> torch.Tensor:
    """Add circular complex white Gaussian noise (receiver thermal noise).

    The noise variance is set per recording so that

        10 log10( P_x / sigma^2 ) = snr_db,   P_x = mean |x|^2.

    P_x is the power of the *measured* signal, which already contains the
    sensor's own noise, so ``snr_db`` is the ratio of measured signal to
    *added* noise: an upper bound on the true SNR after injection.
    """
    snr = torch.as_tensor(snr_db, dtype=x.real.dtype, device=x.device)
    while snr.dim() < x.dim() and snr.dim() > 0:
        snr = snr.unsqueeze(-1)
    sigma2 = signal_power(x) / (10.0 ** (snr / 10.0))
    noise = torch.randn(x.shape, dtype=x.dtype, generator=generator, device=x.device)
    # torch.randn for complex dtypes has unit total variance (1/2 per component).
    return x + noise * torch.sqrt(sigma2)


def time_shift(x: torch.Tensor, shift: int) -> torch.Tensor:
    """Translate along slow time, filling the vacated samples with zeros.

    Models a different trigger instant of the recording. The part of the
    activity shifted out of the window is lost, as it would be in practice.
    """
    if shift == 0:
        return x
    out = torch.zeros_like(x)
    if shift > 0:
        out[..., shift:] = x[..., :-shift]
    else:
        out[..., :shift] = x[..., -shift:]
    return out


def speed_scale(x: torch.Tensor, factor: float) -> torch.Tensor:
    """Resample slow time so the motion is executed ``factor`` times faster.

    If x(t) is the return of a motion R(t), performing the same motion faster,
    R(factor * t), gives x(factor * t): the signature is compressed in time by
    ``factor`` and every Doppler shift is multiplied by ``factor``. Linear
    interpolation of the complex samples; output keeps the input length (zero
    padded when factor > 1, truncated when factor < 1).
    """
    n = x.shape[-1]
    pos = torch.arange(n, device=x.device, dtype=x.real.dtype) * factor
    valid = pos <= n - 1
    i0 = pos.clamp(max=n - 1).floor().long().clamp(max=n - 2)
    frac = (pos.clamp(max=n - 1) - i0).to(x.real.dtype)
    y = x[..., i0] * (1 - frac) + x[..., i0 + 1] * frac
    return y * valid


class SignalAugment:
    """Random composition used during training (per recording)."""

    def __init__(
        self,
        snr_db_range: tuple[float, float] | None = (0.0, 30.0),
        max_shift: int = 250,
        speed_range: tuple[float, float] | None = (0.9, 1.1),
        p_noise: float = 0.5,
        seed: int = 0,
    ):
        self.snr_db_range = snr_db_range
        self.max_shift = max_shift
        self.speed_range = speed_range
        self.p_noise = p_noise
        self.gen = torch.Generator().manual_seed(seed)

    def _u(self, lo: float, hi: float) -> float:
        return float(lo + (hi - lo) * torch.rand((), generator=self.gen))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_range, n_slow) complex, on CPU."""
        out = []
        for xi in x:
            if self.speed_range is not None:
                xi = speed_scale(xi, self._u(*self.speed_range))
            if self.max_shift:
                s = int(torch.randint(-self.max_shift, self.max_shift + 1, (), generator=self.gen))
                xi = time_shift(xi, s)
            if self.snr_db_range is not None and self._u(0, 1) < self.p_noise:
                xi = add_awgn(xi, self._u(*self.snr_db_range), generator=self.gen)
            out.append(xi)
        return torch.stack(out)
