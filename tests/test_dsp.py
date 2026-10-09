"""Numerical checks of the signal-processing chain against closed forms."""

import numpy as np
import pytest
import torch
from scipy import signal

from mdradar import dsp, synth
from mdradar.io import C

META = synth.GLASGOW_META
PRF = META.prf_hz


def test_range_bin_of_static_point_target():
    # R = k c / (2B): a target exactly on bin 20 must peak in bin 20.
    r = 20 * C / (2 * META.bandwidth_hz)
    beat = synth.beat_signal(np.full(8, r))
    prof = np.abs(dsp.range_profiles(beat, window="boxcar"))
    assert np.all(prof.argmax(axis=1) == 20)
    np.testing.assert_allclose(prof[:, 20], 1.0, rtol=1e-9)  # unit coherent gain
    np.testing.assert_allclose(dsp.range_axis(128, META.bandwidth_hz)[20], r)


def test_range_resolution_value():
    assert META.range_resolution_m == pytest.approx(0.3747, abs=1e-4)
    assert META.max_unambiguous_velocity_mps == pytest.approx(12.92, abs=0.01)


def test_stft_matches_scipy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000) + 1j * rng.standard_normal(2000)
    nw, hop, nfft = 128, 32, 256
    ours = dsp.stft(x, nw, hop, nfft)
    sft = signal.ShortTimeFFT(
        signal.get_window("hann", nw), hop=hop, fs=PRF, mfft=nfft, fft_mode="centered"
    )
    # SciPy centres frame 0 on sample 0; ours starts at sample 0 (shift nw/2).
    ref = sft.stft(x, p0=sft.m_num_mid // hop, p1=sft.m_num_mid // hop + ours.shape[-1])
    # The two differ by a per-bin linear phase (window origin); magnitudes agree.
    np.testing.assert_allclose(np.abs(ours), np.abs(ref), rtol=1e-9, atol=1e-9)


def test_stft_torch_equals_numpy():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 4, 1500)) + 1j * rng.standard_normal((3, 4, 1500))
    a = dsp.stft(x, 200, 40, 256)
    b = dsp.stft_torch(torch.from_numpy(x), 200, 40, 256).numpy()
    assert a.shape == b.shape == (3, 4, 256, 33)
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)


def test_parseval_for_rectangular_nonoverlapping_frames():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(1024) + 1j * rng.standard_normal(1024)
    z = dsp.stft(x, 64, 64, 64, window="boxcar")
    np.testing.assert_allclose((np.abs(z) ** 2).sum() / 64, (np.abs(x) ** 2).sum(), rtol=1e-10)


@pytest.mark.parametrize("v", [-3.0, -0.8, 1.5, 4.0])
def test_doppler_of_constant_velocity_target(v):
    # f_d = 2 v / lambda, recovered from the slow-time phase of the range bin.
    n = 2000
    beat = synth.beat_signal(synth.constant_velocity(5.0, v, n))
    prof = dsp.range_profiles(beat)
    x = prof[:, 8:20].T  # bins around the 5 m target (bin 13.3), track migration
    cfg = dsp.SpecConfig(n_window=256, hop=64, n_fft=1024, fd_max_hz=400.0)
    s, f = dsp.power_spectrogram(x, cfg, PRF)
    f_est = f[s.sum(axis=1).argmax()]
    expected = 2 * v / META.wavelength_m
    assert f_est == pytest.approx(expected, abs=dsp.resolution(256, 64, PRF)["freq_res_hz"])
    assert dsp.doppler_to_velocity(f_est, META.wavelength_m) == pytest.approx(v, abs=0.15)


def test_peak_doppler_of_sinusoidal_micro_motion():
    amp, rate, n = 0.15, 1.5, 4000
    beat = synth.beat_signal(synth.oscillating(5.0, amp, rate, n))
    x = dsp.range_profiles(beat)[:, 10:17].T
    cfg = dsp.SpecConfig(n_window=64, hop=8, n_fft=512, fd_max_hz=400.0)
    s, f = dsp.power_spectrogram(x, cfg, PRF)
    ridge = f[s.argmax(axis=0)]
    expected = 2 * (2 * np.pi * rate * amp) / META.wavelength_m  # 54.7 Hz
    assert np.abs(ridge).max() == pytest.approx(expected, rel=0.1)
    # ridge is symmetric about zero and periodic at the micro-motion rate
    assert abs(ridge.mean()) < 3.0


def test_mti_removes_static_clutter_and_keeps_moving_target():
    n = 4000
    t = np.arange(n) / PRF
    clutter = 100.0 * np.ones(n, dtype=complex)
    mover = np.exp(2j * np.pi * 60.0 * t)
    y = dsp.mti_highpass(clutter + mover, PRF)
    tail = y[1500:]  # only the transient of the mover switching on at t=0 remains
    np.testing.assert_allclose(np.abs(tail), 1.0, atol=1e-3)
    assert np.abs(dsp.mti_highpass(clutter, PRF)).max() < 1e-9  # no start-up transient
    multi = dsp.mti_highpass(np.stack([clutter, 2 * clutter])[:, None, :], PRF)
    assert multi.shape == (2, 1, n) and np.abs(multi).max() < 1e-9


def test_resolution_tradeoff_is_window_independent():
    prods = [dsp.resolution(n, n // 4, PRF)["tf_product"] for n in (64, 128, 256, 512)]
    np.testing.assert_allclose(prods, 1.5, rtol=1e-12)  # ENBW of periodic Hann
    assert dsp.resolution(200, 40, PRF)["freq_res_hz"] == pytest.approx(7.5)
    assert dsp.resolution(200, 40, PRF)["time_res_s"] == pytest.approx(0.2)


def test_db_image_range_and_peak_invariance():
    rng = np.random.default_rng(3)
    s = rng.random((64, 50)) + 1e-6
    a = dsp.to_db_image(s, 45.0)
    assert a.min() >= 0.0 and a.max() == pytest.approx(1.0)
    np.testing.assert_allclose(a, dsp.to_db_image(1e4 * s, 45.0), atol=1e-9)
    np.testing.assert_allclose(a, dsp.to_db_image(torch.from_numpy(s), 45.0).numpy(), atol=1e-6)


def test_model_input_shape_is_fixed_across_stft_settings():
    x = torch.randn(2, 6, 5000, dtype=torch.complex64)
    for nw, hop, nfft in [(64, 16, 256), (200, 40, 256), (512, 128, 512)]:
        cfg = dsp.SpecConfig(n_window=nw, hop=hop, n_fft=nfft)
        img = dsp.model_input(x, cfg, PRF)
        assert img.shape == (2, 1, 128, 128)
        assert torch.isfinite(img).all() and img.min() >= -1e-4 and img.max() <= 1 + 1e-4
