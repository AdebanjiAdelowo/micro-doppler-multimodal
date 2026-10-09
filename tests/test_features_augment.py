import numpy as np
import pytest
import torch

from mdradar import augment, dsp, features, synth

PRF = synth.GLASGOW_META.prf_hz


def _tone_spec(f0, n=4000, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / PRF
    x = np.exp(2j * np.pi * f0 * t) + noise * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return dsp.power_spectrogram(x[None, :], dsp.SpecConfig(), PRF)


def test_centroid_and_bandwidth_of_pure_tone():
    s, f = _tone_spec(80.0)
    fc, bw = features.centroid_bandwidth(s, f)
    np.testing.assert_allclose(fc, 80.0, atol=0.5)
    assert bw.max() < 8.0  # of the order of the 7.5 Hz window ENBW


def test_noise_floor_subtraction_removes_centroid_bias():
    s, f = _tone_spec(80.0, noise=1.0)
    raw, _ = features.centroid_bandwidth(s, f)
    den, _ = features.centroid_bandwidth(features.denoise(s), f)
    assert abs(den.mean() - 80.0) < abs(raw.mean() - 80.0)
    assert abs(den.mean() - 80.0) < 3.0


def test_envelopes_bracket_a_two_tone_signal():
    t = np.arange(4000) / PRF
    x = np.exp(2j * np.pi * 100 * t) + np.exp(-2j * np.pi * 60 * t)
    s, f = dsp.power_spectrogram(x[None, :], dsp.SpecConfig(), PRF)
    lo, hi = features.envelopes(s, f)
    assert np.all(np.abs(lo + 60) < 8) and np.all(np.abs(hi - 100) < 8)


def test_cadence_recovers_micro_motion_rate():
    beat = synth.beat_signal(synth.oscillating(5.0, 0.12, 1.5, 5000))
    x = dsp.range_profiles(beat)[:, 10:17].T
    cfg = dsp.SpecConfig(n_window=100, hop=20)
    s, f = dsp.power_spectrogram(x, cfg, PRF)
    feats = features.extract_features(s, f, frame_rate_hz=PRF / cfg.hop)
    assert feats["cadence_hz"] == pytest.approx(1.5, abs=0.25)
    assert all(np.isfinite(v) for v in feats.values())


def test_feature_matrix_shape_and_names():
    s, f = _tone_spec(30.0, noise=0.5)
    x, names = features.feature_matrix(np.stack([s, s]), f, 25.0)
    assert x.shape == (2, len(names)) and len(set(names)) == len(names)
    np.testing.assert_array_equal(x[0], x[1])


@pytest.mark.parametrize("snr_db", [-10.0, 0.0, 20.0])
def test_awgn_hits_requested_snr(snr_db):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 8, 20000, dtype=torch.complex64, generator=g) * 3.0
    y = augment.add_awgn(x, snr_db, generator=g)
    p_x = augment.signal_power(x)
    p_n = augment.signal_power(y - x)
    measured = 10 * torch.log10(p_x / p_n)
    assert torch.allclose(measured, torch.full_like(measured, snr_db), atol=0.15)


def test_awgn_is_circular_and_white():
    g = torch.Generator().manual_seed(1)
    x = torch.ones(1, 1, 200000, dtype=torch.complex64)
    n = (augment.add_awgn(x, 0.0, generator=g) - x).flatten()
    assert abs(float(n.real.var() / n.imag.var()) - 1) < 0.03
    assert abs(float((n.real * n.imag).mean())) < 0.01
    assert abs(float((n[1:] * n[:-1].conj()).real.mean())) < 0.01


def test_time_shift_moves_and_zero_fills():
    x = torch.arange(1, 11, dtype=torch.float32).to(torch.complex64)[None]
    np.testing.assert_array_equal(augment.time_shift(x, 3)[0].real.numpy(), [0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
    np.testing.assert_array_equal(augment.time_shift(x, -2)[0].real.numpy(), [3, 4, 5, 6, 7, 8, 9, 10, 0, 0])


@pytest.mark.parametrize("factor", [0.9, 1.1])
def test_speed_scale_multiplies_doppler(factor):
    t = np.arange(5000) / PRF
    x = torch.from_numpy(np.exp(2j * np.pi * 50.0 * t))[None, None]
    y = augment.speed_scale(x, factor)[0].numpy()
    s, f = dsp.power_spectrogram(y, dsp.SpecConfig(n_fft=1024), PRF)
    assert f[s[:, :60].sum(axis=1).argmax()] == pytest.approx(50.0 * factor, abs=1.0)
    if factor > 1:  # the motion finishes early; the remainder is empty
        assert np.abs(y[0, int(5000 / factor) + 2 :]).max() == 0.0


def test_signal_augment_is_reproducible():
    x = torch.randn(3, 4, 5000, dtype=torch.complex64, generator=torch.Generator().manual_seed(0))
    a = augment.SignalAugment(seed=7)(x)
    b = augment.SignalAugment(seed=7)(x)
    assert a.shape == x.shape and torch.equal(a, b) and not torch.equal(a, x)
