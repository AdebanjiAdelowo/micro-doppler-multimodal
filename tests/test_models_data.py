import numpy as np
import pandas as pd
import pytest
import torch

from mdradar import data, io, models, synth
from mdradar.train import TrainConfig, predict_logits, train_model


@pytest.mark.parametrize("name", sorted(models.MODELS))
def test_model_shapes_and_gradients(name):
    kw = {"n_doppler": 128} if name in ("temporal_cnn", "transformer") else {}
    m = models.build_model(name, 6, **kw)
    x = torch.randn(3, 1, 128, 128)
    out = m(x)
    assert out.shape == (3, 6) and m.features(x).shape == (3, m.feat_dim)
    out.sum().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
    assert models.count_parameters(m) > 0


@pytest.mark.parametrize("kind", ["early", "feature", "cross_attention"])
def test_fusion_models_use_both_inputs(kind):
    m = models.build_fusion(kind, 6).eval()
    a, b = torch.randn(2, 1, 128, 128), torch.randn(2, 1, 128, 128)
    with torch.no_grad():
        base = m(a, b)
        assert base.shape == (2, 6)
        assert not torch.allclose(base, m(a, torch.randn_like(b)), atol=1e-5)
        assert not torch.allclose(base, m(torch.randn_like(a), b), atol=1e-5)


def test_late_fusion_is_a_probability():
    pa, pb = np.array([[0.7, 0.3]]), np.array([[0.2, 0.8]])
    np.testing.assert_allclose(models.late_fusion(pa, pb, 0.25), [[0.325, 0.675]])


def test_training_learns_a_separable_toy_problem():
    g = torch.Generator().manual_seed(0)
    x = 0.3 * torch.randn(120, 1, 32, 32, generator=g)
    y = np.arange(120) % 2
    x[y == 1, :, :16] += 1.0
    cfg = TrainConfig(model="light_cnn", epochs=8, batch_size=16, patience=8)
    model, info = train_model(x[:80], y[:80], x[80:100], y[80:100], 2, cfg, torch.device("cpu"))
    acc = (predict_logits(model, x[100:], torch.device("cpu")).argmax(1) == y[100:]).mean()
    assert acc > 0.9 and info["best_epoch"] >= 0


def test_parse_filename_and_dat_roundtrip():
    lab = io.parse_filename("some/dir/3P07A03R2.dat")
    assert (lab["activity"], lab["subject"], lab["repetition"], lab["activity_name"]) == (3, 7, 2, "standing_up")
    with pytest.raises(ValueError):
        io.parse_filename("readme.txt")
    beat = synth.beat_signal(np.full(3, 4.0))
    vals = np.concatenate([[5.8e9, 1.0, 128, 4e8], beat.ravel()])
    text = "\n".join(f"{v.real:.9g}{v.imag:+.9g}i" for v in vals).encode()
    rec = io.parse_dat_bytes(text + b"\n0.5+0.5i\n")  # trailing partial chirp is dropped
    assert rec.beat.shape == (3, 128) and rec.meta.prf_hz == pytest.approx(1000.0)
    assert rec.meta.bandwidth_hz == 4e8 and rec.duration_s == pytest.approx(0.003)
    np.testing.assert_allclose(rec.beat, beat, atol=1e-6)


def test_preprocess_recording_pads_and_selects_bins():
    beat = synth.beat_signal(synth.constant_velocity(5.0, 1.0, 300))
    rec = io.Recording(beat.astype(np.complex64), synth.GLASGOW_META)
    out, power = data.preprocess_recording(rec, (8, 20), n_slow=400)
    assert out.shape == (12, 400) and out.dtype == np.complex64 and power.shape == (64,)
    assert np.all(out[:, 300:] == 0) and 8 <= power.argmax() < 20


def _manifest(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for session in range(1, 5):
        for subject in rng.choice(np.arange(1, 30), 10, replace=False):
            for label in range(6):
                for _rep in range(3):
                    rows.append({"session": session, "group": int(subject), "label": label})
    return pd.DataFrame(rows)


def test_subject_kfold_has_no_leakage_and_covers_everything():
    man = _manifest()
    folds = data.subject_kfold(man, 5, seed=0)
    tests = np.concatenate([f["test"] for f in folds])
    assert sorted(tests) == list(range(len(man)))
    for f in folds:
        g = man["group"].to_numpy()
        assert not set(g[f["train"]]) & set(g[f["test"]])
        assert not set(g[f["train"]]) & set(g[f["val"]])
        assert not set(g[f["val"]]) & set(g[f["test"]])
        assert set(man["label"].to_numpy()[f["test"]]) == set(range(6))


def test_session_split_drops_recurring_subject_numbers():
    man = _manifest()
    sp = data.session_split(man, [4])
    g, s = man["group"].to_numpy(), man["session"].to_numpy()
    assert set(s[sp["test"]]) == {4} and 4 not in set(s[sp["train"]]) | set(s[sp["val"]])
    assert not set(g[sp["test"]]) & (set(g[sp["train"]]) | set(g[sp["val"]]))
    assert sp["dropped_test_recordings"] == int((s == 4).sum()) - len(sp["test"]) > 0


def test_assert_disjoint_detects_leak():
    man = _manifest()
    bad = {"train": np.array([0, 1]), "val": np.array([2]), "test": np.array([3])}
    with pytest.raises(AssertionError):
        data.assert_disjoint(man, bad)
