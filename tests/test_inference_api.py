"""API tests use a randomly initialised model: they check plumbing, not accuracy."""

import io as _io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from mdradar import dsp, synth
from mdradar.inference import Predictor, save_checkpoint
from mdradar.io import CLASS_NAMES
from mdradar.models import build_model
from mdradar.serve.api import create_app

META = {"range_bins": [5, 17], "n_slow": 1000, "range_window": "hann", "prf_hz": 1000.0,
        "mti": {"cutoff_hz": 3.75, "order": 4}}


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("ckpt") / "m.pt"
    save_checkpoint(path, build_model("light_cnn", 6), model_name="light_cnn", spec=dsp.SpecConfig(),
                    class_names=CLASS_NAMES, cache_meta=META, temperature=1.7)
    return path


def _dat_bytes(n_chirps=1000):
    beat = synth.beat_signal(synth.oscillating(4.0, 0.1, 1.2, n_chirps))
    vals = np.concatenate([[5.8e9, 1.0, 128, 4e8], beat.ravel()])
    return "\n".join(f"{v.real:.6g}{v.imag:+.6g}i" for v in vals).encode()


def test_predictor_on_raw_recording(ckpt):
    p = Predictor.load(ckpt).predict_dat(_dat_bytes())
    assert p.label in CLASS_NAMES and sum(p.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    assert p.image.shape == (128, 128) and {"parse_ms", "range_mti_ms", "stft_ms", "model_ms"} <= set(p.timings_ms)


def test_api_endpoints(ckpt):
    client = TestClient(create_app(ckpt))
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    info = client.get("/model").json()
    assert info["classes"] == CLASS_NAMES and "intend" in info["intended_use"]

    r = client.post("/predict", files={"file": ("1P01A01R1.dat", _dat_bytes())})
    assert r.status_code == 200
    body = r.json()
    assert body["activity"] == body["top_class"] and not body["abstained"]

    r = client.post("/predict?abstain_below=1.0", files={"file": ("x.dat", _dat_bytes())})
    assert r.json()["abstained"] and r.json()["activity"] is None

    buf = _io.BytesIO()
    np.save(buf, (np.random.default_rng(0).standard_normal((12, 1000)) + 0j).astype(np.complex64))
    assert client.post("/predict", files={"file": ("x.npy", buf.getvalue())}).status_code == 200

    buf = _io.BytesIO()
    np.save(buf, np.zeros((3, 1000), dtype=np.complex64))
    assert client.post("/predict", files={"file": ("bad.npy", buf.getvalue())}).status_code == 422
    assert client.post("/predict", files={"file": ("bad.dat", b"not numbers")}).status_code == 422


def test_api_without_checkpoint_reports_unavailable(tmp_path):
    client = TestClient(create_app(tmp_path / "missing.pt"))
    assert client.get("/health").json()["model_loaded"] is False
    assert client.get("/model").status_code == 503
