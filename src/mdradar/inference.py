"""Checkpoint format and single-recording inference used by the API and demo."""

from __future__ import annotations

import io as _io
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax

from . import data, dsp
from .io import parse_dat_bytes
from .models import build_model

INTENDED_USE = (
    "Research demonstration of human-activity classification from a 5 s radar recording made with "
    "the acquisition setup of the training data. The output is an activity label with a confidence. "
    "It says nothing about who a person is or what they intend, and it is not validated for safety, "
    "medical, security or surveillance use."
)


def save_checkpoint(path, model, *, model_name, spec: dsp.SpecConfig, class_names, cache_meta, temperature,
                    extra: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "model_name": model_name, "spec": spec.to_dict(), "class_names": list(class_names),
        "cache_meta": cache_meta, "temperature": float(temperature), "extra": extra or {},
    }, path)


@dataclass
class Prediction:
    label: str
    confidence: float
    abstained: bool
    probabilities: dict[str, float]
    timings_ms: dict[str, float]
    image: np.ndarray  # (n_doppler, n_time) model input in [0, 1]


class Predictor:
    def __init__(self, ckpt: dict, device: str = "cpu"):
        self.ckpt = ckpt
        self.spec = dsp.SpecConfig(**{**ckpt["spec"], "out_size": tuple(ckpt["spec"]["out_size"])})
        self.class_names = ckpt["class_names"]
        self.meta = ckpt["cache_meta"]
        self.temperature = ckpt["temperature"]
        self.device = torch.device(device)
        h, w = self.spec.out_size
        kw = {"n_doppler": h} if ckpt["model_name"] in ("temporal_cnn", "transformer") else {}
        if ckpt["model_name"] == "transformer":
            kw["n_time"] = w
        self.model = build_model(ckpt["model_name"], len(self.class_names), **kw)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> Predictor:
        return cls(torch.load(path, map_location="cpu", weights_only=True), device)

    @property
    def n_range(self) -> int:
        lo, hi = self.meta["range_bins"]
        return hi - lo

    def describe(self) -> dict:
        return {
            "model": self.ckpt["model_name"], "classes": self.class_names, "temperature": self.temperature,
            "spectrogram": self.ckpt["spec"], "preprocessing": {k: self.meta[k] for k in ("range_bins", "n_slow", "mti", "prf_hz")},
            "training": self.ckpt.get("extra", {}), "intended_use": INTENDED_USE,
        }

    @torch.no_grad()
    def predict_range_bins(self, x: np.ndarray, abstain_below: float = 0.0, timings: dict | None = None) -> Prediction:
        """x: complex (n_range_bins, n_slow) MTI-filtered range profiles."""
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[0] != self.n_range or not np.iscomplexobj(x):
            raise ValueError(f"expected a complex array of shape ({self.n_range}, n_slow); got {x.dtype} {x.shape}")
        n_slow = self.meta["n_slow"]
        if x.shape[1] < self.spec.n_window:
            raise ValueError("recording shorter than one STFT window")
        buf = np.zeros((self.n_range, n_slow), dtype=np.complex64)
        buf[:, : min(n_slow, x.shape[1])] = x[:, :n_slow]
        timings = dict(timings or {})
        t0 = time.perf_counter()
        img = dsp.model_input(torch.from_numpy(buf)[None], self.spec, self.meta["prf_hz"])
        t1 = time.perf_counter()
        logits = self.model(img.to(self.device)).float().cpu().numpy()
        t2 = time.perf_counter()
        prob = softmax(logits / self.temperature, axis=1)[0]
        timings |= {"stft_ms": (t1 - t0) * 1e3, "model_ms": (t2 - t1) * 1e3}
        k = int(prob.argmax())
        conf = float(prob[k])
        return Prediction(
            label=self.class_names[k], confidence=conf, abstained=conf < abstain_below,
            probabilities={c: float(p) for c, p in zip(self.class_names, prob, strict=True)},
            timings_ms=timings, image=img[0, 0].numpy(),
        )

    def predict_dat(self, raw: bytes, abstain_below: float = 0.0) -> Prediction:
        """Full chain from the raw text recording of the Glasgow dataset."""
        t0 = time.perf_counter()
        rec = parse_dat_bytes(raw)
        if abs(rec.meta.prf_hz - self.meta["prf_hz"]) > 1e-6 * self.meta["prf_hz"]:
            raise ValueError(f"PRF {rec.meta.prf_hz:.1f} Hz differs from the training PRF {self.meta['prf_hz']:.1f} Hz")
        t1 = time.perf_counter()
        sig, _ = data.preprocess_recording(
            rec, tuple(self.meta["range_bins"]), self.meta["n_slow"], self.meta["range_window"],
            self.meta["mti"]["cutoff_hz"], self.meta["mti"]["order"])
        t2 = time.perf_counter()
        return self.predict_range_bins(sig, abstain_below, {"parse_ms": (t1 - t0) * 1e3, "range_mti_ms": (t2 - t1) * 1e3})

    def predict_file(self, name: str, raw: bytes, abstain_below: float = 0.0) -> Prediction:
        if name.lower().endswith(".npy"):
            return self.predict_range_bins(np.load(_io.BytesIO(raw), allow_pickle=False), abstain_below)
        return self.predict_dat(raw, abstain_below)
