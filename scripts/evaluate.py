#!/usr/bin/env python
"""Evaluate a checkpoint on the held-out fold it was trained against.

    python scripts/evaluate.py checkpoints/light_cnn.pt [--snr-db 10]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mdradar import metrics as M
from mdradar.data import Cache, subject_kfold
from mdradar.experiments import N_CLASSES, evaluate, spectrogram_images
from mdradar.inference import Predictor
from mdradar.train import pick_device, predict_logits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--snr-db", type=float, default=None, help="add white noise at this SNR before the STFT")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    pred = Predictor.load(args.checkpoint, str(device))
    cache = Cache(args.data)
    if cache.meta["range_bins"] != pred.meta["range_bins"]:
        raise SystemExit("cache and checkpoint were built with different range bins")
    sp = pred.ckpt["extra"]["split"]
    te = subject_kfold(cache.manifest, sp["n_splits"], sp["split_seed"])[sp["fold"]]["test"]
    y = cache.labels[te]
    images = spectrogram_images(cache, pred.spec, te, snr_db=args.snr_db)
    logits = predict_logits(pred.model, images, device) / pred.temperature
    res = evaluate(y, logits, groups=cache.groups[te], n_boot=1000)
    res.pop("selective_raw")
    prob = torch.softmax(torch.from_numpy(logits), 1).numpy()
    report = {
        "checkpoint": str(args.checkpoint), "split": sp, "snr_db": args.snr_db, "n_test": int(len(te)),
        "n_test_subject_numbers": int(len(np.unique(cache.groups[te]))),
        **res, "selective": M.selective_metrics(y, prob),
        "confusion": M.confusion(y, prob.argmax(1), N_CLASSES).tolist(),
        "per_class": M.per_class_report(y, prob.argmax(1), pred.class_names),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
