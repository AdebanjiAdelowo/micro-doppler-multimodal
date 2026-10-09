#!/usr/bin/env python
"""Train one neural model on one subject-independent fold and save a checkpoint.

    python scripts/train.py --model light_cnn --fold 0

The checkpoint stores the preprocessing and spectrogram parameters, the
temperature fitted on the validation subjects and the held-out test metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from mdradar import dsp
from mdradar.data import Cache, subject_kfold
from mdradar.experiments import N_CLASSES, evaluate, spectrogram_images
from mdradar.inference import save_checkpoint
from mdradar.io import CLASS_NAMES
from mdradar.metrics import fit_temperature
from mdradar.models import MODELS, count_parameters
from mdradar.tracking import Run
from mdradar.train import TrainConfig, pick_device, predict_logits, train_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/baselines.yaml"), help="source of spec/train settings")
    ap.add_argument("--model", choices=sorted(MODELS), default="light_cnn")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    spec = dsp.SpecConfig(**cfg.get("spec", {}))
    device = pick_device(args.device)
    cache = Cache(args.data)
    fold = subject_kfold(cache.manifest, cfg["n_splits"], cfg["split_seed"])[args.fold]
    tr, va, te = fold["train"], fold["val"], fold["test"]
    y = cache.labels
    run = Run("train", {"model": args.model, "fold": args.fold, "seed": args.seed, **cfg}, device)

    images = spectrogram_images(cache, spec)
    tc = TrainConfig(model=args.model, seed=args.seed, **cfg.get("train", {}))
    model, info = train_model(images[tr], y[tr], images[va], y[va], N_CLASSES, tc, device)
    for h in info.pop("history"):
        run.log(**h)
    lv, lt = predict_logits(model, images[va], device), predict_logits(model, images[te], device)
    metrics = evaluate(y[te], lt, y[va], lv, groups=cache.groups[te], n_boot=1000)
    split = {"protocol": "subject_kfold", "n_splits": cfg["n_splits"], "split_seed": cfg["split_seed"], "fold": args.fold}
    out = args.out or Path("checkpoints") / f"{args.model}.pt"
    save_checkpoint(out, model, model_name=args.model, spec=spec, class_names=CLASS_NAMES, cache_meta=cache.meta,
                    temperature=fit_temperature(lv, y[va]),
                    extra={"split": split, "seed": args.seed, "parameters": count_parameters(model),
                           "test_metrics": {k: metrics[k] for k in ("accuracy", "macro_f1", "balanced_accuracy")},
                           "git_commit": run.env["git_commit"], **info})
    run.finish({"split": split, "train_info": info, "test": metrics}, name=f"train_{args.model}_fold{args.fold}")
    print(json.dumps({k: metrics[k] for k in ("accuracy", "macro_f1", "balanced_accuracy")}, indent=2))
    print("checkpoint:", out)


if __name__ == "__main__":
    main()
