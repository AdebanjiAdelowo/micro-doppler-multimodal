"""Training and inference loops shared by every neural experiment."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from .models import build_fusion, build_model


def pick_device(prefer: str = "auto") -> torch.device:
    """CUDA if present, else Apple MPS, else CPU. Never assumes CUDA."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TrainConfig:
    model: str = "light_cnn"
    fusion: str | None = None  # None | early | feature | cross_attention
    epochs: int = 40
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 1e-2
    patience: int = 10
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _forward(model, xs):
    return model(*xs)


@torch.no_grad()
def predict_logits(model: nn.Module, xs, device, batch_size: int = 256) -> np.ndarray:
    """``xs`` is a tensor or a tuple of aligned tensors (fusion models)."""
    xs = xs if isinstance(xs, (tuple, list)) else (xs,)
    model.eval()
    out = []
    for i in range(0, len(xs[0]), batch_size):
        out.append(_forward(model, [x[i : i + batch_size].to(device) for x in xs]).float().cpu())
    return torch.cat(out).numpy()


def train_model(xs_train, y_train, xs_val, y_val, n_classes: int, cfg: TrainConfig, device,
                batch_transform=None):
    """Fit one model; returns (best model on CPU-agnostic device, history).

    * AdamW + cosine learning-rate schedule, cross-entropy loss.
    * Model selection: epoch with the best validation macro-F1 (ties: lower
      validation loss); early stopping after ``cfg.patience`` stale epochs.
      The validation set is made of training-side subjects only, so the test
      subjects never influence training, selection or calibration.
    * ``batch_transform(indices) -> tuple of tensors`` replaces the training
      inputs for a batch (used for signal-domain augmentation).
    """
    seed_everything(cfg.seed)
    xs_train = xs_train if isinstance(xs_train, (tuple, list)) else (xs_train,)
    xs_val = xs_val if isinstance(xs_val, (tuple, list)) else (xs_val,)
    if cfg.fusion:
        model = build_fusion(cfg.fusion, n_classes, cfg.model)
    else:
        kw = {}
        if cfg.model in ("temporal_cnn", "transformer"):
            kw["n_doppler"] = xs_train[0].shape[-2]
        if cfg.model == "transformer":
            kw["n_time"] = xs_train[0].shape[-1]
        model = build_model(cfg.model, n_classes, in_ch=xs_train[0].shape[1], **kw)
    model.to(device)

    xs_train = [x.to(device) for x in xs_train]
    y_train_t = torch.as_tensor(y_train, dtype=torch.long, device=device)
    y_val_t = torch.as_tensor(y_val, dtype=torch.long)
    n = len(y_train)
    steps = cfg.epochs * int(np.ceil(n / cfg.batch_size))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps, pct_start=0.15)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(cfg.seed)

    best = {"f1": -1.0, "loss": float("inf"), "epoch": -1, "state": None}
    history, stale, t0 = [], 0, time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        tot = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            if len(idx) < 2:  # BatchNorm needs more than one sample
                continue
            if batch_transform is not None:
                xb = [x.to(device) for x in batch_transform(idx.numpy())]
            else:
                xb = [x[idx.to(device)] for x in xs_train]
            loss = loss_fn(_forward(model, xb), y_train_t[idx.to(device)])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        logits = torch.from_numpy(predict_logits(model, xs_val, device))
        val_loss = float(loss_fn(logits, y_val_t))
        val_f1 = float(f1_score(y_val, logits.argmax(1).numpy(), average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_loss": tot / n, "val_loss": val_loss, "val_macro_f1": val_f1})
        if (val_f1, -val_loss) > (best["f1"], -best["loss"]):
            best = {"f1": val_f1, "loss": val_loss, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.load_state_dict(best["state"])
    info = {
        "best_epoch": best["epoch"],
        "best_val_macro_f1": best["f1"],
        "epochs_run": len(history),
        "train_seconds": time.perf_counter() - t0,
        "history": history,
    }
    return model, info
