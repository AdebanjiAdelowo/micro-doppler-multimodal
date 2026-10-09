"""Classification, calibration and selective-prediction metrics."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def classification_metrics(y: np.ndarray, pred: np.ndarray, n_classes: int) -> dict:
    labels = np.arange(n_classes)
    return {
        "accuracy": float(np.mean(y == pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def per_class_report(y: np.ndarray, pred: np.ndarray, class_names: list[str]) -> dict:
    labels = np.arange(len(class_names))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    return {
        name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i, name in enumerate(class_names)
    }


def confusion(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    return confusion_matrix(y, pred, labels=np.arange(n_classes))


# ---------------------------------------------------------------- calibration
def expected_calibration_error(y: np.ndarray, prob: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width confidence bins:

        ECE = sum_b (|B_b| / N) | acc(B_b) - conf(B_b) |
    """
    conf = prob.max(axis=1)
    correct = (prob.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_bins(y: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> dict:
    conf = prob.max(axis=1)
    correct = (prob.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    out = {"edges": edges.tolist(), "count": [], "accuracy": [], "confidence": []}
    for b in range(n_bins):
        m = idx == b
        out["count"].append(int(m.sum()))
        out["accuracy"].append(float(correct[m].mean()) if m.any() else None)
        out["confidence"].append(float(conf[m].mean()) if m.any() else None)
    return out


def brier_score(y: np.ndarray, prob: np.ndarray) -> float:
    """Multi-class Brier score: mean_i sum_c (p_ic - 1[y_i = c])^2, in [0, 2]."""
    onehot = np.eye(prob.shape[1])[y]
    return float(np.mean(np.sum((prob - onehot) ** 2, axis=1)))


def nll(y: np.ndarray, prob: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, None))))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature T > 0 minimising the NLL of softmax(logits / T).

    Must be fitted on data disjoint from the test set (we use the validation
    subjects of each fold).
    """

    def loss(log_t: float) -> float:
        lp = log_softmax(logits / np.exp(log_t), axis=1)
        return -lp[np.arange(len(y)), y].mean()

    res = minimize_scalar(loss, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(res.x))


def apply_temperature(logits: np.ndarray, t: float) -> np.ndarray:
    return softmax(logits / t, axis=1)


def calibration_metrics(y: np.ndarray, prob: np.ndarray) -> dict:
    return {
        "ece": expected_calibration_error(y, prob),
        "brier": brier_score(y, prob),
        "nll": nll(y, prob),
    }


# ------------------------------------------------------- selective prediction
def risk_coverage(y: np.ndarray, prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Error rate among the k most confident predictions, for k = 1..N.

    Returns (coverage, risk). Confidence = maximum class probability.
    """
    conf = prob.max(axis=1)
    err = (prob.argmax(axis=1) != y).astype(float)
    order = np.argsort(-conf, kind="stable")
    n = len(y)
    k = np.arange(1, n + 1)
    return k / n, np.cumsum(err[order]) / k


def selective_metrics(y: np.ndarray, prob: np.ndarray, coverages=(0.8, 0.9)) -> dict:
    cov, risk = risk_coverage(y, prob)
    out = {"aurc": float(np.mean(risk))}
    for c in coverages:
        i = min(int(np.ceil(c * len(y))) - 1, len(y) - 1)
        out[f"acc_at_cov_{int(round(c * 100))}"] = float(1.0 - risk[i])
    return out


# ------------------------------------------------------------------ bootstrap
def bootstrap_ci(
    fn,
    y: np.ndarray,
    pred_or_prob: np.ndarray,
    groups: np.ndarray | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI of ``fn(y, pred_or_prob)``.

    With ``groups`` the resampling unit is the group (subject): recordings of
    one person are correlated, so resampling recordings independently would
    understate the uncertainty (cluster bootstrap).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    if groups is None:
        for _ in range(n_boot):
            i = rng.integers(0, n, n)
            stats.append(fn(y[i], pred_or_prob[i]))
    else:
        uniq = np.unique(groups)
        members = [np.flatnonzero(groups == g) for g in uniq]
        for _ in range(n_boot):
            pick = rng.integers(0, len(uniq), len(uniq))
            i = np.concatenate([members[j] for j in pick])
            stats.append(fn(y[i], pred_or_prob[i]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
