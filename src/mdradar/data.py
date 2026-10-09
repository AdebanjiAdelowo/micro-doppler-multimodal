"""Processed-data cache, manifest and leakage-safe splits.

Cache layout (``data/processed/`` by default, never committed):

* ``range_bins.npy``  complex64 memmap, shape (N, n_range_bins, n_slow):
  MTI-filtered range profiles restricted to the selected range bins and to a
  fixed slow-time length;
* ``manifest.csv``    one row per recording (labels, session, subject,
  acquisition header, original duration);
* ``cache_meta.json`` the exact preprocessing parameters.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import dsp
from .io import Recording


def preprocess_recording(
    rec: Recording,
    range_bins: tuple[int, int],
    n_slow: int,
    range_window: str = "hann",
    mti_cutoff_hz: float = 3.75,
    mti_order: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Raw beat signal -> (selected range bins x slow time, per-bin power).

    1. range compression (fast-time DFT);
    2. MTI high-pass along slow time on the *whole* recording;
    3. keep range bins [lo, hi) and the first ``n_slow`` chirps (zero-padded
       if the recording is shorter).

    The second output is the mean post-MTI power of every range bin over the
    kept time span, used to audit the range-bin selection.
    """
    prof = dsp.range_profiles(rec.beat, window=range_window)  # (n_chirps, n_range)
    prof = dsp.mti_highpass(prof, rec.meta.prf_hz, mti_cutoff_hz, mti_order, axis=0)
    prof = prof[:n_slow]
    bin_power = (np.abs(prof) ** 2).mean(axis=0)
    lo, hi = range_bins
    out = np.zeros((hi - lo, n_slow), dtype=np.complex64)
    out[:, : prof.shape[0]] = prof[:, lo:hi].T
    return out, bin_power.astype(np.float32)


class Cache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = pd.read_csv(self.root / "manifest.csv")
        self.meta = json.loads((self.root / "cache_meta.json").read_text())
        self.signals = np.load(self.root / "range_bins.npy", mmap_mode="r")
        if len(self.manifest) != self.signals.shape[0]:
            raise RuntimeError("manifest and signal cache disagree on the number of recordings")

    @property
    def prf_hz(self) -> float:
        return float(self.meta["prf_hz"])

    @property
    def wavelength_m(self) -> float:
        return float(self.meta["wavelength_m"])

    @property
    def labels(self) -> np.ndarray:
        return self.manifest["label"].to_numpy()

    @property
    def groups(self) -> np.ndarray:
        return self.manifest["group"].to_numpy()


# ------------------------------------------------------------------- splits
def assert_disjoint(manifest: pd.DataFrame, split: dict, key: str = "group") -> None:
    """Fail loudly if any group appears in more than one partition."""
    sets = {k: set(manifest[key].to_numpy()[v]) for k, v in split.items() if k in ("train", "val", "test")}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            common = sets[a] & sets[b]
            if common:
                raise AssertionError(f"{key} leakage between {a} and {b}: {sorted(common)[:5]}")
    idx = np.concatenate([split[k] for k in names])
    if len(np.unique(idx)) != len(idx):
        raise AssertionError("a recording appears in more than one partition")


def _carve_val(manifest: pd.DataFrame, idx: np.ndarray, seed: int, n_splits: int = 5):
    """Split ``idx`` into (train, val) with disjoint groups, ~1/n_splits in val."""
    y = manifest["label"].to_numpy()[idx]
    g = manifest["group"].to_numpy()[idx]
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr, va = next(sgkf.split(idx, y, g))
    return idx[tr], idx[va]


def subject_kfold(manifest: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> list[dict]:
    """Subject-independent stratified K-fold with a subject-disjoint validation
    set carved from each training side.

    The grouping key is ``group`` = the subject number printed in the file
    name. The dataset reuses numbers across collection sessions for different
    people, and may reuse them for the same person; grouping by the bare
    number is leak-free under both readings (see docs/AUDIT.md).
    """
    y, g = manifest["label"].to_numpy(), manifest["group"].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for k, (rest, test) in enumerate(sgkf.split(np.zeros(len(y)), y, g)):
        train, val = _carve_val(manifest, rest, seed + 1000 + k)
        split = {"name": f"fold{k}", "train": train, "val": val, "test": test}
        assert_disjoint(manifest, split)
        folds.append(split)
    return folds


def session_split(manifest: pd.DataFrame, test_sessions: list[int], seed: int = 0) -> dict:
    """Train on every other session, test on ``test_sessions``.

    Test recordings whose subject number also occurs on the training side are
    dropped, so the test set is unseen in both session and (numbered) subject.
    """
    sess = manifest["session"].to_numpy()
    g = manifest["group"].to_numpy()
    is_test = np.isin(sess, test_sessions)
    rest = np.flatnonzero(~is_test)
    test = np.flatnonzero(is_test & ~np.isin(g, np.unique(g[rest])))
    dropped = int(is_test.sum() - len(test))
    train, val = _carve_val(manifest, rest, seed + 2000)
    split = {
        "name": "test_sessions_" + "_".join(map(str, test_sessions)),
        "train": train, "val": val, "test": test, "dropped_test_recordings": dropped,
    }
    assert_disjoint(manifest, split)
    return split
