"""Minimal file-based experiment tracking.

Each run gets ``runs/<experiment>/<timestamp>/`` holding ``config.json``,
``env.json`` (git commit, library versions, device) and ``metrics.jsonl``.
The final summary is also written to ``results/<experiment>.json`` so the
numbers quoted in the documentation are traceable to a commit and a config.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def git_commit(cwd: str | Path = ".") -> str | None:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "configs"], cwd=cwd,
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


def environment(device) -> dict:
    import scipy
    import sklearn

    return {
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "device": str(device),
    }


class Run:
    def __init__(self, experiment: str, config: dict, device, root: str | Path = "runs",
                 results_dir: str | Path = "results"):
        self.experiment = experiment
        self.dir = Path(root) / experiment / time.strftime("%Y%m%d-%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = Path(results_dir)
        self.config = config
        self.env = environment(device)
        (self.dir / "config.json").write_text(json.dumps(config, indent=2, default=_jsonable))
        (self.dir / "env.json").write_text(json.dumps(self.env, indent=2))
        self._t0 = time.time()

    def log(self, **record) -> None:
        with open(self.dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record, default=_jsonable) + "\n")

    def finish(self, summary: dict, name: str | None = None) -> Path:
        out = {
            "experiment": self.experiment,
            "config": self.config,
            "env": self.env,
            "wall_seconds": round(time.time() - self._t0, 1),
            "results": summary,
        }
        text = json.dumps(out, indent=2, default=_jsonable)
        (self.dir / "summary.json").write_text(text)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        path = self.results_dir / f"{name or self.experiment}.json"
        path.write_text(text)
        return path
