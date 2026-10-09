#!/usr/bin/env python
"""Kaggle / Colab GPU launcher.

Paste into a notebook cell (GPU accelerator and internet enabled), or run as a
script kernel. It installs the package from a source you provide, downloads and
prepares the dataset on the kernel's disk, runs the ``configs/gpu_*.yaml``
studies and packs ``results_gpu/`` into ``/kaggle/working/results_gpu.zip``.

Source of the code, in order of precedence:
  1. environment variable MDRADAR_REPO_URL (a git URL you control);
  2. a Kaggle dataset containing this repository, attached at
     /kaggle/input/micro-doppler-multimodal.

Disk: 2.1 GB archive + 1.7 GB cache (Kaggle offers about 20 GB in
/kaggle/working and more in /kaggle/tmp). The archive is CC BY 4.0.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path(os.environ.get("MDRADAR_WORKDIR", "/kaggle/working" if Path("/kaggle").exists() else "/content"))
REPO = WORK / "micro-doppler-multimodal"
CONFIGS = os.environ.get("MDRADAR_CONFIGS", "gpu_baselines gpu_cross_session gpu_snr_sweep").split()


def sh(*cmd: str, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        print("WARNING: no CUDA device visible; the studies will run but slowly.", file=sys.stderr)
    else:
        print("GPU:", torch.cuda.get_device_name(0))

    if not REPO.exists():
        url = os.environ.get("MDRADAR_REPO_URL")
        attached = Path("/kaggle/input/micro-doppler-multimodal")
        if url:
            sh("git", "clone", "--depth", "1", url, str(REPO))
        elif attached.exists():
            shutil.copytree(attached, REPO)
        else:
            raise SystemExit("set MDRADAR_REPO_URL or attach the repository as a Kaggle dataset")
    # torch is preinstalled on Kaggle/Colab GPU images; do not reinstall it.
    sh(sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", ".", cwd=REPO)
    sh(sys.executable, "-m", "pip", "install", "-q", "py7zr", "pyyaml", cwd=REPO)

    free_gb = shutil.disk_usage(WORK).free / 1e9
    print(f"free disk: {free_gb:.1f} GB")
    if free_gb < 5:
        raise SystemExit("need about 4 GB of free disk for the archive and the cache")
    if not (REPO / "data/processed/manifest.csv").exists():
        sh(sys.executable, "scripts/download_data.py", "--yes", cwd=REPO)
        sh(sys.executable, "scripts/prepare_data.py", "--workers", str(os.cpu_count() or 2), cwd=REPO)
        (REPO / "data/raw/Dataset_848.7z").unlink()  # free 2 GB once the cache exists

    for name in CONFIGS:
        sh(sys.executable, "scripts/run_experiment.py", f"configs/{name}.yaml", "--results-dir", "results_gpu", cwd=REPO)
    shutil.make_archive(str(WORK / "results_gpu"), "zip", REPO / "results_gpu")
    print("done:", WORK / "results_gpu.zip")


if __name__ == "__main__":
    main()
