#!/usr/bin/env python
"""Run one experiment from a YAML config: python scripts/run_experiment.py configs/baselines.yaml"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from mdradar.data import Cache
from mdradar.experiments import EXPERIMENTS
from mdradar.tracking import Run
from mdradar.train import pick_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--name", default=None, help="results file stem (default: config stem)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override top-level config keys (YAML values)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    for item in args.set:
        k, v = item.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    device = pick_device(args.device)
    torch.set_float32_matmul_precision("high")
    cache = Cache(args.data)
    run = Run(cfg["experiment"], {**cfg, "cache_meta": cache.meta, "n_recordings": len(cache.manifest)},
              device, results_dir=args.results_dir)
    print(f"experiment={cfg['experiment']} device={device} recordings={len(cache.manifest)} run={run.dir}")
    summary = EXPERIMENTS[cfg["experiment"]](cache, cfg, device, run)
    print("results written to", run.finish(summary, args.name or args.config.stem))


if __name__ == "__main__":
    main()
