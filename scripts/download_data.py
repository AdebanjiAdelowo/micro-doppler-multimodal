#!/usr/bin/env python
"""Download the University of Glasgow radar dataset (DOI 10.5525/gla.researchdata.848).

Licence: CC BY 4.0. Nothing is downloaded unless ``--yes`` is given; the
script first prints the disk space the full pipeline needs.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

URL = "https://researchdata.gla.ac.uk/848/20/Dataset_848.7z"
SIZE = 2_060_197_481  # bytes, Content-Length reported by the repository
SHA256 = "652f9adc2c9921f9fa08ba2de2d40e3f357d89a91f1b13ac790bc68f4f54ae61"
CACHE_ESTIMATE = 1.7e9  # processed cache written by prepare_data.py with default settings


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/raw/Dataset_848.7z"))
    ap.add_argument("--yes", action="store_true", help="actually download")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.out.parent).free
    need = SIZE + CACHE_ESTIMATE
    print(f"archive: {SIZE / 1e9:.2f} GB   processed cache: ~{CACHE_ESTIMATE / 1e9:.1f} GB   "
          f"needed: ~{need / 1e9:.1f} GB   free: {free / 1e9:.1f} GB")
    print("The archive is never extracted to disk (it holds tens of GB of text); "
          "prepare_data.py streams it.")
    if not args.yes:
        print("dry run; pass --yes to download")
        return 0
    have = args.out.stat().st_size if args.out.exists() else 0
    if free < need - have:
        print("not enough free disk space", file=sys.stderr)
        return 1
    while have < SIZE:  # resume with HTTP range requests
        req = urllib.request.Request(URL, headers={"Range": f"bytes={have}-", "User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(args.out, "ab") as f:
                shutil.copyfileobj(r, f, 1 << 20)
        except OSError as e:
            print(f"connection dropped ({e}); resuming")
        have = args.out.stat().st_size
        print(f"{have / 1e9:.2f} / {SIZE / 1e9:.2f} GB")
    digest = sha256(args.out)
    print("sha256", digest)
    if digest != SHA256:
        print("checksum differs from the one recorded when the results were produced", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
