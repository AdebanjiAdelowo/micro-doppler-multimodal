#!/usr/bin/env python
"""Stream the raw archive into the compact processed cache.

The 7z archive is decompressed in memory file by file and each recording is
reduced to MTI-filtered range bins (see mdradar.data.preprocess_recording),
so the tens of GB of text it contains never touch the disk.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
import py7zr
from py7zr.io import Py7zIO, WriterFactory

from mdradar import data, io

# Collection sessions; the archive's top-level folders start with these numbers.
SESSIONS = {
    1: "December 2017, University of Glasgow laboratory",
    2: "March 2017, University of Glasgow laboratory",
    3: "June 2017, University of Glasgow laboratory",
    4: "July 2018, University of Glasgow common room",
    5: "February 2019, University of Glasgow laboratory",
    6: "February 2019, North Glasgow Homes (3 rooms)",
    7: "March 2019, Age UK West Cumbria (2 rooms)",
}


def session_of(path: str) -> tuple[int, str]:
    folder = path.split("/")[0]
    m = re.match(r"\s*(\d)\b", folder)
    if "/" not in path or not m or int(m.group(1)) not in SESSIONS:
        raise ValueError(f"cannot assign a collection session to {path!r}")
    return int(m.group(1)), folder


def _work(name: str, raw: bytes, range_bins, n_slow):
    rec = io.parse_dat_bytes(raw)
    sig, power = data.preprocess_recording(rec, tuple(range_bins), n_slow)
    m = rec.meta
    head = {"carrier_hz": m.carrier_hz, "chirp_s": m.chirp_s, "samples_per_chirp": m.samples_per_chirp,
            "bandwidth_hz": m.bandwidth_hz, "n_chirps": rec.beat.shape[0], "duration_s": rec.duration_s}
    return name, sig, power, head


class _Sink(Py7zIO):
    """Collects one archive member in memory and hands it over when complete."""

    def __init__(self, name, size, on_done):
        self.name, self.expected, self.on_done = name, size, on_done
        self.buf = bytearray()

    def write(self, s):
        self.buf += s
        if len(self.buf) >= self.expected:
            raw, self.buf = bytes(self.buf), bytearray()
            self.on_done(self.name, raw)
        return len(s)

    def read(self, size=None):
        return b""

    def seek(self, offset, whence=0):
        return 0

    def flush(self):
        pass

    def size(self):
        return len(self.buf)


class _Null(_Sink):
    def write(self, s):
        return len(s)


class _Factory(WriterFactory):
    def __init__(self, sizes, on_done):
        self.sizes, self.on_done = sizes, on_done

    def create(self, filename):
        if filename in self.sizes:
            return _Sink(filename, self.sizes[filename], self.on_done)
        return _Null(filename, 0, None)


class _Stop(Exception):
    pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, default=Path("data/raw/Dataset_848.7z"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--range-bins", type=int, nargs=2, default=[5, 29], metavar=("LO", "HI"))
    ap.add_argument("--n-slow", type=int, default=5000, help="chirps kept per recording (5 s)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many recordings (inspection)")
    ap.add_argument("--power-only", action="store_true",
                    help="only write range_power.npy (per-range-bin power of every recording), "
                         "used to choose --range-bins before building the cache")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with py7zr.SevenZipFile(args.archive) as z:
        members = [f for f in z.list() if not f.is_directory and f.filename.lower().endswith(".dat")]
    skipped = []
    keep = []
    for f in members:
        try:
            io.parse_filename(f.filename)
            session_of(f.filename)
            keep.append(f)
        except ValueError as e:
            skipped.append({"file": f.filename, "reason": str(e)})
    if args.limit:
        keep = keep[: args.limit]
    names = [f.filename for f in keep]
    index = {n: i for i, n in enumerate(names)}
    sizes = {f.filename: f.uncompressed for f in keep}
    n = len(names)
    print(f"{n} recordings, {sum(sizes.values()) / 1e9:.1f} GB of text to stream; {len(skipped)} files skipped")

    lo, hi = args.range_bins
    sig_path = args.out / "range_bins.npy"
    signals = None if args.power_only else np.lib.format.open_memmap(
        sig_path, mode="w+", dtype=np.complex64, shape=(n, hi - lo, args.n_slow))
    powers = np.zeros((n, 64), dtype=np.float32)
    heads: dict[int, dict] = {}
    pending, lock, done_count = set(), threading.Lock(), [0]
    t0 = time.time()

    def collect(futs):
        for fut in futs:
            name, sig, power, head = fut.result()
            i = index[name]
            powers[i], heads[i] = power, head
            if signals is not None:
                signals[i] = sig
            done_count[0] += 1
            if done_count[0] % 100 == 0:
                print(f"  {done_count[0]}/{n}  {time.time() - t0:.0f} s", flush=True)

    with ProcessPoolExecutor(args.workers) as pool:
        def on_done(name, raw):
            with lock:
                pending.add(pool.submit(_work, name, raw, args.range_bins, args.n_slow))
                if len(pending) >= 2 * args.workers:  # back-pressure: bound memory
                    finished, rest = wait(pending, return_when=FIRST_COMPLETED)
                    pending.clear()
                    pending.update(rest)
                    collect(finished)
                if args.limit and done_count[0] + len(pending) >= n:
                    raise _Stop

        try:
            with py7zr.SevenZipFile(args.archive) as z:
                z.extract(targets=names, factory=_Factory(sizes, on_done))
        except _Stop:
            pass
        collect(wait(pending).done)
    if len(heads) != n:
        raise RuntimeError(f"processed {len(heads)} of {n} recordings")

    rows = []
    for name in names:
        i = index[name]
        lab = io.parse_filename(name)
        sid, folder = session_of(name)
        rows.append({
            "index": i, "file": name, "session": sid, "session_folder": folder,
            "subject": lab["subject"], "person": f"S{sid}P{lab['subject']:02d}", "group": lab["subject"],
            "label": lab["activity"] - 1, "activity": lab["activity_name"], "repetition": lab["repetition"],
            **heads[i], "zero_padded": heads[i]["n_chirps"] < args.n_slow,
        })
    man = pd.DataFrame(rows)
    np.save(args.out / "range_power.npy", powers)
    man.to_csv(args.out / ("manifest_power_pass.csv" if args.power_only else "manifest.csv"), index=False)
    if args.power_only:
        print(f"power pass done in {time.time() - t0:.0f} s")
        return
    signals.flush()
    first = heads[0]
    meta = {
        "source": "University of Glasgow, Radar signatures of human activities, DOI 10.5525/gla.researchdata.848",
        "licence": "CC BY 4.0",
        "range_bins": [lo, hi], "n_slow": args.n_slow, "range_window": "hann",
        "mti": {"type": "butterworth highpass", "order": 4, "cutoff_hz": 3.75, "causal": True},
        "prf_hz": 1.0 / first["chirp_s"], "wavelength_m": io.C / first["carrier_hz"],
        "range_resolution_m": io.C / (2 * first["bandwidth_hz"]),
        "n_recordings": n, "skipped_files": skipped,
        "sessions": SESSIONS,
    }
    (args.out / "cache_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done in {time.time() - t0:.0f} s -> {sig_path} ({sig_path.stat().st_size / 1e9:.2f} GB)")
    print(man.groupby(["session", "activity"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
