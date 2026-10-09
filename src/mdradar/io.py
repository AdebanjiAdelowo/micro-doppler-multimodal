"""Loading of raw FMCW recordings with metadata preserved.

File format (University of Glasgow dataset 848, see docs/DATASETS.md): a text
file holding one long 1-D complex array. The first four entries are a header

    [carrier frequency (Hz), chirp duration (ms), samples per chirp, bandwidth (Hz)]

and the remaining entries are the complex (I + jQ) beat-note samples of all
chirps, concatenated chirp after chirp.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

C = 299_792_458.0  # speed of light, m/s

ACTIVITIES = {
    1: "walking",
    2: "sitting_down",
    3: "standing_up",
    4: "picking_up_object",
    5: "drinking",
    6: "falling",
}
CLASS_NAMES = [ACTIVITIES[k] for k in sorted(ACTIVITIES)]

_NAME_RE = re.compile(r"^(\d)P(\d+)A(\d+)R(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class RecordingMeta:
    """Acquisition parameters read from the file header."""

    carrier_hz: float
    chirp_s: float
    samples_per_chirp: int
    bandwidth_hz: float

    @property
    def prf_hz(self) -> float:
        """Pulse (chirp) repetition frequency = slow-time sampling rate."""
        return 1.0 / self.chirp_s

    @property
    def wavelength_m(self) -> float:
        return C / self.carrier_hz

    @property
    def range_resolution_m(self) -> float:
        return C / (2.0 * self.bandwidth_hz)

    @property
    def max_unambiguous_velocity_mps(self) -> float:
        """|v| at the slow-time Nyquist frequency PRF/2."""
        return self.wavelength_m * self.prf_hz / 4.0


@dataclass(frozen=True)
class Recording:
    """Raw beat signal, shape (n_chirps, samples_per_chirp), complex64."""

    beat: np.ndarray
    meta: RecordingMeta

    @property
    def duration_s(self) -> float:
        return self.beat.shape[0] * self.meta.chirp_s


def parse_filename(name: str) -> dict:
    """Extract labels from a ``KPXXAYYRZ.dat`` file name.

    K is the activity code (1-6), XX the subject ID *within its collection
    session* (IDs are reused across sessions for different people), YY the
    activity index and Z the repetition.
    """
    m = _NAME_RE.match(Path(name).name)
    if not m:
        raise ValueError(f"unrecognised recording name: {name!r}")
    k, subject, act, rep = (int(g) for g in m.groups())
    if k not in ACTIVITIES:
        raise ValueError(f"unknown activity code {k} in {name!r}")
    return {
        "activity": k,
        "activity_name": ACTIVITIES[k],
        "subject": subject,
        "activity_index": act,
        "repetition": rep,
    }


def _parse_complex_text(raw: bytes) -> np.ndarray:
    # MATLAB writes the imaginary unit as "i"; NumPy expects "j".
    tokens = raw.replace(b"i", b"j").split()
    return np.array(tokens, dtype=np.complex128)


def parse_dat_bytes(raw: bytes) -> Recording:
    """Parse the content of one ``.dat`` file."""
    values = _parse_complex_text(raw)
    if values.size < 5:
        raise ValueError("file too short to contain a header and data")
    header = values[:4].real
    n = int(round(header[2]))
    meta = RecordingMeta(
        carrier_hz=float(header[0]),
        chirp_s=float(header[1]) * 1e-3,
        samples_per_chirp=n,
        bandwidth_hz=float(header[3]),
    )
    data = values[4:]
    n_chirps = data.size // n
    if n_chirps == 0:
        raise ValueError("file holds less than one chirp")
    # A trailing incomplete chirp (if any) is dropped, never padded.
    beat = data[: n_chirps * n].reshape(n_chirps, n).astype(np.complex64)
    return Recording(beat=beat, meta=meta)


def load_dat(path: str | Path) -> Recording:
    return parse_dat_bytes(Path(path).read_bytes())
