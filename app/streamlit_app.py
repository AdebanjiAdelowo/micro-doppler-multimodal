"""Lightweight demo: classify one recording and browse the measured results.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from mdradar.inference import INTENDED_USE, Predictor

ROOT = Path(__file__).resolve().parents[1]
CKPT = Path(os.environ.get("MDRADAR_CHECKPOINT", ROOT / "checkpoints" / "light_cnn.pt"))

st.set_page_config(page_title="Micro-Doppler activity classification", layout="wide")
st.title("Micro-Doppler radar activity classification")
st.caption(INTENDED_USE)


@st.cache_resource
def load_predictor():
    return Predictor.load(CKPT) if CKPT.exists() else None


tab_demo, tab_results = st.tabs(["Classify a recording", "Measured results"])

with tab_demo:
    pred = load_predictor()
    if pred is None:
        st.warning(f"No checkpoint at {CKPT}. Train one with `python scripts/train.py`.")
    else:
        threshold = st.slider("Abstain when confidence is below", 0.0, 1.0, 0.6, 0.05)
        up = st.file_uploader("Raw .dat recording (Glasgow format) or .npy of complex range bins", type=["dat", "npy"])
        if up is not None:
            try:
                p = pred.predict_file(up.name, up.getvalue(), threshold)
            except ValueError as e:
                st.error(str(e))
            else:
                left, right = st.columns([3, 2])
                with left:
                    fd = pred.spec.fd_max_hz
                    v = fd * pred.meta["wavelength_m"] / 2
                    fig, ax = plt.subplots(figsize=(7, 3.6))
                    im = ax.imshow(p.image * pred.spec.dynamic_range_db - pred.spec.dynamic_range_db, origin="lower",
                                   aspect="auto", cmap="viridis",
                                   extent=[0, pred.meta["n_slow"] / pred.meta["prf_hz"], -v, v])
                    ax.set_xlabel("time (s)")
                    ax.set_ylabel("radial velocity (m/s), sign per receiver convention")
                    fig.colorbar(im, label="dB re. peak")
                    st.pyplot(fig)
                with right:
                    if p.abstained:
                        st.subheader("No decision")
                        st.write(f"Most likely `{p.label}` at confidence {p.confidence:.2f}, below the threshold.")
                    else:
                        st.subheader(p.label.replace("_", " "))
                        st.write(f"Calibrated confidence {p.confidence:.2f}")
                    st.bar_chart(pd.Series(p.probabilities, name="probability"))
                    st.json({k: round(v, 2) for k, v in p.timings_ms.items()})

with tab_results:
    files = sorted((ROOT / "results").glob("*.json"))
    if not files:
        st.info("No results yet. Run the experiments in `configs/`.")
    else:
        choice = st.selectbox("Experiment", [f.stem for f in files])
        doc = json.loads((ROOT / "results" / f"{choice}.json").read_text())
        st.write({k: doc["env"][k] for k in ("git_commit", "device", "torch")})
        models = doc["results"].get("models")
        if isinstance(models, dict) and all("macro_f1" in m for m in models.values()):
            st.dataframe(pd.DataFrame({
                name: {k: m[k]["mean"] for k in ("accuracy", "macro_f1", "balanced_accuracy") if k in m}
                for name, m in models.items()}).T.style.format("{:.3f}"))
        figs = sorted((ROOT / "results" / "figures").glob(f"{choice}*.png"))
        for f in figs:
            st.image(str(f))
        with st.expander("Raw JSON"):
            st.json(doc["results"])

