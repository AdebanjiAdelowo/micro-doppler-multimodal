#!/usr/bin/env python
"""Render every figure in results/figures/ from results/*.json (and, when
present, the processed cache and the out-of-fold logits of the last baselines
run). Figures are derived artefacts: nothing here computes a new result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.special import softmax

from mdradar import metrics as M

# Fixed categorical order: colour follows the model, never its rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MODEL_ORDER = ["svm", "random_forest", "light_cnn", "resnet_small", "temporal_cnn", "transformer", "light_cnn+aug"]
COLOR = {m: SERIES[i] for i, m in enumerate(MODEL_ORDER)}
LABEL = {"svm": "features + SVM", "random_forest": "features + random forest", "light_cnn": "light CNN",
         "resnet_small": "ResNet-style", "temporal_cnn": "temporal CNN", "transformer": "transformer",
         "light_cnn+aug": "light CNN + signal augmentation"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 2, "lines.markersize": 6,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
})


def load(results: Path, name: str):
    p = results / f"{name}.json"
    return json.loads(p.read_text())["results"] if p.exists() else None


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    print("wrote", out / f"{name}.png")


def short(name: str) -> str:
    return name.replace("_", " ")


def fig_baselines(res, out):
    models = [m for m in MODEL_ORDER if m in res["models"]]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for i, m in enumerate(models):
        e = res["models"][m]
        mean, (lo, hi) = e["macro_f1"]["mean"], e["macro_f1_ci95"]
        ax.barh(i, mean, color=COLOR[m], height=0.55)
        ax.plot([lo, hi], [i, i], color=INK, lw=1.2)
        ax.text(hi + 0.004, i, f"{mean:.3f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(models)), [LABEL[m] for m in models])
    ax.invert_yaxis()
    ax.set_xlim(max(0.0, min(res["models"][m]["macro_f1_ci95"][0] for m in models) - 0.08), 1.0)
    ax.set_xlabel("macro-F1, subject-independent 5-fold (mean of seeds; line = 95 % subject-bootstrap CI)")
    ax.set_title("Feature-model and architecture comparison")
    ax.grid(axis="y", visible=False)
    save(fig, out, "baselines_macro_f1")

    names = [short(c) for c in res["class_names"]]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.4))
    for ax, m in zip(axes.ravel(), models, strict=False):
        cm = np.array(res["models"][m]["confusion_seed0"], dtype=float)
        rate = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(rate, cmap=SEQ, vmin=0, vmax=1)
        for i in range(len(names)):
            for j in range(len(names)):
                if cm[i, j]:
                    ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=8,
                            color="white" if rate[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.set_title(LABEL[m])
        ax.grid(False)
    for ax in axes.ravel()[len(models):]:
        ax.axis("off")
    fig.supxlabel("predicted", color=INK2, fontsize=10)
    fig.supylabel("true", color=INK2, fontsize=10)
    fig.tight_layout()
    save(fig, out, "baselines_confusion")


def fig_calibration(res, oof_path, out):
    if oof_path is None:
        return
    z = np.load(oof_path)
    y = z["y"]
    models = [m for m in MODEL_ORDER if m in z.files]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    for m in models:
        rb = M.reliability_bins(y, softmax(z[m][0], axis=1), 10)
        pts = [(c, a) for c, a, n in zip(rb["confidence"], rb["accuracy"], rb["count"], strict=True) if n >= 10]
        ax.plot(*zip(*pts, strict=True), marker="o", color=COLOR[m], label=LABEL[m])
    ax.set_xlabel("confidence (uncalibrated, bins with at least 10 recordings)")
    ax.set_ylabel("accuracy")
    ax.set_title("Reliability, seed 0")
    ax.set_xlim(0.3, 1.01)
    ax.set_ylim(0.3, 1.01)
    ax = axes[1]
    for m in models:
        cov, risk = M.risk_coverage(y, softmax(z[m][0], axis=1))
        keep = cov >= 0.3
        ax.plot(cov[keep], risk[keep], color=COLOR[m], label=LABEL[m])
    ax.set_xlabel("coverage (fraction of recordings answered)")
    ax.set_ylabel("error rate among answered")
    ax.set_title("Selective prediction, seed 0")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, out, "baselines_calibration")


def fig_stft(res, out):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for ax, study, xlabel in ((axes[0], "window", "window length (ms)"), (axes[1], "overlap", "overlap (%)")):
        rows = [r for r in res["rows"] if r["study"] == study]
        x = [r["n_window"] if study == "window" else 100 * r["overlap"] for r in rows]
        for m in rows[0]["models"]:
            mean = np.array([r["models"][m]["macro_f1"]["mean"] for r in rows])
            ci = np.array([r["models"][m]["macro_f1_ci95_last_seed"] for r in rows])
            ax.fill_between(x, ci[:, 0], ci[:, 1], color=COLOR[m], alpha=0.15, lw=0)
            ax.plot(x, mean, marker="o", color=COLOR[m], label=LABEL[m])
        if study == "window":
            ax.set_xscale("log", base=2)
            ax.set_xticks(x, [str(v) for v in x])
            sec = ax.secondary_xaxis("top")
            sec.set_xticks(x, [f"{r['freq_res_hz']:.1f}" for r in rows])
            sec.set_xlabel("Doppler resolution (Hz)", color=INK2, fontsize=9)
            sec.spines["top"].set_visible(False)
        ax.set_xlabel(xlabel)
        ax.set_title("STFT window length (frame period 40 ms)" if study == "window"
                     else "STFT overlap (window 256 ms)", pad=28 if study == "window" else 6)
    axes[0].set_ylabel("macro-F1 (band = 95 % subject-bootstrap CI)")
    axes[1].legend()
    fig.tight_layout()
    save(fig, out, "stft_ablation")


def fig_snr(res, out):
    snrs = res["snr_db"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for m in [m for m in MODEL_ORDER if m in res["models"]]:
        curve = res["models"][m]
        ys = [curve[str(s)]["macro_f1"]["mean"] for s in snrs]
        ax.plot(snrs, ys, marker="o", color=COLOR[m], label=LABEL[m], ls="--" if m.endswith("+aug") else "-")
        ax.plot([max(snrs) + 6], [curve["clean"]["macro_f1"]["mean"]], marker="s", color=COLOR[m])
    ax.axhline(1 / 6, color=MUTED, lw=1, ls=":")
    ax.text(min(snrs), 1 / 6 + 0.015, "chance (6 classes)", color=MUTED, fontsize=8)
    ax.set_xticks([*snrs, max(snrs) + 6], [*map(str, snrs), "no added\nnoise"])
    ax.invert_xaxis()
    ax.set_xlabel("measured-signal to added-noise ratio (dB)")
    ax.set_ylabel("macro-F1 on held-out subjects")
    ax.set_title("Robustness to additive white noise injected before the STFT")
    ax.legend(fontsize=8)
    save(fig, out, "snr_sweep")


def fig_cross_session(res, base, out):
    splits = list(res["splits"])
    models = [m for m in MODEL_ORDER if m in res["splits"][splits[0]]["models"]]
    labels = [s.replace("unseen_", "").replace("_", " ") for s in splits]
    if base:
        splits, labels = ["__cv__", *splits], ["subject CV\n(all sessions)", *labels]
    fig, ax = plt.subplots(figsize=(10, 3.9))
    w = 0.8 / len(models)
    for k, m in enumerate(models):
        vals = [base["models"][m]["macro_f1"]["mean"] if s == "__cv__" else res["splits"][s]["models"][m]["macro_f1"]["mean"]
                for s in splits]
        ax.bar(np.arange(len(splits)) + (k - (len(models) - 1) / 2) * w, vals, w * 0.9, color=COLOR[m], label=LABEL[m])
    ax.set_xticks(range(len(splits)), labels)
    ax.set_ylabel("macro-F1 on the unseen session(s)")
    ax.set_ylim(0, 1)
    ax.set_title("Generalisation to unseen collection sessions")
    ax.grid(axis="x", visible=False)
    ax.legend(ncols=len(models), fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.36))
    save(fig, out, "cross_session")


def fig_multiview(res, out):
    v = res["variants"]
    names = list(v)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for i, n in enumerate(names):
        lo, hi = v[n]["macro_f1_ci95_seed0"]
        ax.barh(i, v[n]["macro_f1"]["mean"], color=SERIES[0], height=0.55)
        ax.plot([lo, hi], [i, i], color=INK, lw=1.2)
        ax.text(hi + 0.004, i, f"{v[n]['macro_f1']['mean']:.3f}", va="center", fontsize=9)
    ax.set_yticks(range(len(names)), [short(n) for n in names])
    ax.invert_yaxis()
    ax.set_xlim(max(0, min(v[n]["macro_f1_ci95_seed0"][0] for n in names) - 0.08), 1.0)
    ax.set_xlabel("macro-F1 (line = 95 % subject-bootstrap CI, seed 0)")
    ax.set_title("Fusion of two radar views of the same recording (not radar + vision)")
    ax.grid(axis="y", visible=False)
    save(fig, out, "multiview_fusion")


def fig_latency(res, base, out):
    if not base:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for m in [m for m in MODEL_ORDER if m in res["models"] and m in base["models"]]:
        x = res["models"][m]["latency_cpu"]["median_ms"]
        yv = base["models"][m]["macro_f1"]["mean"]
        ax.scatter([x], [yv], s=70, color=COLOR[m], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(LABEL[m], (x, yv), xytext=(7, 5), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("classifier latency on CPU, batch 1, median (ms, log scale)")
    ax.set_ylabel("macro-F1 (subject CV)")
    ax.set_title("Accuracy against measured classifier latency")
    save(fig, out, "latency_tradeoff")


def fig_data(cache_dir: Path, out: Path):
    if not (cache_dir / "manifest.csv").exists():
        return
    from mdradar import dsp
    from mdradar.data import Cache
    from mdradar.experiments import power_specs
    from mdradar.io import CLASS_NAMES

    cache = Cache(cache_dir)
    spec = dsp.SpecConfig()
    man = cache.manifest
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharex=True, sharey=True)
    v = spec.fd_max_hz * cache.wavelength_m / 2
    for ax, k in zip(axes.ravel(), range(len(CLASS_NAMES)), strict=True):
        idx = man.index[(man.label == k) & (man.session == 1)][0]
        s, _ = power_specs(cache, spec, np.array([idx]))
        im = ax.imshow(dsp.to_db_image(s[0], spec.dynamic_range_db) * spec.dynamic_range_db - spec.dynamic_range_db,
                       origin="lower", aspect="auto", cmap=SEQ, extent=[0, 5, -v, v])
        ax.set_title(f"{short(CLASS_NAMES[k])}  ({Path(man.file[idx]).name})", fontsize=9)
        ax.grid(False)
    fig.supxlabel("time (s)", color=INK2, fontsize=10)
    fig.supylabel("radial velocity (m/s), sign per receiver convention", color=INK2, fontsize=10)
    fig.colorbar(im, ax=axes, label="dB relative to peak", shrink=0.8)
    save(fig, out, "example_spectrograms")

    power = np.load(cache_dir / "range_power.npy")
    lo, hi = cache.meta["range_bins"]
    r = np.arange(power.shape[1]) * cache.meta["range_resolution_m"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for sid in sorted(man.session.unique()):
        p = power[(man.session == sid).to_numpy()].mean(0)
        ax.plot(r, 10 * np.log10(p / p.max()), color=SERIES[sid - 1], label=f"session {sid}", lw=1.6)
    ax.axvspan(r[lo], r[hi - 1], color=GRID, alpha=0.6, lw=0)
    ax.text(r[lo] + 0.1, -38, "range bins kept", color=INK2, fontsize=8)
    ax.set_xlabel("range (m)")
    ax.set_ylabel("mean post-MTI power (dB re. session peak)")
    ax.set_ylim(-40, 1)
    ax.set_title("Where the moving-target energy is, per collection session")
    ax.legend(ncols=2, fontsize=8)
    save(fig, out, "range_power_by_session")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    args = ap.parse_args()
    out = args.results / "figures"
    base = load(args.results, "baselines")
    if base:
        fig_baselines(base, out)
        oofs = sorted((args.runs / "baselines").glob("*/oof_logits.npz"))
        fig_calibration(base, oofs[-1] if oofs else None, out)
    for name, fn in (("stft_ablation", fig_stft), ("snr_sweep", fig_snr), ("multiview_fusion", fig_multiview)):
        if (res := load(args.results, name)):
            fn(res, out)
    if (res := load(args.results, "cross_session")):
        fig_cross_session(res, base, out)
    if (res := load(args.results, "latency")):
        fig_latency(res, base, out)
    fig_data(args.data, out)


if __name__ == "__main__":
    main()
