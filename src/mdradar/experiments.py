"""Experiment drivers. Each function returns a JSON-serialisable summary and
is launched through ``scripts/run_experiment.py`` with a YAML config.

Shared protocol
---------------
* one prediction per recording from subject-independent K-fold (out-of-fold);
* the validation subjects of each fold (disjoint from train and test) are the
  only data used for early stopping, model selection and temperature fitting;
* every model sees exactly the same folds;
* confidence intervals: percentile cluster bootstrap over subjects.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from scipy.special import log_softmax, softmax
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import augment, dsp, features
from . import metrics as M
from .data import Cache, session_split, subject_kfold
from .io import CLASS_NAMES
from .models import count_parameters, late_fusion
from .train import TrainConfig, predict_logits, train_model

N_CLASSES = len(CLASS_NAMES)
CLASSICAL = ("svm", "random_forest")


# ------------------------------------------------------------ representations
def _batches(cache: Cache, idx, batch: int = 128):
    for i in range(0, len(idx), batch):
        j = np.sort(idx[i : i + batch])  # memmap fancy indexing wants sorted indices
        order = np.argsort(np.argsort(idx[i : i + batch]))
        yield torch.from_numpy(np.ascontiguousarray(cache.signals[j]))[order]


def _noisy(x: torch.Tensor, snr_db, gen):
    return x if snr_db is None else augment.add_awgn(x, snr_db, generator=gen)


def spectrogram_images(cache: Cache, cfg: dsp.SpecConfig, idx=None, snr_db=None, seed: int = 0) -> torch.Tensor:
    """(N, 1, H, W) model inputs; optional additive noise at ``snr_db``."""
    idx = np.arange(len(cache.manifest)) if idx is None else np.asarray(idx)
    gen = torch.Generator().manual_seed(seed)
    return torch.cat([dsp.model_input(_noisy(x, snr_db, gen), cfg, cache.prf_hz) for x in _batches(cache, idx)])


def power_specs(cache: Cache, cfg: dsp.SpecConfig, idx=None, snr_db=None, seed: int = 0):
    """(N, F, T) linear power spectrograms and the Doppler axis."""
    idx = np.arange(len(cache.manifest)) if idx is None else np.asarray(idx)
    gen = torch.Generator().manual_seed(seed)
    out, f = [], None
    for x in _batches(cache, idx):
        s, f = dsp.power_spectrogram(_noisy(x, snr_db, gen), cfg, cache.prf_hz)
        out.append(s.numpy())
    return np.concatenate(out), f


def feature_table(cache: Cache, cfg: dsp.SpecConfig, idx=None, snr_db=None, seed: int = 0):
    s, f = power_specs(cache, cfg, idx, snr_db, seed)
    return features.feature_matrix(s, f, cache.prf_hz / cfg.hop)


def range_time_images(cache: Cache, idx=None, out_size=(128, 128), dynamic_range_db: float = 45.0) -> torch.Tensor:
    """Second radar view of the same recordings: range-time power map.

    |x[r, n]|^2 of the MTI-filtered range profiles, averaged over slow time in
    blocks, in dB relative to the per-recording peak, resampled to out_size.
    """
    idx = np.arange(len(cache.manifest)) if idx is None else np.asarray(idx)
    out = []
    for x in _batches(cache, idx):
        p = (x.real**2 + x.imag**2).unsqueeze(1)
        p = torch.nn.functional.adaptive_avg_pool2d(p, (p.shape[-2], out_size[1]))
        img = dsp.to_db_image(p.squeeze(1), dynamic_range_db).unsqueeze(1).float()
        out.append(torch.nn.functional.interpolate(img, size=out_size, mode="bilinear", align_corners=False))
    return torch.cat(out)


# ------------------------------------------------------------------- models
def _fit_classical(name: str, x_tr, y_tr, x_va, y_va, seed: int):
    """Returns (model, info). SVM C is chosen on the validation subjects."""
    if name == "random_forest":
        m = RandomForestClassifier(n_estimators=500, min_samples_leaf=1, n_jobs=-1, random_state=seed)
        return m.fit(x_tr, y_tr), {}
    best = None
    for c in (1.0, 10.0, 100.0):
        m = make_pipeline(StandardScaler(), SVC(C=c, kernel="rbf", gamma="scale", probability=True, random_state=seed))
        m.fit(x_tr, y_tr)
        score = M.classification_metrics(y_va, m.predict(x_va), N_CLASSES)["macro_f1"]
        if best is None or score > best[0]:
            best = (score, c, m)
    return best[2], {"svm_C": best[1]}


def _classical_logits(model, x, classes_seen) -> np.ndarray:
    """Log-probabilities over all classes (classes unseen in training get ~0)."""
    p = np.full((len(x), N_CLASSES), 1e-6)
    p[:, classes_seen] = model.predict_proba(x)
    return np.log(p / p.sum(axis=1, keepdims=True))


def evaluate(y, logits, y_val=None, logits_val=None, groups=None, n_boot: int = 0) -> dict:
    """All reported metrics for one set of predictions."""
    prob = softmax(logits, axis=1)
    pred = prob.argmax(1)
    out = M.classification_metrics(y, pred, N_CLASSES)
    out["calibration_raw"] = M.calibration_metrics(y, prob)
    out["selective_raw"] = M.selective_metrics(y, prob)
    if logits_val is not None:
        t = M.fit_temperature(logits_val, y_val)
        prob_t = M.apply_temperature(logits, t)
        out["temperature"] = t
        out["calibration_ts"] = M.calibration_metrics(y, prob_t)
        out["selective_ts"] = M.selective_metrics(y, prob_t)
    if n_boot and groups is not None:
        for key in ("accuracy", "macro_f1", "balanced_accuracy"):
            fn = lambda a, b, key=key: M.classification_metrics(a, b, N_CLASSES)[key]  # noqa: E731
            out[f"{key}_ci95"] = M.bootstrap_ci(fn, y, pred, groups, n_boot)
    return out


def _mean_std(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}


def run_cv(cache: Cache, model: str, folds: list[dict], seed: int, *, images=None, feats=None,
           train_cfg: dict | None = None, device=None, fusion: str | None = None, images_b=None,
           test_inputs_fn=None, batch_transform_fn=None):
    """Train ``model`` on every fold; returns out-of-fold logits and extras.

    ``test_inputs_fn(fold, model_obj, classes)`` lets callers evaluate each
    fold's trained model on extra inputs (noise sweep) without retraining.
    """
    y = cache.labels
    n = len(y)
    oof = np.full((n, N_CLASSES), np.nan, dtype=np.float64)
    val_logits, infos, extras = {}, [], []
    for fold in folds:
        tr, va, te = fold["train"], fold["val"], fold["test"]
        if model in CLASSICAL:
            m, info = _fit_classical(model, feats[tr], y[tr], feats[va], y[va], seed)
            classes = m.classes_ if model == "random_forest" else m[-1].classes_
            predict = lambda x, m=m, classes=classes: _classical_logits(m, x, classes)  # noqa: E731
            oof[te], val_logits[fold["name"]] = predict(feats[te]), predict(feats[va])
        else:
            cfg = TrainConfig(model=model, fusion=fusion, seed=seed, **(train_cfg or {}))
            pack = (lambda i: (images[i], images_b[i])) if images_b is not None else (lambda i: images[i])
            bt = batch_transform_fn(tr) if batch_transform_fn else None
            m, info = train_model(pack(tr), y[tr], pack(va), y[va], N_CLASSES, cfg, device, batch_transform=bt)
            info.pop("history")
            predict = lambda x, m=m: predict_logits(m, x, device)  # noqa: E731
            oof[te], val_logits[fold["name"]] = predict(pack(te)), predict(pack(va))
        infos.append(info)
        if test_inputs_fn is not None:
            extras.append(test_inputs_fn(fold, predict))
    return {"oof": oof, "val": val_logits, "info": infos, "extras": extras}


def _pooled_temperature_eval(cache: Cache, folds, res, n_boot: int = 0) -> dict:
    """Evaluate OOF predictions; temperature fitted per fold on its val split."""
    y, g = cache.labels, cache.groups
    logits = res["oof"]
    scaled = np.empty_like(logits)
    temps = []
    for fold in folds:
        t = M.fit_temperature(res["val"][fold["name"]], y[fold["val"]])
        temps.append(t)
        scaled[fold["test"]] = logits[fold["test"]] / t
    out = evaluate(y, logits, groups=g, n_boot=n_boot)
    prob_t = softmax(scaled, axis=1)
    out["temperature_per_fold"] = temps
    out["calibration_ts"] = M.calibration_metrics(y, prob_t)
    out["selective_ts"] = M.selective_metrics(y, prob_t)
    return out


# ------------------------------------------------------------- experiments
def baselines(cache: Cache, cfg: dict, device, run) -> dict:
    """RQ1/RQ5: feature-model and architecture comparison with calibration."""
    spec = dsp.SpecConfig(**cfg.get("spec", {}))
    folds = subject_kfold(cache.manifest, cfg["n_splits"], cfg["split_seed"])
    y, g = cache.labels, cache.groups
    images = spectrogram_images(cache, spec)
    feats, names = feature_table(cache, spec)
    summary = {"class_names": CLASS_NAMES, "n_recordings": len(y), "feature_names": names, "models": {}}
    all_logits = {}
    for model in cfg["models"]:
        per_seed, logit_list = [], []
        for seed in cfg["seeds"]:
            t0 = time.perf_counter()
            res = run_cv(cache, model, folds, seed, images=images, feats=feats,
                         train_cfg=cfg.get("train"), device=device)
            ev = _pooled_temperature_eval(cache, folds, res)
            ev["train_seconds_total"] = time.perf_counter() - t0
            ev["fold_info"] = res["info"]
            per_seed.append(ev)
            logit_list.append(res["oof"])
            run.log(experiment="baselines", model=model, seed=seed,
                    **{k: ev[k] for k in ("accuracy", "macro_f1", "balanced_accuracy")})
            print(f"[baselines] {model} seed={seed} acc={ev['accuracy']:.4f} f1={ev['macro_f1']:.4f}", flush=True)
        all_logits[model] = np.stack(logit_list)
        preds = all_logits[model].argmax(-1)
        entry = {k: _mean_std([e[k] for e in per_seed]) for k in ("accuracy", "macro_f1", "balanced_accuracy")}
        for cal in ("calibration_raw", "calibration_ts", "selective_raw", "selective_ts"):
            entry[cal] = {k: _mean_std([e[cal][k] for e in per_seed]) for k in per_seed[0][cal]}
        entry["temperature"] = _mean_std([np.mean(e["temperature_per_fold"]) for e in per_seed])
        # CI of the seed-averaged metric under subject resampling
        for key in ("accuracy", "macro_f1", "balanced_accuracy"):
            fn = lambda a, b, key=key: float(np.mean(  # noqa: E731
                [M.classification_metrics(a, p, N_CLASSES)[key] for p in b.T]))
            entry[f"{key}_ci95"] = M.bootstrap_ci(fn, y, preds.T, g, cfg["n_boot"])
        entry["confusion_seed0"] = M.confusion(y, preds[0], N_CLASSES)
        entry["per_class_seed0"] = M.per_class_report(y, preds[0], CLASS_NAMES)
        entry["train_seconds_per_seed"] = _mean_std([e["train_seconds_total"] for e in per_seed])
        if len(cfg["seeds"]) > 1:  # deep ensemble = mean of the seeds' probabilities
            ens = np.log(softmax(all_logits[model], axis=-1).mean(0))
            entry["seed_ensemble"] = evaluate(y, ens)
        summary["models"][model] = entry

    # Paired comparison against the reference model on identical recordings.
    ref = cfg.get("reference", cfg["models"][0])
    summary["paired_macro_f1_difference_vs_" + ref] = {}
    for model, lg in all_logits.items():
        if model == ref:
            continue
        both = np.concatenate([lg.argmax(-1), all_logits[ref].argmax(-1)]).T  # (N, 2 * seeds)
        k = lg.shape[0]
        fn = lambda a, b, k=k: float(  # noqa: E731
            np.mean([M.classification_metrics(a, p, N_CLASSES)["macro_f1"] for p in b.T[:k]])
            - np.mean([M.classification_metrics(a, p, N_CLASSES)["macro_f1"] for p in b.T[k:]]))
        summary["paired_macro_f1_difference_vs_" + ref][model] = {
            "difference": fn(y, both), "ci95": M.bootstrap_ci(fn, y, both, g, cfg["n_boot"])}
    np.savez_compressed(run.dir / "oof_logits.npz", y=y, groups=g, **all_logits)
    return summary


def stft_ablation(cache: Cache, cfg: dict, device, run) -> dict:
    """RQ2: window-size and overlap ablations on identical folds."""
    folds = subject_kfold(cache.manifest, cfg["n_splits"], cfg["split_seed"])
    base = cfg.get("spec", {})
    rows = []
    for study, settings in cfg["studies"].items():
        for s in settings:
            spec = dsp.SpecConfig(**{**base, **s})
            res_info = dsp.resolution(spec.n_window, spec.hop, cache.prf_hz, spec.window)
            images = spectrogram_images(cache, spec)
            feats, _ = feature_table(cache, spec)
            n_frames = 1 + (cache.signals.shape[-1] - spec.n_window) // spec.hop
            row = {"study": study, **s, **res_info, "n_frames": n_frames, "models": {}}
            for model in cfg["models"]:
                evs = []
                for seed in cfg["seeds"]:
                    r = run_cv(cache, model, folds, seed, images=images, feats=feats,
                               train_cfg=cfg.get("train"), device=device)
                    evs.append(evaluate(cache.labels, r["oof"]))
                row["models"][model] = {k: _mean_std([e[k] for e in evs]) for k in ("accuracy", "macro_f1")}
                pred = r["oof"].argmax(1)
                fn = lambda a, b: M.classification_metrics(a, b, N_CLASSES)["macro_f1"]  # noqa: E731
                row["models"][model]["macro_f1_ci95_last_seed"] = M.bootstrap_ci(
                    fn, cache.labels, pred, cache.groups, cfg["n_boot"])
                print(f"[stft:{study}] {s} {model} f1={row['models'][model]['macro_f1']['mean']:.4f}", flush=True)
            run.log(experiment="stft", **row)
            rows.append(row)
    return {"rows": rows}


def snr_sweep(cache: Cache, cfg: dict, device, run) -> dict:
    """RQ2: accuracy against additive white noise injected before the STFT.

    Models are trained on the clean recordings (plus one CNN trained with
    signal-domain noise augmentation) and tested on noisy versions of the
    held-out subjects. Noise realisations are identical for every model.
    """
    spec = dsp.SpecConfig(**cfg.get("spec", {}))
    folds = subject_kfold(cache.manifest, cfg["n_splits"], cfg["split_seed"])
    y = cache.labels
    snrs = cfg["snr_db"]
    images = spectrogram_images(cache, spec)
    feats, _ = feature_table(cache, spec)
    noisy_img = {s: spectrogram_images(cache, spec, snr_db=s, seed=cfg["noise_seed"]) for s in snrs}
    noisy_feat = {s: feature_table(cache, spec, snr_db=s, seed=cfg["noise_seed"])[0] for s in snrs}
    print("[snr] noisy representations computed", flush=True)

    def aug_fn(seed):
        def make(train_idx):
            aug = augment.SignalAugment(snr_db_range=tuple(cfg["augment"]["snr_db_range"]),
                                        max_shift=cfg["augment"]["max_shift"],
                                        speed_range=tuple(cfg["augment"]["speed_range"]),
                                        p_noise=cfg["augment"]["p_noise"], seed=seed)

            def transform(batch_idx):
                rec = np.sort(train_idx[batch_idx])
                order = np.argsort(np.argsort(train_idx[batch_idx]))
                x = torch.from_numpy(np.ascontiguousarray(cache.signals[rec]))[order]
                return (dsp.model_input(aug(x), spec, cache.prf_hz),)

            return transform
        return make

    variants = [(m, False) for m in cfg["models"]] + [(m, True) for m in cfg.get("augmented_models", [])]
    out = {"snr_db": snrs, "models": {}}
    for model, use_aug in variants:
        name = model + ("+aug" if use_aug else "")
        per_seed = []
        for seed in cfg["seeds"]:
            def test_fn(fold, predict, model=model):
                te = fold["test"]
                src = noisy_feat if model in CLASSICAL else noisy_img
                return {s: predict(src[s][te]) for s in snrs}

            res = run_cv(cache, model, folds, seed, images=images, feats=feats, train_cfg=cfg.get("train"),
                         device=device, test_inputs_fn=test_fn,
                         batch_transform_fn=aug_fn(seed) if use_aug else None)
            curve = {"clean": evaluate(y, res["oof"])}
            for s in snrs:
                logits = np.empty_like(res["oof"])
                for fold, ex in zip(folds, res["extras"], strict=True):
                    logits[fold["test"]] = ex[s]
                curve[str(s)] = evaluate(y, logits)
            per_seed.append(curve)
        out["models"][name] = {
            cond: {k: _mean_std([c[cond][k] for c in per_seed]) for k in ("accuracy", "macro_f1")}
            | {"ece": _mean_std([c[cond]["calibration_raw"]["ece"] for c in per_seed])}
            for cond in per_seed[0]
        }
        run.log(experiment="snr", model=name, curve=out["models"][name])
        print(f"[snr] {name} " + " ".join(
            f"{c}:{v['macro_f1']['mean']:.3f}" for c, v in out["models"][name].items()), flush=True)
    return out


def cross_session(cache: Cache, cfg: dict, device, run) -> dict:
    """RQ4: unseen collection sessions (other rooms, dates and populations)."""
    spec = dsp.SpecConfig(**cfg.get("spec", {}))
    y, g = cache.labels, cache.groups
    images = spectrogram_images(cache, spec)
    feats, _ = feature_table(cache, spec)
    out = {"splits": {}}
    for name, test_sessions in cfg["splits"].items():
        sp = session_split(cache.manifest, test_sessions, cfg["split_seed"])
        tr, va, te = sp["train"], sp["val"], sp["test"]
        entry = {"n_train": len(tr), "n_val": len(va), "n_test": len(te),
                 "dropped_test_recordings": sp["dropped_test_recordings"],
                 "test_class_counts": np.bincount(y[te], minlength=N_CLASSES), "models": {}}
        for model in cfg["models"]:
            evs = []
            for seed in cfg["seeds"]:
                if model in CLASSICAL:
                    m, _ = _fit_classical(model, feats[tr], y[tr], feats[va], y[va], seed)
                    classes = m.classes_ if model == "random_forest" else m[-1].classes_
                    lt, lv = _classical_logits(m, feats[te], classes), _classical_logits(m, feats[va], classes)
                else:
                    tc = TrainConfig(model=model, seed=seed, **cfg.get("train", {}))
                    m, _ = train_model(images[tr], y[tr], images[va], y[va], N_CLASSES, tc, device)
                    lt, lv = predict_logits(m, images[te], device), predict_logits(m, images[va], device)
                ev = evaluate(y[te], lt, y[va], lv, groups=g[te], n_boot=cfg["n_boot"] if seed == cfg["seeds"][0] else 0)
                ev["confusion"] = M.confusion(y[te], lt.argmax(1), N_CLASSES)
                ev["val_macro_f1"] = M.classification_metrics(y[va], lv.argmax(1), N_CLASSES)["macro_f1"]
                evs.append(ev)
            e = {k: _mean_std([v[k] for v in evs]) for k in ("accuracy", "macro_f1", "balanced_accuracy", "val_macro_f1")}
            e["ece_raw"] = _mean_std([v["calibration_raw"]["ece"] for v in evs])
            e["ece_ts"] = _mean_std([v["calibration_ts"]["ece"] for v in evs])
            e["macro_f1_ci95_seed0"] = evs[0]["macro_f1_ci95"]
            e["confusion_seed0"] = evs[0]["confusion"]
            entry["models"][model] = e
            run.log(experiment="cross_session", split=name, model=model, **{k: e[k] for k in ("accuracy", "macro_f1")})
            print(f"[cross_session] {name} {model} f1={e['macro_f1']['mean']:.4f} "
                  f"(val {e['val_macro_f1']['mean']:.4f})", flush=True)
        out["splits"][name] = entry
    return out


def multiview_fusion(cache: Cache, cfg: dict, device, run) -> dict:
    """Fusion ablation on two RADAR views of the same recordings.

    This is not a multimodal (radar + vision) experiment: both inputs are
    computed from the same radar signal and are therefore exactly aligned. It
    exercises the fusion code path on real paired inputs and asks whether the
    range-time view adds information to the Doppler-time view.
    """
    spec = dsp.SpecConfig(**cfg.get("spec", {}))
    folds = subject_kfold(cache.manifest, cfg["n_splits"], cfg["split_seed"])
    y, g = cache.labels, cache.groups
    a = spectrogram_images(cache, spec)
    b = range_time_images(cache)
    out, logits = {}, {}
    fn = lambda yy, pp: M.classification_metrics(yy, pp, N_CLASSES)["macro_f1"]  # noqa: E731

    def record(name, per_seed_logits):
        evs = [evaluate(y, lg) for lg in per_seed_logits]
        out[name] = {k: _mean_std([e[k] for e in evs]) for k in ("accuracy", "macro_f1", "balanced_accuracy")}
        out[name]["macro_f1_ci95_seed0"] = M.bootstrap_ci(fn, y, per_seed_logits[0].argmax(1), g, cfg["n_boot"])
        run.log(experiment="multiview", variant=name, **out[name])
        print(f"[multiview] {name} f1={out[name]['macro_f1']['mean']:.4f}", flush=True)

    for name, img in (("doppler_time_only", a), ("range_time_only", b)):
        logits[name] = [run_cv(cache, cfg["backbone"], folds, s, images=img, train_cfg=cfg.get("train"),
                               device=device)["oof"] for s in cfg["seeds"]]
        record(name, logits[name])
    record("late_fusion", [np.log(late_fusion(softmax(la, 1), softmax(lb, 1)))
                           for la, lb in zip(logits["doppler_time_only"], logits["range_time_only"], strict=True)])
    for kind in cfg["fusions"]:
        record(f"{kind}_fusion", [run_cv(cache, cfg["backbone"], folds, s, images=a, images_b=b, fusion=kind,
                                         train_cfg=cfg.get("train"), device=device)["oof"] for s in cfg["seeds"]])
    return {"variants": out, "note": "two radar views of one recording; not radar+vision"}


def latency(cache: Cache, cfg: dict, device, run) -> dict:
    """RQ6: parameters, memory and measured inference latency (batch size 1)."""
    from .models import build_model

    spec = dsp.SpecConfig(**cfg.get("spec", {}))
    reps, warm = cfg["repeats"], cfg["warmup"]

    def timeit(fn, sync=None):
        for _ in range(warm):
            fn()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            if sync:
                sync()
            ts.append((time.perf_counter() - t0) * 1e3)
        ts = np.array(ts)
        return {"median_ms": float(np.median(ts)), "p95_ms": float(np.quantile(ts, 0.95)), "n": reps}

    out = {"window_seconds": cache.signals.shape[-1] / cache.prf_hz, "torch_threads": torch.get_num_threads(), "stages": {}, "models": {}}
    # Front end from a cached recording (range bins x slow time).
    x1 = torch.from_numpy(np.ascontiguousarray(cache.signals[:1]))
    out["stages"]["stft_frontend_cpu"] = timeit(lambda: dsp.model_input(x1, spec, cache.prf_hz))
    # Range FFT + MTI on a synthetic raw array of the dataset's shape (timing only).
    raw = (np.random.default_rng(0).standard_normal((cache.signals.shape[-1], 128)) + 0j).astype(np.complex64)
    out["stages"]["range_fft_mti_cpu"] = timeit(
        lambda: dsp.mti_highpass(dsp.range_profiles(raw), cache.prf_hz, axis=0))
    s, f = power_specs(cache, spec, np.arange(1))
    out["stages"]["handcrafted_features_cpu"] = timeit(
        lambda: features.extract_features(s[0], f, cache.prf_hz / spec.hop))

    img = dsp.model_input(x1, spec, cache.prf_hz)
    devices = [torch.device("cpu")] + ([device] if device.type != "cpu" else [])
    for name in cfg["models"]:
        kw = {"n_doppler": img.shape[-2]} if name in ("temporal_cnn", "transformer") else {}
        if name == "transformer":
            kw["n_time"] = img.shape[-1]
        model = build_model(name, N_CLASSES, **kw).eval()
        acts = []
        def hook(_m, _i, o, acts=acts):
            if isinstance(o, torch.Tensor):
                acts.append(o.numel() * o.element_size())

        hooks = [m.register_forward_hook(hook) for m in model.modules() if not list(m.children())]
        with torch.no_grad():
            model(img)
        for h in hooks:
            h.remove()
        n_par = count_parameters(model)
        entry = {"parameters": n_par, "weights_mb_float32": n_par * 4 / 2**20,
                 "activations_mb_batch1_upper_bound": float(sum(acts)) / 2**20}
        for dev in devices:
            m, xi = model.to(dev), img.to(dev)
            sync = (lambda: torch.mps.synchronize()) if dev.type == "mps" else (
                (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else None)
            with torch.no_grad():
                entry[f"latency_{dev.type}"] = timeit(lambda m=m, xi=xi: m(xi).cpu(), sync)
        out["models"][name] = entry
        print(f"[latency] {name} {entry}", flush=True)

    # Classical models: fit once on features to time prediction (accuracy not reported here).
    feats, _ = feature_table(cache, spec, np.arange(min(600, len(cache.manifest))))
    yy = cache.labels[: len(feats)]
    for name in CLASSICAL:
        m, _ = _fit_classical(name, feats, yy, feats, yy, 0)
        out["models"][name] = {"latency_cpu": timeit(lambda m=m: m.predict_proba(feats[:1])),
                               "n_features": feats.shape[1]}
    return out


EXPERIMENTS = {
    "baselines": baselines,
    "stft_ablation": stft_ablation,
    "snr_sweep": snr_sweep,
    "cross_session": cross_session,
    "multiview_fusion": multiview_fusion,
    "latency": latency,
}

__all__ = ["EXPERIMENTS", "log_softmax"]
