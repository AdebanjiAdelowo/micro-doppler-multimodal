# Micro-Doppler Multimodal Intelligence

Radar activity classification with signal processing, classical and neural baselines, robustness experiments, calibration, and inference tooling.

## Methods

The `mdradar` package preprocesses radar recordings into range-bin signals and Doppler-time spectrograms. Experiment configurations compare SVM, random forest, CNN, temporal CNN, and transformer models. Additional experiments examine STFT settings, signal-to-noise ratio, held-out collection sessions, calibration, and inference latency.

The fusion experiment combines Doppler-time and range-time views derived from the same radar recording. It does not evaluate radar and camera fusion.

## Installation and usage

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[data,serve,demo,dev]"
python scripts/download_data.py --help
python scripts/prepare_data.py --help
```

After downloading and preparing the dataset:

```bash
python scripts/run_experiment.py configs/baselines.yaml --device auto
pytest
uvicorn mdradar.serve.api:app --port 8000
streamlit run app/streamlit_app.py
```

Experiment outputs are written under `results/`. Configurations for session transfer, noise robustness, fusion, and latency are in `configs/`.

## Evaluation limitations

This repository contains experiment implementations and configurations but no committed benchmark result files. Quantitative accuracy, robustness, and latency conclusions require completed evaluations on the prepared dataset. Synthetic fixtures used by tests do not establish performance on real recordings.

## Repository structure

- `src/mdradar/`: preprocessing, models, training, evaluation, and serving.
- `scripts/`: dataset preparation and experiment entry points.
- `configs/`: experiment settings.
- `tests/`: signal processing, metrics, models, and API checks.
- `kaggle/`: GPU experiment tooling.
- `app/`: Streamlit interface.

## License

The software is distributed under the [MIT license](LICENSE). Dataset usage is subject to its own terms.
