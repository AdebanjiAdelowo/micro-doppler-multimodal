PY ?= python

.PHONY: install test lint data experiments figures serve demo

install:
	pip install -e ".[data,serve,demo,dev]"

test:
	pytest

lint:
	ruff check src scripts tests app

data:
	$(PY) scripts/download_data.py --yes
	$(PY) scripts/prepare_data.py

experiments:
	for c in baselines stft_ablation snr_sweep cross_session multiview_fusion latency; do \
		$(PY) scripts/run_experiment.py configs/$$c.yaml || exit 1; done

figures:
	$(PY) scripts/make_figures.py

serve:
	uvicorn mdradar.serve.api:app --port 8000

demo:
	streamlit run app/streamlit_app.py
