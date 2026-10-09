"""FastAPI inference service.

    MDRADAR_CHECKPOINT=checkpoints/light_cnn.pt uvicorn mdradar.serve.api:app

The service wraps the trained classifier only. It returns an activity label,
calibrated class probabilities and an abstention flag.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from ..inference import INTENDED_USE, Predictor

MAX_UPLOAD_BYTES = 64 * 2**20


def create_app(checkpoint: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="mdradar activity classifier", version="0.1.0", description=INTENDED_USE)
    path = Path(checkpoint or os.environ.get("MDRADAR_CHECKPOINT", "checkpoints/light_cnn.pt"))
    state: dict = {"predictor": Predictor.load(path) if path.exists() else None, "path": str(path)}

    def predictor() -> Predictor:
        if state["predictor"] is None:
            raise HTTPException(503, f"no checkpoint at {state['path']}; train one with scripts/train.py")
        return state["predictor"]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_loaded": state["predictor"] is not None}

    @app.get("/model")
    def model() -> dict:
        return predictor().describe()

    @app.post("/predict")
    async def predict(
        file: UploadFile = File(..., description="raw .dat recording, or .npy of complex range bins (n_range, n_slow)"),
        abstain_below: float = Query(0.0, ge=0.0, le=1.0, description="abstain when confidence is below this value"),
    ) -> dict:
        raw = await file.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "file too large")
        try:
            p = predictor().predict_file(file.filename or "upload.dat", raw, abstain_below)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return {
            "activity": None if p.abstained else p.label,
            "abstained": p.abstained,
            "top_class": p.label,
            "confidence": p.confidence,
            "probabilities": p.probabilities,
            "timings_ms": p.timings_ms,
            "note": "Activity label for one radar recording; not an assessment of identity or intent.",
        }

    return app


app = create_app()
