# CPU inference image for the FastAPI service. Training is not done in this image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve]"

# Mount a trained checkpoint at /app/checkpoints (checkpoints are not baked into the image).
ENV MDRADAR_CHECKPOINT=/app/checkpoints/light_cnn.pt
RUN useradd --create-home app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "mdradar.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
