# Fire-Vision demo API: ONNX Runtime only (no PyTorch), runs as uid 1000 (HF Spaces compatible).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=7860
RUN useradd --create-home --uid 1000 app
WORKDIR /home/app/service

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app:app firevision ./firevision
COPY --chown=app:app models ./models
COPY --chown=app:app public ./public
COPY --chown=app:app index.py .

USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health', timeout=4)"
CMD ["sh", "-c", "uvicorn index:app --host 0.0.0.0 --port ${PORT}"]
