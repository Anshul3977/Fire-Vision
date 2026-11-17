FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY code_web ./code_web
COPY models ./models
COPY samples ./samples
COPY NOTICE ./NOTICE

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/code_web/static/uploads \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 10000
CMD ["python", "code_web/app.py"]
