FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/deps -r requirements.txt

# ── Runtime image ──────────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /deps /usr/local
COPY *.py ./

RUN useradd -m -u 1000 watcher \
    && chown -R watcher:watcher /app
USER watcher

ENV LOG_LEVEL=INFO
ENV ENRICH_ENABLED=true
ENV ENRICH_CONCURRENCY=2

CMD ["python", "main.py"]
