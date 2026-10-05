FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    BM25_STORE_PATH=/app/bm25/bm25_index

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

RUN addgroup --system chatbot \
    && adduser --system --ingroup chatbot chatbot

COPY --chown=chatbot:chatbot . .

RUN mkdir -p /app/logs /app/bm25 /app/.cache/huggingface \
    && chown -R chatbot:chatbot /app/logs /app/bm25 /app/.cache

USER chatbot

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=5m --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/api/health', timeout=5)"]

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
