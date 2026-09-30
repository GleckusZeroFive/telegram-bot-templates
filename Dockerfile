FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-spa \
    && rm -rf /var/lib/apt/lists/*

# Python deps (heavy layers, cached while requirements.txt is unchanged)
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.5.1+cpu \
      --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Models are baked into the image so the container starts without network access
# to Hugging Face (compose sets HF_HUB_OFFLINE=1). HF_HOME is outside volume mounts.
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-large
ARG WHISPER_MODEL=small
ENV HF_HOME=/app/models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')" \
    && python -c "from faster_whisper import download_model; download_model('${WHISPER_MODEL}')"

# App code
COPY alembic.ini .
COPY alembic/ ./alembic/
COPY scripts/ ./scripts/
COPY app/ ./app/

# Intent classifier weights (not stored in git)
RUN python scripts/download_intent_model.py

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
