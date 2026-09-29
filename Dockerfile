FROM python:3.12-slim

# unbuffered stdout/stderr so logs (Logfire spans, tracebacks) show up in
# `docker compose logs` immediately instead of batched/delayed
ENV PYTHONUNBUFFERED=1

# libmagic1 backs python-magic's content-based MIME sniffing (app/indexing/document_loader/detection.py)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements.txt doesn't pin torch directly (it's a transitive dep of
# sentence-transformers/langchain-huggingface), so plain `pip install`
# resolves PyPI's default build — which bundles ~2GB of CUDA libraries
# (nvidia-cudnn, nvidia-cublas, etc.) that are completely unused here since
# this container has no GPU. Installing the CPU-only build first means the
# later `pip install -r requirements.txt` sees torch already satisfied and
# skips it, avoiding that multi-GB dead weight entirely.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Installed before the app code is copied so this layer is cached across
# rebuilds unless requirements.txt itself changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8010

# --host 0.0.0.0 is required here (unlike local dev) — 127.0.0.1 inside a
# container is unreachable from outside it even with the port published
CMD ["uvicorn", "app.web.main:app", "--host", "0.0.0.0", "--port", "8010"]
