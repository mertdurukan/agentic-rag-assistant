# Deploy hedefi: Hugging Face Spaces (Docker SDK) veya Railway.
# Not: Embedding/reranker modelleri ilk istekte indirilir; isterseniz build
# aşamasında ön-indirme ekleyerek soğuk başlangıcı kısaltabilirsiniz.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000
# HF Spaces 7860 bekler; Railway $PORT enjekte eder. Varsayılan 8000.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
