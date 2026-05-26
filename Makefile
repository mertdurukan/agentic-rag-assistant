.PHONY: install db ingest eval serve test lint

install:
	pip install -r requirements.txt

db:                ## Postgres+pgvector'ı ayağa kaldır
	docker compose up -d

ingest:            ## arXiv'den çek, chunk'la, indeksle
	python -m scripts.ingest

eval:              ## RAGAS baseline vs hybrid karşılaştırması
	python -m src.eval.ragas_eval --mode both

serve:             ## API + UI (http://localhost:8000)
	uvicorn app.main:app --host 0.0.0.0 --port 8000

test:
	pytest

lint:
	ruff check .
