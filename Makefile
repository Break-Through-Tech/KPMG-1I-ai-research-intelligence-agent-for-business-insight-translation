PYTHON ?= ./venv/bin/python
# arXiv id or saved HTML page for `make parse` / `make chunk` / `make embed` / `make store`
PAPER ?= tests/fixtures/arxiv_html/dualquant_2610.13767v1.html

.PHONY: ingest agent test test-parse test-chunk test-embed test-store parse chunk embed store

ingest:
	$(PYTHON) -m scripts.backfill_papers

agent:
	$(PYTHON) -m src.agent

ui:
	$(PYTHON) -m streamlit run app.py

test:
	$(PYTHON) -m pytest tests

test-parse:
	$(PYTHON) -m pytest tests/test_arxiv_blocks.py -v

test-chunk:
	$(PYTHON) -m pytest tests/test_chunking.py -v

test-embed:
	$(PYTHON) -m pytest tests/test_embeddings.py -v

test-store:
	$(PYTHON) -m pytest tests/test_vector_store.py -v

parse:
	$(PYTHON) -m src.arxiv_html $(PAPER)

chunk:
	$(PYTHON) -m src.chunking $(PAPER)

embed:
	$(PYTHON) -m src.embeddings $(PAPER)

store:
	$(PYTHON) -m src.vector_store $(PAPER)
