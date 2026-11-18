PYTHON ?= ./venv/bin/python
# arXiv id or saved HTML page for `make parse` / `make chunk` / `make embed` / `make store`
PAPER ?= tests/fixtures/arxiv_html/dualquant_2610.13767v1.html
# research question for `make search` / `make agent`; leave empty to type questions one after another
Q ?=

.PHONY: ingest ingest-force agent test test-parse test-chunk test-embed test-store test-ingest test-retrieval test-agent parse chunk embed store search search-explain

# Stores the papers in data/corpus_ids.txt that aren't in Qdrant yet (QDRANT_URL if set, else ./qdrant_data).
# -u: print each line immediately, even when piped into tee
ingest:
	$(PYTHON) -u -m scripts.backfill_papers

# re ingest all documents
ingest-force:
	$(PYTHON) -u -m scripts.backfill_papers --force

# answer with qwen2.5:7b through Ollama, from the top-ranked papers and chunks that fit in 6,000 tokens
agent:
	$(PYTHON) -m src.agent $(if $(Q),"$(Q)")

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

test-ingest:
	$(PYTHON) -m pytest tests/test_scrape_papers.py -v

test-retrieval:
	$(PYTHON) -m pytest tests/test_retrieval.py -v

test-agent:
	$(PYTHON) -m pytest tests/test_agent.py -v

parse:
	$(PYTHON) -m src.arxiv_html $(PAPER)

chunk:
	$(PYTHON) -m src.chunking $(PAPER)

embed:
	$(PYTHON) -m src.embeddings $(PAPER)

store:
	$(PYTHON) -m src.vector_store $(PAPER)

# top 10 papers (dense) and top 20 chunks (hybrid) from ./qdrant_data
search:
	$(PYTHON) -m src.retrieval $(if $(Q),"$(Q)")

# also shows each chunk's rank and score in the dense-only and sparse-only searches
search-explain:
	$(PYTHON) -m src.retrieval --explain $(if $(Q),"$(Q)")
