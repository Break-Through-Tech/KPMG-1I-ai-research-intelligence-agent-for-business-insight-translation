# Running the Project

Run all commands from the repository root unless a section says otherwise.

## 1. Set up Python

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Install [Ollama](https://ollama.com) and pull the model the agent uses:

```bash
ollama pull qwen2.5:7b
```

## 2. Load papers into the vector store

```bash
caffeinate -i make ingest 2>&1 | tee ingest.log
```

`caffeinate -i` (macOS) keeps the laptop awake and `tee` saves the output; plain `make ingest` also works.

This ingests the fixed corpus in `data/corpus_ids.txt` (100 papers: the ones the retrieval eval questions expect,
plus others spread over Sep/Oct/Nov 2026). For each paper it:

1. Gets the title and abstract from arXiv.
2. Downloads `arxiv.org/html/<arxiv_id>` and parses it into paragraphs with their section headings
   (`src/arxiv_html.py`). Tables and figures are skipped for now.
3. Packs the paragraphs into chunks of about 500 tokens that never cross a top-level section (`src/chunking.py`).
4. Embeds and stores them in the local `qdrant_data/` database (`src/embeddings.py`, `src/vector_store.py`):
   - `papers_high_level`: one point per paper, a dense vector of title + abstract.
   - `paper_chunks`: one point per chunk, with a dense (Nomic v1.5, 768 dimensions) and a sparse (BM25) vector.

The first run downloads the two embedding models (about 650 MB). The whole corpus takes about an hour and a few
GB of memory, so avoid running other heavy jobs at the same time.

Rerunning it is safe:

- Papers that are already fully stored are skipped, so an interrupted run picks up where it stopped.
- A paper is only counted as stored once both its record and its chunks are written; a paper that failed partway
  is redone on the next run.
- Papers with no HTML version on arXiv yet are skipped (each one says so in the output); the next run tries them
  again.
- `make ingest-force` re-ingests every paper. Storing a paper again overwrites its points and removes any of its
  old chunks that no longer exist, so nothing is duplicated.
- `./venv/bin/python -m scripts.backfill_papers --all` ingests every paper in `data/seen_papers.json` instead of
  the corpus.

To start over, delete `qdrant_data/` and run it again.

Check how many points are stored:

```bash
./venv/bin/python -c "from qdrant_client import QdrantClient; c = QdrantClient(path='./qdrant_data'); print({x.name: c.count(x.name).count for x in c.get_collections().collections}); c.close()"
```

Local Qdrant allows one process at a time, so don't run two ingests at once.

## 3. The agent (temporarily unavailable)

`make agent`, the Streamlit UI and the API's research endpoint don't work right now. They were built on the old
abstract-only `papers` collection, which has been replaced by `papers_high_level` and `paper_chunks`. Searching the
new collections and moving the agent over to them is the next piece of work.

## 4. Run the tests

```bash
make test
```

Run one stage at a time:

```bash
make test-parse    # HTML -> paragraphs with section headings
make test-chunk    # paragraphs -> chunks
make test-embed    # chunks -> dense and sparse vectors (uses the real models)
make test-store    # vectors -> Qdrant collections (in-memory Qdrant)
make test-ingest   # the whole pipeline, scraper and backfill
make test-retrieval  # paper search and hybrid chunk search (in-memory Qdrant, fake encoders)
```

arXiv and Qdrant are faked or in memory. The embedding tests load the real models, so they need them downloaded
once (any `make ingest` or `make embed` run does that).

## 5. Look at each stage's output

These run one paper through the pipeline and print what each stage produces. Nothing is written to `qdrant_data/`.

```bash
make parse     # paragraphs with their section headings
make chunk     # chunks and their token counts
make embed     # the text each encoder sees and the vectors it returns
make store     # what lands in each collection (in a temporary in-memory Qdrant)
```

They use a saved paper (`tests/fixtures/arxiv_html/dualquant_2610.13767v1.html`) by default. Pass `PAPER` to use
any arXiv ID or saved HTML page:

```bash
make chunk PAPER=2610.31552v1
```

## 6. Search the vector store

Two independent searches over what `make ingest` stored (`src/retrieval.py`):

- **Paper search**: the question's Nomic vector (with the `search_query: ` prefix) against each paper's
  title + abstract in `papers_high_level`. Top 10 papers, scored by cosine similarity.
- **Chunk search**: hybrid search over every chunk in `paper_chunks`. A dense (Nomic) search and a sparse
  (BM25) search each pick their top 100 chunks, and Qdrant merges the two lists with Reciprocal Rank Fusion
  (k = 60). Top 20 chunks, scored by RRF score; only their order is meaningful.

```bash
make search Q="How should operators plan memory capacity for the KV cache when serving LLMs?"
make search-explain Q="..."   # also each chunk's rank and score in the dense-only and sparse-only searches
make search                   # no Q: type one question after another; blank line to quit
```

`./venv/bin/python -m src.retrieval --help` shows the other options (`--papers N`, `--chunks N`; 0 skips that
search). Local Qdrant allows one process at a time, so don't search while `make ingest` is running.

From Python:

```python
from src.vector_store import get_client
from src.retrieval import search_papers, search_chunks

client = get_client()
papers = search_papers(client, "How can companies stop AI chatbots from making things up?")              # list[PaperHit]
chunks = search_chunks(client, "How can companies stop AI chatbots from making things up?", explain=True)  # list[ChunkHit]
client.close()
```

`PaperHit` has `arxiv_id`, `title`, `abstract`, `score` and `rank`. `ChunkHit` has `chunk_id`, `arxiv_id`, `title`,
`content`, `breadcrumb`, `chunk_index` (document order), `content_type`, `score` and `rank`; with `explain=True`
also `dense_rank`, `dense_score`, `sparse_rank` and `sparse_score` (None when that search didn't return the
chunk among its top 100). Result counts, the candidate pool and the RRF constant are arguments with defaults in
`src/retrieval.py`. A missing collection or a Qdrant failure raises `RetrievalError`.

## 7. Run commands without Make

The Make targets are shortcuts for these commands:

```bash
./venv/bin/python -m scripts.backfill_papers            # make ingest (add --force for make ingest-force)
./venv/bin/python -m pytest tests                       # make test
./venv/bin/python -m src.chunking <arxiv_id | page.html>   # make chunk (likewise src.arxiv_html, src.embeddings, src.vector_store)
```

## 8. Run the Stage 1 React/FastAPI frontend

The research endpoint fails until the agent moves to the new collections (see section 3); the rest of this
section still applies.

The React frontend is a separate Vite application. Keep the backend and frontend
running in separate terminals:

**Terminal 1 — FastAPI bridge**

```bash
./venv/bin/python -m uvicorn api.main:app --reload --port 8000
```

**Terminal 2 — React/Vite frontend**

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal, usually `http://localhost:5173`.
Vite proxies `/api` requests to FastAPI on port 8000. The existing Streamlit
fallback is started with `make ui` on port 8501.

Stage 1 connects only Research → loading → Results to the real backend. Latest
AI, Trends, KPMG Focus, Paper Workspace, feedback, saving, sharing, and other
prototype areas are not connected to live backend data yet.

## Scrape new arXiv papers

The scheduled GitHub Action (Mon/Wed/Fri) runs `python -m scripts.scrape_papers --record-only`: it adds new arXiv
IDs to `data/seen_papers.json` and doesn't touch the vector store.

Run without `--record-only`, the scraper ingests each new paper the same way `make ingest` does and marks it seen
only once both collections are written. Papers without HTML or that fail are left unmarked and retried next run.

```bash
./venv/bin/python -m scripts.scrape_papers --record-only   # what the scheduled job does
./venv/bin/python -m scripts.scrape_papers                 # also ingest the new papers
```

Leave `QDRANT_URL` empty locally to use `./qdrant_data`; set `QDRANT_URL` / `QDRANT_API_KEY` to use a Qdrant Cloud
cluster instead.

## Local data

- `data/corpus_ids.txt` lists the papers `make ingest` loads.
- `data/seen_papers.json` lists every paper the scraper has found.
- `qdrant_data/` is the generated local Qdrant database and is ignored by Git. Deleting it removes all locally
  stored vectors and payloads.
