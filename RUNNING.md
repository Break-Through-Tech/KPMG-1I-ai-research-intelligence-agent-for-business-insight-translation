# Running the Project

Everything runs locally: the papers live in a local Qdrant database (`qdrant_data/`), the embedding models run
in Python, and the LLM runs in Ollama. Run all commands from the repository root.

## Requirements

- **Python 3.10+** (developed on 3.12)
- **Node.js 20.19+ or 22.12+** (for the Vite frontend)
- **[Ollama](https://ollama.com)**
- About **8 GB of free disk** (the LLM is 4.7 GB, the embedding models about 650 MB) and **16 GB of RAM**
  recommended; the LLM alone uses 6–8 GB while answering.

## 1. Set up

```bash
# Python environment: must be called venv, since the Makefile runs ./venv/bin/python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend
cd frontend && npm install && cd ..

# LLM
ollama pull qwen2.5:7b
```

## 2. Load the papers

```bash
make ingest
```

This downloads the 100 papers listed in `data/corpus_ids.txt` from arXiv, splits them into chunks, embeds them and
stores them in `qdrant_data/`. Expect:

- **About an hour.** The first run also downloads the two embedding models. On a Mac, `caffeinate -i make ingest`
  stops the laptop from sleeping partway.
- **A few papers skipped** with "No HTML version on arXiv": arXiv hasn't converted them yet. Today 96 of the 100
  get stored.

It's safe to stop and rerun: papers already stored are skipped, and a paper that failed halfway is redone.
`make ingest-force` re-ingests everything; deleting `qdrant_data/` starts from scratch.

Check what got stored:

```bash
./venv/bin/python -c "from qdrant_client import QdrantClient; c = QdrantClient(path='./qdrant_data'); print({x.name: c.count(x.name).count for x in c.get_collections().collections}); c.close()"
```

You should see about 96 points in `papers_high_level` and about 3,600 in `paper_chunks`.

## 3. Run the app

Make sure Ollama is running (open the Ollama app, or run `ollama serve`), then:

```bash
make dev
```

Open **http://localhost:5173** and ask a question. An answer takes about a minute. `make dev` starts the API on
port 8000 and the frontend on port 5173; Ctrl+C stops both.

What works: asking a question (the search box, suggested questions or the KPMG Focus buttons), the executive
insight with citation links, the Copy button, and the evidence cards with each passage's paper, section, authors
and date. Latest AI, Trends, View paper, relevance percentages, the research summary numbers and the other
prototype panels are still placeholders.

To run the two servers in separate terminals instead:

```bash
./venv/bin/python -m uvicorn api.main:app --reload --port 8000   # terminal 1: API
cd frontend && npm run dev -- --port 5173                        # terminal 2: frontend
```

## 4. Use it from the command line

```bash
make agent Q="How does quantization affect the inference speed and accuracy of large language models?"
make search Q="How should operators plan memory capacity for the KV cache when serving LLMs?"
```

- `make agent` prints the answer, the papers and passages the LLM was given, and the context size.
- `make search` shows the raw search results (top 10 papers, top 20 passages) without calling the LLM.
  `make search-explain` also shows each passage's rank in the meaning-based and keyword-based searches.
- Leave out `Q` to type one question after another; a blank line quits.

## 5. Run the tests

```bash
make test
```

The tests run offline (arXiv, Ollama and Qdrant are faked or in memory), but a few load the real embedding
models, so run `make ingest` once first.

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| `Storage folder ./qdrant_data is already accessed by another instance of Qdrant client` | Only one process can open `qdrant_data/` at a time. Stop `make dev`, `make agent` or `make search` before running `make ingest`, and the other way round. |
| The page shows no answer and the API returns **503** | The search failed: `make ingest` hasn't been run, or another process has `qdrant_data/` open. |
| The API returns **502** | The LLM failed: Ollama isn't running, or `qwen2.5:7b` isn't pulled. |
| The first answer is slow | The first question loads the embedding models and the LLM, and downloads the Qwen tokenizer from Hugging Face. Later questions take about a minute. |
| `make dev` says a port is in use | Something is already on port 8000 or 5173, often an earlier `make dev`. Stop it and retry. |
| `make: ./venv/bin/python: No such file or directory` | The virtual environment isn't at `./venv`. Create it as in section 1, or run with `make PYTHON=python ...`. |

## Reference

### Make targets

| Target | What it does |
|---|---|
| `make ingest` / `make ingest-force` | Store the corpus papers that aren't stored yet / re-store all of them |
| `make dev` | API (port 8000) and frontend (port 5173) together |
| `make agent Q="..."` | Answer a question from the command line |
| `make search Q="..."` / `make search-explain Q="..."` | Show search results without the LLM |
| `make test` | All tests |
| `make ui` | Old Streamlit UI; currently broken, use `make dev` |

### Research API

`POST /api/research` with `{"query": "..."}` returns:

- `answer`: Markdown, citing papers as `[Paper Title](https://arxiv.org/abs/<arxiv_id>)`.
- `papers`: the papers the LLM was given (`arxiv_id`, `title`, `abstract`, `url`, `rank`, `authors`,
  `publish_date`, `categories`).
- `retrieved_chunks`: the passages the LLM was given (`paper_id`, `title`, `section_title`, `chunk_text`,
  `chunk_index`, `score`, `url`, `authors`, `publish_date`, `categories`).

Questions are answered one at a time. `GET /api/health` returns `{"status": "ok"}`.

### Data files

- `data/corpus_ids.txt`: the papers `make ingest` loads. Add arXiv IDs here and rerun `make ingest` to grow the
  corpus.
- `data/seen_papers.json`: every paper the scheduled scraper has found. `./venv/bin/python -m
  scripts.backfill_papers --all` ingests these instead of the corpus.
- `qdrant_data/`: the local vector database, generated by `make ingest` and ignored by Git.

### Scheduled scraper

A GitHub Action (Mon/Wed/Fri) runs `python -m scripts.scrape_papers --record-only`, which only adds new arXiv IDs to
`data/seen_papers.json`; it doesn't touch any vector store. Without `--record-only`, the scraper also ingests the
new papers the same way `make ingest` does.

### Hosted Qdrant

Leave `QDRANT_URL` unset to use the local `qdrant_data/`. Set `QDRANT_URL` (and `QDRANT_API_KEY` for Qdrant
Cloud) to use a Qdrant server instead; every command picks it up.
