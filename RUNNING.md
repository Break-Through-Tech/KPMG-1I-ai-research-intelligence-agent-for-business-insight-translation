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
make ingest
```

This looks up every arXiv ID in `data/seen_papers.json` (plus the original PDFs in `data/`) on arXiv and stores one embedding per abstract, tagged with its arXiv ID, in the local `qdrant_data/` database. It takes a few minutes. The first run downloads the `all-MiniLM-L6-v2` embedding model.

Rerunning it overwrites papers instead of adding duplicates. To start over, delete `qdrant_data/` and run it again.

Check how many vectors are stored (expect one per paper, about 245):

```bash
./venv/bin/python -c "from qdrant_client import QdrantClient; c = QdrantClient(path='./qdrant_data'); print(c.count('papers').count); c.close()"
```

Local Qdrant allows one process at a time, so close the agent before running this.

## 3. Ask the agent a question

```bash
make agent
```

At the prompt, enter a research question:

```text
Enter your research query: How can a system decide which AI model to route a query to when estimating each model's value is expensive?
```

The agent:

1. Finds the 3 most similar abstracts in Qdrant and puts their arXiv IDs in the LangGraph state.
2. Fetches each paper's H2 sections from `arxiv.org/html/<arxiv_id>`.
3. Sends the abstracts and sections to `qwen2.5:7b`, which answers with cited claims.

It prints the answer, the cited paper IDs with supporting quotes, and the 3 retrieved papers, numbered [1] to [3]. A run takes 1 to 3 minutes.

### Follow-up questions

After an answer you can keep asking. The agent remembers the last 3 turns and handles a follow-up in one of two ways (logic in `src/followup.py`):

- **About the papers already shown** ("What are the limitations of paper 2?", "Explain that more simply", "What does this mean for a bank?"): it answers from those papers again, without a new search or arXiv fetch, so it is faster.
- **Needs different papers** ("What about in healthcare?", "Are there other papers on this topic?", or a new topic): it searches again. If you ask for *other* papers, the ones already shown are left out.

Either way the follow-up is first rewritten into a question that makes sense on its own, and the agent prints it as `(Understood as: ...)` when it changed. References to a shown paper (`paper 2`, `the second paper`, `[2]`, its arXiv ID or title, `these papers`) are recognized without an LLM call; anything else costs one short extra call to `qwen2.5:7b`.

After each answer the agent suggests three follow-ups. Type a number to ask one, `/new` to start a new conversation, or press Enter on an empty line to quit.

From code, pass the previous result's history back in:

```python
first = run_agent("How can hallucination in LLMs be reduced?")
second = run_agent("What are the limitations of paper 2?", first["history"])
```

## 4. Run the tests

```bash
make test
```

The tests run offline: arXiv, the embedding model, and Qdrant are faked or in memory.

## 5. Run commands without Make

The Make targets are shortcuts for these commands:

```bash
./venv/bin/python -m scripts.backfill_papers
./venv/bin/python -m src.agent
./venv/bin/python -m pytest tests
```

## Scrape new arXiv papers into the vector store

The scraper finds new arXiv papers and stores each paper's abstract in Qdrant,
tagged with its arXiv ID. The agent fetches a paper's full sections from
`arxiv.org/html/<arxiv_id>` only when it needs them. No PDFs are downloaded.

```bash
python3 -m scripts.scrape_papers      # new papers only; safe to re-run
python3 -m scripts.backfill_papers    # one-off: store papers scraped before this change
python3 -m unittest tests.test_scrape_papers -v
```

Stored paper IDs are recorded in `data/seen_papers.json`. Leave `QDRANT_URL` empty
locally to use `./qdrant_data`.

The scheduled GitHub Action (Mon/Wed/Fri) runs `python -m scripts.scrape_papers --record-only`:
it only adds new IDs to `data/seen_papers.json` and doesn't add anything to the vector store.

## Local data

- `data/seen_papers.json` lists the arXiv IDs that `make ingest` loads.
- `qdrant_data/` is the generated local Qdrant database and is ignored by Git.
- Deleting `qdrant_data/` removes all locally stored vectors and payloads.

If the agent reports that the `papers` collection does not exist, run `make ingest` first.
