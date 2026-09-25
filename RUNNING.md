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

## 2. Seed the vector store

The seed command reads the five papers in `data/sample_papers.json`, creates one embedding per abstract, and stores the vectors and metadata in the local `qdrant_data/` database.

```bash
make seed-vector-store
```

Expected final output:

```text
Finished: stored 5 papers and 5 chunks
```

The first run downloads the `all-MiniLM-L6-v2` embedding model. Later runs reuse the downloaded model.

Qdrant uses stable point IDs, so rerunning this command with the same paper IDs overwrites those points instead of adding duplicates.

## 3. Inspect the stored abstracts

Print the number of points and the payload of every stored abstract:

```bash
make test-seed-vector-store
```

The output should begin with:

```text
Total points: 5
```

This command prints payloads but not the 384-dimensional numerical vectors.

## 4. Run the retrieval agent

Start the LangGraph retrieval workflow:

```bash
make agent
```

At the prompt, enter a research question:

```text
Enter your research query: How can AI explain medical reports to patients?
```

The agent:

1. Embeds the query with `all-MiniLM-L6-v2`.
2. Searches the local Qdrant `papers` collection.
3. Retrieves the three most similar abstracts.
4. Writes their `paper_id` values into the LangGraph state.

Example output:

```text
Relevant document IDs:
paper_005
paper_003
paper_002
```

The exact ranking depends on the query.

## 5. Run commands without Make

The Make targets are shortcuts for these Python commands:

```bash
python3 -m scripts.seed_vector_store
python3 -m tests.test_seed_vector_store
python3 -m src.agent
```

## Optional: download recent arXiv PDFs

The scraper is separate from the vector-store seed process. It downloads PDFs but does not add them to Qdrant.

Install its additional dependency:

```bash
pip install arxiv
```

Run it from the `scripts` directory because its data paths are relative to that directory:

```bash
cd scripts
python scrape_papers.py
cd ..
```

Downloaded PDFs are placed in `data/`, and their IDs are recorded in `data/seen_papers.json`.

## Local data

- `data/sample_papers.json` is the source for the five seeded abstracts.
- `qdrant_data/` is the generated local Qdrant database and is ignored by Git.
- Deleting `qdrant_data/` removes all locally stored vectors and payloads.

If the agent reports that the `papers` collection does not exist, run `make seed-vector-store` first.
