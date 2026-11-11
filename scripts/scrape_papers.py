"""
Scrape new arXiv papers and store them in the vector store (Qdrant).

For each new paper:
  1. Get its info from arXiv (title, authors, abstract...) using its arXiv ID.
  2. Fetch its HTML version from arxiv.org/html/<arxiv_id>, parse it into paragraphs
     and chunk them (src/arxiv_html.py, src/chunking.py).
  3. Store the title + abstract in papers_high_level and the chunks in paper_chunks
     (src/vector_store.py).

Papers without an HTML version are skipped and not marked "seen", so they're
retried on the next run. No PDFs are downloaded or committed to GitHub.

Rescraping is safe to run any number of times:
  - data/seen_papers.json skips papers we've already stored
  - a paper is only marked "seen" after both collections were written,
    so a paper that failed is retried on the next run
  - each paper's Qdrant IDs are fixed, so storing it again never creates a duplicate

Run from the repo root:
    python -m scripts.scrape_papers
"""

import json
import os
import sys
from pathlib import Path

import arxiv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))  # so `src` imports work

from src.schema import Paper  # noqa: E402

DATA_DIR = ROOT_DIR / "data"
LOG_FILE = DATA_DIR / "seen_papers.json"
VECTOR_STORE_PATH = ROOT_DIR / "qdrant_data"
QUERY = "LLM routing"
MAX_RESULTS = 20


def load_seen_ids():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            return set(json.load(f))
    return set()


def save_seen_ids(seen_ids):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LOG_FILE, "w") as f:
        json.dump(sorted(seen_ids), f, indent=0)


def get_paper_id(result):
    """'http://arxiv.org/abs/2609.24974v1' -> '2609.24974v1'"""
    return result.entry_id.split("/")[-1]


def build_paper(result):
    """Turn an arXiv search result into our Paper schema."""
    published = getattr(result, "published", None)
    return Paper(
        paper_id=get_paper_id(result),
        title=result.title.strip(),
        authors=[a.name for a in result.authors],
        abstract=" ".join(result.summary.split()),
        categories=list(getattr(result, "categories", []) or []),
        publish_date=published.date() if published else None,
        pdf_url=result.pdf_url,
        comments=getattr(result, "comment", None),
    )

def ingest_paper(result, vector_client):
    """Parse, chunk and store one paper. Returns the Paper, or None if arXiv has no HTML
    version of it yet."""
    # Imported here so --record-only runs with just arxiv and pydantic installed.
    from src.arxiv_html import fetch_blocks
    from src.chunking import chunk_blocks
    from src.vector_store import store_chunks, store_paper_record

    paper = build_paper(result)
    blocks = fetch_blocks(paper.paper_id)
    if blocks is None:
        print("  No HTML version on arXiv; skipped, will retry next run")
        return None
    chunks = chunk_blocks(blocks)

    store_paper_record(vector_client, paper)
    store_chunks(vector_client, paper, chunks)
    print(f"  Stored paper record and {len(chunks)} chunks")
    return paper

def _default_vector_client():
    from src.vector_store import get_client

    if os.getenv("GITHUB_ACTIONS") and not os.getenv("QDRANT_URL"):
        raise RuntimeError(
            "QDRANT_URL is not set. In GitHub Actions a local Qdrant store is deleted "
            "when the job ends. Add QDRANT_URL / QDRANT_API_KEY as repository secrets."
        )
    return get_client(path=str(VECTOR_STORE_PATH))


def scrape_new_papers(vector_client=None, record_only=False):
    """record_only=True only adds new IDs to seen_papers.json, without touching Qdrant."""
    if vector_client is None and not record_only:
        vector_client = _default_vector_client()

    seen_ids = load_seen_ids()

    client = arxiv.Client()
    search = arxiv.Search(
        query=QUERY,
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    new_count, failed, no_html = 0, [], []
    try:
        for result in client.results(search):
            paper_id = get_paper_id(result)

            # `continue`, not `break`, so a paper that failed last time is retried
            if paper_id in seen_ids:
                continue

            print(f"New paper: {paper_id} - {result.title.strip()}")
            if not record_only:
                try:
                    if ingest_paper(result, vector_client) is None:
                        no_html.append(paper_id)
                        continue
                except Exception as e:
                    print(f"  FAILED {paper_id}: {e}")
                    failed.append(paper_id)
                    continue

            seen_ids.add(paper_id)  # only after it was saved successfully
            new_count += 1
    finally:
        save_seen_ids(seen_ids)
        close = getattr(vector_client, "close", None)
        if close:
            close()

    print(f"Done. {new_count} new paper(s) added.")
    if no_html:
        print(f"{len(no_html)} paper(s) have no HTML version yet and will be retried next run: {no_html}")
    if failed:
        print(f"{len(failed)} paper(s) failed and will be retried next run: {failed}")
    return new_count, failed


if __name__ == "__main__":
    _, failed = scrape_new_papers(record_only="--record-only" in sys.argv)
    sys.exit(1 if failed else 0)
