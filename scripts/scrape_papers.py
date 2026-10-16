"""
Scrape new arXiv papers and store them in the vector store (Qdrant), section by section.

For each new paper:
  1. Get its info from arXiv (title, authors, abstract...) using its arXiv ID.
  2. Open the paper's HTML version (arxiv.org/html/<id>) and split it into
     sections at every <h2> header (Introduction, Method, Results, ...).
  3. Store the abstract + every section in Qdrant, each one tagged with the
     arXiv ID so all pieces of a paper stay connected.
     If arXiv has no HTML version, only the abstract is stored.

No PDFs are downloaded or committed to GitHub.

Rescraping is safe to run any number of times:
  - data/seen_papers.json skips papers we've already stored
  - a paper is only marked "seen" after it was saved successfully,
    so a paper that failed is retried on the next run
  - chunk IDs are fixed per paper, so re-storing a paper never creates duplicates

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

from src.arxiv_html import fetch_sections, split_long_text  # noqa: E402
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


def build_chunks(paper, sections):
    """(section_title, text) pairs to store: the abstract first, then each section.

    Long sections are split into smaller pieces because the embedding model
    only reads about the first 1000 characters of any text.
    """
    chunks = [("Abstract", paper.abstract)]
    for section in sections or []:
        for piece in split_long_text(section.text):
            chunks.append((section.title, piece))
    return chunks


def ingest_paper(result, vector_client, fetch=fetch_sections, store=None, delete=None):
    """Store one paper (abstract + sections) in Qdrant. Returns how many chunks were stored."""
    if store is None or delete is None:
        from src.vector_store import delete_paper, store_paper_chunks
        store = store or store_paper_chunks
        delete = delete or delete_paper

    paper = build_paper(result)
    sections = fetch(paper.paper_id)
    if sections is None:
        print("  No HTML version on arXiv - storing the abstract only")
    else:
        print(f"  Found {len(sections)} section(s): {', '.join(s.title for s in sections)}")

    chunks = build_chunks(paper, sections)
    delete(vector_client, paper.paper_id)  # clear any half-saved chunks from an earlier failed run
    store(vector_client, paper, chunks)
    print(f"  Stored {len(chunks)} chunk(s) in the vector store")
    return len(chunks)


def _default_vector_client():
    from src.vector_store import get_client

    if os.getenv("GITHUB_ACTIONS") and not os.getenv("QDRANT_URL"):
        raise RuntimeError(
            "QDRANT_URL is not set. In GitHub Actions a local Qdrant store is deleted "
            "when the job ends. Add QDRANT_URL / QDRANT_API_KEY as repository secrets."
        )
    return get_client(path=str(VECTOR_STORE_PATH))


def scrape_new_papers(vector_client=None, **ingest_kwargs):
    if vector_client is None:
        vector_client = _default_vector_client()

    seen_ids = load_seen_ids()

    client = arxiv.Client()
    search = arxiv.Search(
        query=QUERY,
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    new_count, failed = 0, []
    try:
        for result in client.results(search):
            paper_id = get_paper_id(result)

            # `continue`, not `break`, so a paper that failed last time is retried
            if paper_id in seen_ids:
                continue

            print(f"New paper: {paper_id} - {result.title.strip()}")
            try:
                ingest_paper(result, vector_client, **ingest_kwargs)
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
    if failed:
        print(f"{len(failed)} paper(s) failed and will be retried next run: {failed}")
    return new_count, failed


if __name__ == "__main__":
    _, failed = scrape_new_papers()
    sys.exit(1 if failed else 0)
