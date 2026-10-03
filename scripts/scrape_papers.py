"""
Scrape new arXiv papers and store each one in BOTH storage layers (issue #15):
  1. Extract title, authors, abstract (+ categories, date, URL) and the PDF.
  2. Upload the PDF to object storage (Backblaze B2) and get its object ID.
  3. Add the paper to vector storage (Qdrant) with its metadata + object ID.

PDFs go to a temporary folder and are deleted afterwards, so they're never
committed to GitHub anymore.

Run from the repo root:
    python -m scripts.scrape_papers
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

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


def build_paper(result, object_key):
    """Turn an arXiv result into our Paper schema, linked to its PDF in B2."""
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
        object_storage_key=object_key,
    )


def ingest_paper(result, storage, vector_client, store_embeddings, local_pdf_path=None):
    """Store one paper in B2 AND Qdrant. Returns the Paper."""
    paper_id = get_paper_id(result)
    object_key = storage.object_key(paper_id)

    # PDF -> B2 (skip if a previous run already uploaded it)
    if storage.pdf_exists(paper_id):
        print(f"  PDF already in object storage: {object_key}")
    elif local_pdf_path:
        object_key = storage.upload_pdf(str(local_pdf_path), paper_id)
        print(f"  Uploaded local PDF -> {object_key}")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = os.path.join(tmp, f"{paper_id}.pdf")
            urlretrieve(result.pdf_url, tmp_path)
            object_key = storage.upload_pdf(tmp_path, paper_id)
        print(f"  Uploaded PDF -> {object_key}")

    # Info + object ID -> Qdrant
    paper = build_paper(result, object_key)
    chunks = store_embeddings(vector_client, paper)
    print(f"  Stored in vector store ({len(chunks)} chunk(s))")
    return paper


def _default_dependencies():
    """Real B2 + Qdrant clients. Imported here so tests can run without them."""
    from src.storage.b2_storage import B2Storage
    from src.vector_store import generate_and_store_embeddings, get_client

    if os.getenv("GITHUB_ACTIONS") and not os.getenv("QDRANT_URL"):
        raise RuntimeError(
            "QDRANT_URL is not set. In GitHub Actions a local Qdrant store is deleted "
            "when the job ends. Add QDRANT_URL / QDRANT_API_KEY as repository secrets."
        )

    return B2Storage(), get_client(path=str(VECTOR_STORE_PATH)), generate_and_store_embeddings


def scrape_new_papers(storage=None, vector_client=None, store_embeddings=None):
    if storage is None or vector_client is None or store_embeddings is None:
        storage, vector_client, store_embeddings = _default_dependencies()

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

            # "continue", not "break", so a paper that failed last time is retried
            if paper_id in seen_ids:
                continue

            print(f"New paper: {paper_id} - {result.title.strip()}")
            try:
                ingest_paper(result, storage, vector_client, store_embeddings)
            except Exception as e:
                print(f"  FAILED {paper_id}: {e}")
                failed.append(paper_id)
                continue

            seen_ids.add(paper_id)  # only after BOTH stores succeeded
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
