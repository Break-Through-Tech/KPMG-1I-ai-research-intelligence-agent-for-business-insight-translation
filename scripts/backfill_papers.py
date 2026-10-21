"""
One-off backfill: store every paper we've ALREADY scraped in the vector store,
section by section.

The old scraper only downloaded PDFs into data/ and listed their IDs in
data/seen_papers.json, so the normal scraper skips them. This script takes
every arXiv ID from seen_papers.json and from the PDF filenames in data/,
looks each one up on arXiv, and stores it the same way the scraper does.

Safe to re-run: re-storing a paper replaces its old chunks, never duplicates.

Run from the repo root:
    python -m scripts.backfill_papers
"""

import sys
from pathlib import Path

import arxiv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.scrape_papers import (  # noqa: E402
    DATA_DIR,
    _default_vector_client,
    get_paper_id,
    ingest_paper,
    load_seen_ids,
)

BATCH_SIZE = 50  # how many IDs to look up on arXiv per request


def all_known_ids():
    from_pdfs = {p.name.split("_")[0].removesuffix(".pdf") for p in DATA_DIR.glob("*.pdf")}
    return sorted(load_seen_ids() | from_pdfs)


def main():
    vector_client = _default_vector_client()
    ids = all_known_ids()
    print(f"Backfilling {len(ids)} paper(s)")

    client = arxiv.Client()
    done, failed = 0, []
    try:
        for i in range(0, len(ids), BATCH_SIZE):
            batch = ids[i : i + BATCH_SIZE]
            found = set()
            for result in client.results(arxiv.Search(id_list=batch)):
                paper_id = get_paper_id(result)
                found.add(paper_id)
                print(f"{paper_id} - {result.title.strip()}")
                try:
                    ingest_paper(result, vector_client)
                    done += 1
                except Exception as e:
                    print(f"  FAILED {paper_id}: {e}")
                    failed.append(paper_id)
            for missing in set(batch) - found:
                print(f"{missing}: not found on arXiv")
                failed.append(missing)
    finally:
        vector_client.close()

    print(f"Done. {done} paper(s) stored, {len(failed)} failed: {failed}")


if __name__ == "__main__":
    main()
