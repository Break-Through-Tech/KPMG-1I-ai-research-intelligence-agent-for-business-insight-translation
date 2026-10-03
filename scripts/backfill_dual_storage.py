"""
One-off backfill: move papers we scraped before issue #15 into dual storage.

The old scraper saved PDFs into data/ and committed them to GitHub. Those
papers are already in data/seen_papers.json, so the normal scraper skips them.
This script:
  - collects every paper ID from data/*.pdf filenames + data/seen_papers.json
  - fetches each paper's metadata (title, authors, abstract...) from arXiv
  - uploads the PDF to object storage (reusing the local file when we have one)
  - adds the paper + object ID to vector storage

Safe to re-run: existing PDFs in B2 are not re-uploaded and Qdrant point IDs
are deterministic, so nothing is duplicated.

Run from the repo root:
    python -m scripts.backfill_dual_storage
"""

import sys
from pathlib import Path

import arxiv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.scrape_papers import (  # noqa: E402
    DATA_DIR,
    _default_dependencies,
    get_paper_id,
    ingest_paper,
    load_seen_ids,
    save_seen_ids,
)

BATCH_SIZE = 50  # arXiv id_list lookups per request


def local_pdfs_by_id():
    """{'2608.20316v1': Path('data/2608.20316v1_Some Title.pdf'), ...}"""
    return {p.name.split("_")[0].removesuffix(".pdf"): p for p in DATA_DIR.glob("*.pdf")}


def main():
    storage, vector_client, store_embeddings = _default_dependencies()
    local = local_pdfs_by_id()
    seen_ids = load_seen_ids()
    all_ids = sorted(set(local) | seen_ids)
    print(f"Backfilling {len(all_ids)} paper(s) ({len(local)} with a local PDF)")

    client = arxiv.Client()
    done, failed = 0, []
    try:
        for i in range(0, len(all_ids), BATCH_SIZE):
            batch = all_ids[i : i + BATCH_SIZE]
            found = set()
            for result in client.results(arxiv.Search(id_list=batch)):
                paper_id = get_paper_id(result)
                found.add(paper_id)
                print(f"{paper_id} - {result.title.strip()}")
                try:
                    ingest_paper(
                        result, storage, vector_client, store_embeddings,
                        local_pdf_path=local.get(paper_id),
                    )
                    seen_ids.add(paper_id)
                    done += 1
                except Exception as e:
                    print(f"  FAILED {paper_id}: {e}")
                    failed.append(paper_id)
            for missing in set(batch) - found:
                print(f"{missing}: not found on arXiv")
                failed.append(missing)
    finally:
        save_seen_ids(seen_ids)
        vector_client.close()

    print(f"Done. {done} paper(s) in dual storage, {len(failed)} failed: {failed}")
    if not failed:
        print("All good - the PDFs in data/ can now be deleted (git rm data/*.pdf).")


if __name__ == "__main__":
    main()
