"""
Store every paper we've ALREADY seen that isn't in the vector store yet.

The GitHub Action only records new IDs in data/seen_papers.json (--record-only),
so the scraper skips them afterwards. This script takes every arXiv ID from
seen_papers.json and from the PDF filenames in data/, skips the ones already
stored (a papers_high_level record plus at least one chunk), looks the rest up
on arXiv and stores them the same way the scraper does.

Papers without an HTML version are skipped and picked up by a later run.
Safe to re-run: re-storing a paper replaces its old chunks, never duplicates.

Run from the repo root:
    python -m scripts.backfill_papers            # only papers not stored yet
    python -m scripts.backfill_papers --force    # re-store every paper
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


def main(force=False):
    from src.vector_store import paper_is_stored

    vector_client = _default_vector_client()
    done, failed, no_html = 0, [], []
    try:
        known = all_known_ids()
        ids = known if force else [i for i in known if not paper_is_stored(vector_client, i)]
        print(f"Backfilling {len(ids)} paper(s); {len(known) - len(ids)} already stored")

        client = arxiv.Client()
        for i in range(0, len(ids), BATCH_SIZE):
            batch = ids[i : i + BATCH_SIZE]
            found = set()
            for result in client.results(arxiv.Search(id_list=batch)):
                paper_id = get_paper_id(result)
                found.add(paper_id)
                print(f"{paper_id} - {result.title.strip()}")
                try:
                    if ingest_paper(result, vector_client) is None:
                        no_html.append(paper_id)
                    else:
                        done += 1
                except Exception as e:
                    print(f"  FAILED {paper_id}: {e}")
                    failed.append(paper_id)
            for missing in set(batch) - found:
                print(f"{missing}: not found on arXiv")
                failed.append(missing)
    finally:
        vector_client.close()

    print(f"Done. {done} paper(s) stored, {len(no_html)} without HTML, {len(failed)} failed: {failed}")
    return done, failed, no_html


if __name__ == "__main__":
    main(force="--force" in sys.argv)
