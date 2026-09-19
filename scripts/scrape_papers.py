import arxiv
import os
import json
from urllib.request import urlretrieve

DATA_DIR = "../data"
LOG_FILE = "../data/seen_papers.json"
QUERY = "LLM routing"
MAX_RESULTS = 20


def load_seen_ids():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            return set(json.load(f))
    return set()


def save_seen_ids(seen_ids):
    with open(LOG_FILE, "w") as f:
        json.dump(list(seen_ids), f)


def scrape_new_papers():
    os.makedirs(DATA_DIR, exist_ok=True)
    seen_ids = load_seen_ids()

    client = arxiv.Client()
    search = arxiv.Search(
        query=QUERY,
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    new_count = 0
    for paper in client.results(search):
        paper_id = paper.entry_id.split("/")[-1]

        if paper_id in seen_ids:
            break  # results are newest-first, so anything after this is old too

        filename = f"{paper_id}.pdf"
        urlretrieve(paper.pdf_url, os.path.join(DATA_DIR, filename))
        seen_ids.add(paper_id)
        new_count += 1
        print(f"Downloaded: {filename}")

    save_seen_ids(seen_ids)
    print(f"Done. {new_count} new paper(s) added.")


if __name__ == "__main__":
    scrape_new_papers()


    