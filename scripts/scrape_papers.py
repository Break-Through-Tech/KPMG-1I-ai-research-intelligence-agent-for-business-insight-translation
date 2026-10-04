import arxiv
import os
from urllib.request import urlretrieve
from src.storage.b2_storage import B2Storage

DATA_DIR = "../data"
QUERY = "LLM routing"
MAX_RESULTS = 20

def scrape_new_papers():
    os.makedirs(DATA_DIR, exist_ok=True)
    storage = B2Storage()

    client = arxiv.Client()
    search = arxiv.Search(
        query=QUERY,
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    new_count = 0
    for paper in client.results(search):
        paper_id = paper.entry_id.split("/")[-1]

        if storage.pdf_exists(paper_id):
            print(f"Already stored: {paper_id}")
            continue

        filename = f"{paper_id}.pdf"
        local_path = os.path.join(DATA_DIR, filename)
        
        urlretrieve(paper.pdf_url, local_path)
        storage.upload_pdf(local_path, paper_id)
        
        new_count += 1
        print(f"Downloaded and uploaded: {filename}")

    print(f"Done. {new_count} new paper(s) added.")


if __name__ == "__main__":
    scrape_new_papers()


    