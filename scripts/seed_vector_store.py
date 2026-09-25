'''
Adds 5 research papers to vector store from 
data/sample_papers.json
'''

import json
from pathlib import Path

from src.schema import Paper
from src.vector_store import get_client, generate_and_store_embeddings

# important file paths
ROOT_DIR = Path(__file__).resolve().parents[1]
PAPERS_FILE = ROOT_DIR / "data" / "sample_papers.json"
VECTOR_STORE_PATH = ROOT_DIR / "qdrant_data"

# open json, read json, add json to qdrant vector store
def main():
    with PAPERS_FILE.open() as file:
        paper_records = json.load(file)

    papers = [Paper.model_validate(record) for record in paper_records]
    client = get_client(path=str(VECTOR_STORE_PATH))

    total_chunks = 0

    try:
        for paper in papers:
            chunks = generate_and_store_embeddings(client, paper)
            total_chunks += len(chunks)
            print(f"Stored {paper.paper_id}: {len(chunks)} chunk(s)")
    finally:
        client.close()

    print(f"Finished: stored {len(papers)} papers and {total_chunks} chunks")

if __name__ == "__main__":
    main()