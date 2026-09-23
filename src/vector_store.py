from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from src.schema import Paper, Chunk
from src.embeddings import get_model, embed_chunks, EMBEDDING_MODEL_NAME

COLLECTION_NAME = "papers"
VECTOR_SIZE = 384  

def get_client(path: str = "./qdrant_data") -> QdrantClient:
    return QdrantClient(path=path)

def ensure_collection(client: QdrantClient):
    if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

def generate_and_store_embeddings(client: QdrantClient, paper: Paper) -> list[Chunk]:
    """Chunk a paper's abstract (or full text), embed each chunk, and upsert into Qdrant.

    Returns the list of Chunk objects that were stored, so the caller can
    inspect or log them.
    """
    ensure_collection(client)

    text = paper.abstract  
    raw_chunks = [paper.abstract]
    vectors = embed_chunks(raw_chunks)

    chunks: list[Chunk] = []
    points: list[PointStruct] = []

    for i, (raw_text, vector) in enumerate(zip(raw_chunks, vectors)):
        chunk = Chunk(
            chunk_id=f"{paper.paper_id}_chunk_{i}",
            paper_id=paper.paper_id,
            chunk_text=raw_text,
            chunk_index=i,
            embedding_model=EMBEDDING_MODEL_NAME,
        )
        chunks.append(chunk)

        points.append(
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)),
                vector=vector,
                payload={
                    **chunk.model_dump(mode="json"),
                    "title": paper.title,
                    "authors": paper.authors,
                    "categories": paper.categories,
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return chunks

def semantic_search(client: QdrantClient, query: str, top_k: int = 5) -> list[dict]:
    model = get_model()
    query_vector = model.encode(query).tolist()
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
    )
    return [
        {
            "score": r.score,
            "paper_id": r.payload["paper_id"],
            "title": r.payload["title"],
            "chunk_text": r.payload["chunk_text"],
            "chunk_index": r.payload["chunk_index"],
        }
        for r in results
    ]
def search_papers(client: QdrantClient, query: str, top_k_papers: int = 2, chunk_pool: int = 20) -> list[dict]:
    """Semantic search that returns paper-level results.

    Searches across all chunks, groups hits by paper_id, and returns the
    top_k_papers papers ranked by their single best-matching chunk.

    chunk_pool controls how many raw chunk hits we pull before grouping —
    needs to be bigger than top_k_papers so papers with a slightly-lower
    numbered chunk don't get shut out by chunks from other papers ranking above them.
    """
    model = get_model()
    query_vector = model.encode(query).tolist()

    raw_hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=chunk_pool,
    )

    best_per_paper: dict[str, dict] = {}
    for hit in raw_hits:
        pid = hit.payload["paper_id"]
        if pid not in best_per_paper or hit.score > best_per_paper[pid]["score"]:
            best_per_paper[pid] = {
                "score": hit.score,
                "paper_id": pid,
                "title": hit.payload["title"],
                "authors": hit.payload.get("authors", []),
                "categories": hit.payload.get("categories", []),
                "best_chunk_text": hit.payload["chunk_text"],
                "chunk_index": hit.payload["chunk_index"],
            }

    ranked_papers = sorted(best_per_paper.values(), key=lambda p: p["score"], reverse=True)
    return ranked_papers[:top_k_papers]