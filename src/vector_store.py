import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from src.schema import Paper, Chunk
from src.embeddings import get_model, embed_chunks, EMBEDDING_MODEL_NAME

COLLECTION_NAME = "papers"
VECTOR_SIZE = 384


def get_client(path: str = "./qdrant_data") -> QdrantClient:
    """Use an online Qdrant if QDRANT_URL is set (needed for GitHub Actions),
    otherwise the local qdrant_data/ folder."""
    url = os.getenv("QDRANT_URL")
    if url:
        return QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"))
    return QdrantClient(path=path)


def ensure_collection(client: QdrantClient):
    if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def chunk_point_id(chunk_id: str) -> str:
    """Same chunk always gets the same ID, so saving again overwrites instead of duplicating."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def paper_exists(client: QdrantClient, paper_id: str) -> bool:
    """True if this paper already has its first chunk (the abstract) in the vector store."""
    ensure_collection(client)
    found = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[chunk_point_id(f"{paper_id}_chunk_0")],
        with_payload=False,
        with_vectors=False,
    )
    return len(found) > 0


def delete_paper(client: QdrantClient, paper_id: str):
    """Remove every chunk of a paper (used before re-storing it, so no old chunks are left behind)."""
    ensure_collection(client)
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
        ),
    )


def store_paper_chunks(
    client: QdrantClient, paper: Paper, sections: list[tuple[str, str]]
) -> list[Chunk]:
    """Embed and store a paper as one chunk per (section_title, text) pair.

    Every chunk carries the paper's arXiv ID (paper_id) plus its title/authors,
    so a search hit on any section can be traced back to its paper.
    """
    ensure_collection(client)

    texts = [text for _, text in sections]
    vectors = embed_chunks(texts)

    chunks: list[Chunk] = []
    points: list[PointStruct] = []

    for i, ((section_title, text), vector) in enumerate(zip(sections, vectors)):
        chunk = Chunk(
            chunk_id=f"{paper.paper_id}_chunk_{i}",
            paper_id=paper.paper_id,
            chunk_text=text,
            chunk_index=i,
            section_title=section_title,
            embedding_model=EMBEDDING_MODEL_NAME,
        )
        chunks.append(chunk)

        points.append(
            PointStruct(
                id=chunk_point_id(chunk.chunk_id),
                vector=vector,
                payload={
                    **chunk.model_dump(mode="json"),
                    "title": paper.title,
                    "authors": paper.authors,
                    "categories": paper.categories,
                    "publish_date": paper.publish_date.isoformat() if paper.publish_date else None,
                    "pdf_url": str(paper.pdf_url) if paper.pdf_url else None,
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return chunks


def generate_and_store_embeddings(client: QdrantClient, paper: Paper) -> list[Chunk]:
    """Store just the paper's abstract as a single chunk."""
    return store_paper_chunks(client, paper, [("Abstract", paper.abstract)])


def semantic_search(client: QdrantClient, query: str, top_k: int = 5) -> list[dict]:
    model = get_model()
    query_vector = model.encode(query).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points
    return [
        {
            "score": r.score,
            "paper_id": r.payload["paper_id"],
            "title": r.payload["title"],
            "section_title": r.payload.get("section_title"),
            "chunk_text": r.payload["chunk_text"],
            "chunk_index": r.payload["chunk_index"],
        }
        for r in results
    ]
