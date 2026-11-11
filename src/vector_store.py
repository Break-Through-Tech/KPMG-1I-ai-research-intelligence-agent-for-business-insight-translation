"""Qdrant storage.

Two collections in the same Qdrant instance:
- papers_high_level: one point per paper, a dense vector of its title + abstract.
  Payload: arxiv_id, title, authors, publish_date, categories, content (the abstract).
- paper_chunks: one point per chunk, with a dense and a sparse (BM25, IDF applied by Qdrant)
  vector. Payload: arxiv_id, title, authors, publish_date, categories, content_type, content
  (the plain chunk text), breadcrumb, chunk_index, chunk_id, record_id.

publish_date is an ISO date string ("2026-10-15") or None.

Point IDs are derived from arXiv IDs, so storing a paper again overwrites it instead of
duplicating it.
"""

import os
import uuid
import warnings

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    HasIdCondition,
    MatchValue,
    Modifier,
    PayloadSchemaType,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

from src.chunking import Chunk
from src.schema import Paper
from src.embeddings import (
    DENSE_VECTOR_SIZE,
    dense_chunk_text,
    embed_documents,
    embed_sparse,
    paper_text,
    sparse_chunk_text,
)

HIGH_LEVEL_COLLECTION = "papers_high_level"
CHUNK_COLLECTION = "paper_chunks"
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"


def get_client(path: str = "./qdrant_data") -> QdrantClient:
    """Use an online Qdrant if QDRANT_URL is set (needed for GitHub Actions),
    otherwise the local qdrant_data/ folder."""
    url = os.getenv("QDRANT_URL")
    if url:
        return QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"))
    return QdrantClient(path=path)


def point_id(record_id: str) -> str:
    """Same record always gets the same ID, so saving again overwrites instead of duplicating."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, record_id))


def chunk_record_id(arxiv_id: str, chunk_index: int) -> str:
    return f"{arxiv_id}_chunk_{chunk_index}"


def _paper_filter(arxiv_id: str) -> Filter:
    return Filter(must=[FieldCondition(key="arxiv_id", match=MatchValue(value=arxiv_id))])


def ensure_ingestion_collections(client: QdrantClient):
    """Create papers_high_level and paper_chunks if they don't exist yet."""
    existing = {c.name for c in client.get_collections().collections}
    dense = {DENSE_VECTOR: VectorParams(size=DENSE_VECTOR_SIZE, distance=Distance.COSINE)}

    if HIGH_LEVEL_COLLECTION not in existing:
        client.create_collection(collection_name=HIGH_LEVEL_COLLECTION, vectors_config=dense)
    if CHUNK_COLLECTION not in existing:
        client.create_collection(
            collection_name=CHUNK_COLLECTION,
            vectors_config=dense,
            sparse_vectors_config={SPARSE_VECTOR: SparseVectorParams(modifier=Modifier.IDF)},
        )

    for name in (HIGH_LEVEL_COLLECTION, CHUNK_COLLECTION):
        if name not in existing:
            # Qdrant Cloud needs this index to filter by paper; local Qdrant warns that it has no effect.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                client.create_payload_index(name, "arxiv_id", PayloadSchemaType.KEYWORD)


def paper_metadata(paper: Paper) -> dict:
    """The paper fields stored on its record and on each of its chunks."""
    return {
        "authors": paper.authors,
        "publish_date": paper.publish_date.isoformat() if paper.publish_date else None,
        "categories": paper.categories,
    }


def paper_is_stored(client: QdrantClient, arxiv_id: str) -> bool:
    """True if the paper has its high-level record and at least one chunk."""
    ensure_ingestion_collections(client)
    if not client.retrieve(HIGH_LEVEL_COLLECTION, [point_id(arxiv_id)], with_payload=False):
        return False
    return client.count(CHUNK_COLLECTION, count_filter=_paper_filter(arxiv_id), exact=True).count > 0


def store_paper_record(client: QdrantClient, paper: Paper):
    """Embed the paper's title + abstract and store it in papers_high_level."""
    ensure_ingestion_collections(client)
    [vector] = embed_documents([paper_text(paper.title, paper.abstract)])
    client.upsert(
        collection_name=HIGH_LEVEL_COLLECTION,
        points=[PointStruct(
            id=point_id(paper.paper_id),
            vector={DENSE_VECTOR: vector},
            payload={"arxiv_id": paper.paper_id, "title": paper.title, **paper_metadata(paper), "content": paper.abstract},
        )],
    )


def store_chunks(client: QdrantClient, paper: Paper, chunks: list[Chunk]):
    """Embed and store a paper's chunks in paper_chunks, then delete any of its chunks left
    over from an earlier version that had more of them."""
    ensure_ingestion_collections(client)
    dense = embed_documents([dense_chunk_text(c.breadcrumb, c.text) for c in chunks])
    sparse = embed_sparse([sparse_chunk_text(c.breadcrumb, c.text) for c in chunks])

    points = []
    for chunk, dense_vector, sparse_vector in zip(chunks, dense, sparse):
        record_id = chunk_record_id(paper.paper_id, chunk.chunk_index)
        points.append(PointStruct(
            id=point_id(record_id),
            vector={DENSE_VECTOR: dense_vector, SPARSE_VECTOR: sparse_vector},
            payload={
                "arxiv_id": paper.paper_id,
                "title": paper.title,
                **paper_metadata(paper),
                "content_type": "text",
                "content": chunk.text,
                "breadcrumb": chunk.breadcrumb,
                "chunk_index": chunk.chunk_index,
                "chunk_id": record_id,
                "record_id": record_id,
            },
        ))
    if points:
        client.upsert(collection_name=CHUNK_COLLECTION, points=points)

    leftovers = _paper_filter(paper.paper_id)
    if points:
        leftovers.must_not = [HasIdCondition(has_id=[p.id for p in points])]
    client.delete(collection_name=CHUNK_COLLECTION, points_selector=FilterSelector(filter=leftovers))


if __name__ == "__main__":
    # python -m src.vector_store <arxiv_id | path/to/page.html>
    # Stores one paper in a temporary in-memory Qdrant and prints what landed in each collection.
    # Title and abstract come from the HTML page here; ingestion takes them from arXiv's metadata.
    import json
    import sys
    from pathlib import Path

    from src.arxiv_html import fetch_html, parse_blocks, parse_title
    from src.chunking import chunk_blocks

    source = sys.argv[1]
    html = Path(source).read_text() if source.endswith(".html") else fetch_html(source)
    if html is None:
        sys.exit(f"arXiv has no HTML version of {source}")
    blocks = parse_blocks(html)
    paper = Paper(
        paper_id=Path(source).stem.split("_")[-1],
        title=parse_title(html) or source,
        abstract=" ".join(b.text for b in blocks if b.breadcrumb == ["Abstract"]) or "(no abstract)",
    )
    chunks = chunk_blocks(blocks)

    client = QdrantClient(":memory:")
    store_paper_record(client, paper)
    store_chunks(client, paper, chunks)

    def show(point):
        dense = point.vector[DENSE_VECTOR]
        sparse = point.vector.get(SPARSE_VECTOR)
        payload = {k: (v[:150] + "..." if isinstance(v, str) and len(v) > 150 else v)
                   for k, v in point.payload.items()}
        print(f"id {point.id}")
        print(f"vectors: dense {len(dense)} floats" + (f", sparse {len(sparse.indices)} terms" if sparse else ""))
        print(json.dumps(payload, indent=2, ensure_ascii=False), "\n")

    for name in (HIGH_LEVEL_COLLECTION, CHUNK_COLLECTION):
        points, _ = client.scroll(name, limit=1000, with_vectors=True)
        points.sort(key=lambda p: p.payload.get("chunk_index", 0))
        print(f"===== {name}: {len(points)} point(s) =====")
        for point in points[:2]:
            show(point)

    store_chunks(client, paper, chunks)
    print(f"Stored again: paper_chunks still has {client.count(CHUNK_COLLECTION).count} points")
