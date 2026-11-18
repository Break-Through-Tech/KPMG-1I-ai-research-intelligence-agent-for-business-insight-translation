"""Search the two collections written by ingestion.

- search_papers(): dense search over papers_high_level (title + abstract), top 10 papers by default.
- search_chunks(): hybrid search over all of paper_chunks, top 20 chunks by default. The dense and
  sparse (BM25) searches each pick their top candidates and Qdrant merges the two lists with
  Reciprocal Rank Fusion. explain=True also records each chunk's rank and score in each list.

Each search takes a QdrantClient (src.vector_store.get_client()) and a research question. A blank
question or no matches gives an empty list; a missing collection or a Qdrant failure raises
RetrievalError, so callers can tell an outage apart from "nothing relevant".
"""

from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, Rrf, RrfQuery, ScoredPoint

from src.embeddings import embed_query, embed_sparse_query
from src.vector_store import CHUNK_COLLECTION, DENSE_VECTOR, HIGH_LEVEL_COLLECTION, SPARSE_VECTOR

PAPER_TOP_K = 10
CHUNK_TOP_K = 20
HYBRID_CANDIDATES = 100  # chunks each of the dense and sparse searches contributes to fusion
RRF_K = 60               # RRF score = sum of 1 / (RRF_K + position, counted from 0) over the lists a chunk is in


class RetrievalError(RuntimeError):
    pass


@dataclass
class PaperHit:
    arxiv_id: str
    title: str | None
    abstract: str | None
    score: float  # cosine similarity between the question and the paper's title + abstract
    rank: int     # 1-based
    authors: list[str] = field(default_factory=list)
    publish_date: str | None = None  # ISO date, e.g. "2026-10-15"
    categories: list[str] = field(default_factory=list)


@dataclass
class ChunkHit:
    chunk_id: str
    arxiv_id: str
    title: str | None
    content: str
    breadcrumb: list[str]
    chunk_index: int | None  # position within the paper, i.e. document order
    content_type: str | None
    score: float  # RRF score; only the order matters, not the size
    rank: int     # 1-based position in the fused results
    authors: list[str] = field(default_factory=list)
    publish_date: str | None = None  # ISO date, e.g. "2026-10-15"
    categories: list[str] = field(default_factory=list)
    # Filled only with explain=True. A rank is None if that search didn't have the chunk among
    # its HYBRID_CANDIDATES; dense scores are cosine similarities, sparse scores BM25 scores.
    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None


def _check_collection(client: QdrantClient, collection: str):
    try:
        exists = client.collection_exists(collection)
    except Exception as e:
        raise RetrievalError(f"Couldn't reach Qdrant: {e}") from e
    if not exists:
        raise RetrievalError(f"Collection '{collection}' doesn't exist; run `make ingest` first")


def search_papers(
    client: QdrantClient,
    question: str,
    top_k: int = PAPER_TOP_K,
    collection: str = HIGH_LEVEL_COLLECTION,
) -> list[PaperHit]:
    """The top_k papers whose title + abstract are closest in meaning to the question."""
    if not question.strip():
        return []
    _check_collection(client, collection)
    try:
        points = client.query_points(
            collection, query=embed_query(question), using=DENSE_VECTOR, limit=top_k, with_payload=True,
        ).points
    except Exception as e:
        raise RetrievalError(f"Paper search failed: {e}") from e

    return [
        PaperHit(
            arxiv_id=p.payload.get("arxiv_id", ""),
            title=p.payload.get("title"),
            abstract=p.payload.get("content"),
            score=p.score,
            rank=rank,
            authors=p.payload.get("authors", []),
            publish_date=p.payload.get("publish_date"),
            categories=p.payload.get("categories", []),
        )
        for rank, p in enumerate(points, start=1)
    ]


def _ranks(points: list[ScoredPoint]) -> dict:
    """Point ID -> (1-based rank, score)."""
    return {p.id: (rank, p.score) for rank, p in enumerate(points, start=1)}


def search_chunks(
    client: QdrantClient,
    question: str,
    top_k: int = CHUNK_TOP_K,
    collection: str = CHUNK_COLLECTION,
    candidates: int = HYBRID_CANDIDATES,
    rrf_k: int = RRF_K,
    explain: bool = False,
) -> list[ChunkHit]:
    """The top_k chunks from the whole collection by dense + sparse hybrid search."""
    if not question.strip():
        return []
    _check_collection(client, collection)

    dense = embed_query(question)
    sparse = embed_sparse_query(question)
    searches = [(dense, DENSE_VECTOR)]
    if sparse.indices:  # a question of only stopwords has no BM25 terms
        searches.append((sparse, SPARSE_VECTOR))

    try:
        points = client.query_points(
            collection,
            prefetch=[Prefetch(query=query, using=using, limit=candidates) for query, using in searches],
            query=RrfQuery(rrf=Rrf(k=rrf_k)),
            limit=top_k,
            with_payload=True,
        ).points
        ranks = {}
        if explain:
            for query, using in searches:
                ranks[using] = _ranks(client.query_points(collection, query=query, using=using, limit=candidates).points)
    except Exception as e:
        raise RetrievalError(f"Chunk search failed: {e}") from e

    hits = []
    for rank, p in enumerate(points, start=1):
        hit = ChunkHit(
            chunk_id=p.payload.get("chunk_id", ""),
            arxiv_id=p.payload.get("arxiv_id", ""),
            title=p.payload.get("title"),
            content=p.payload.get("content", ""),
            breadcrumb=p.payload.get("breadcrumb", []),
            chunk_index=p.payload.get("chunk_index"),
            content_type=p.payload.get("content_type"),
            score=p.score,
            rank=rank,
            authors=p.payload.get("authors", []),
            publish_date=p.payload.get("publish_date"),
            categories=p.payload.get("categories", []),
        )
        if explain:
            hit.dense_rank, hit.dense_score = ranks[DENSE_VECTOR].get(p.id, (None, None))
            hit.sparse_rank, hit.sparse_score = ranks.get(SPARSE_VECTOR, {}).get(p.id, (None, None))
        hits.append(hit)
    return hits


def _print_results(client: QdrantClient, question: str, papers: int, chunks: int, explain: bool):
    if papers:
        print(f"\n===== Top {papers} papers (dense, papers_high_level) =====")
        for h in search_papers(client, question, top_k=papers):
            print(f"{h.rank:3}. {h.score:.3f}  {h.arxiv_id}  {h.title}")
    if chunks:
        print(f"\n===== Top {chunks} chunks (hybrid, paper_chunks) =====")
        for h in search_chunks(client, question, top_k=chunks, explain=explain):
            line = f"{h.rank:3}. RRF {h.score:.4f}"
            if explain:
                dense = f"#{h.dense_rank} ({h.dense_score:.3f})" if h.dense_rank else "-"
                sparse = f"#{h.sparse_rank} ({h.sparse_score:.1f})" if h.sparse_rank else "-"
                line += f"  dense {dense:<14} sparse {sparse:<13}"
            print(f"{line}  {h.chunk_id}  {(h.title or '')[:70]}")
            print(f"      {' > '.join(h.breadcrumb)}")
            print(f"      {' '.join(h.content.split())[:300]}\n")


if __name__ == "__main__":
    # python -m src.retrieval ["question"] [--explain] [--papers N] [--chunks N]
    # Without a question, asks for one question after another (models and Qdrant load only once).
    import argparse

    from src.vector_store import get_client

    parser = argparse.ArgumentParser(description="Search papers_high_level and paper_chunks.")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--explain", action="store_true", help="show each chunk's dense and sparse rank and score")
    parser.add_argument("--papers", type=int, default=PAPER_TOP_K, help="papers to show (0 skips paper search)")
    parser.add_argument("--chunks", type=int, default=CHUNK_TOP_K, help="chunks to show (0 skips chunk search)")
    args = parser.parse_args()

    client = get_client()
    try:
        if args.question:
            _print_results(client, args.question, args.papers, args.chunks, args.explain)
        else:
            while question := input("\nResearch question (blank to quit): ").strip():
                _print_results(client, question, args.papers, args.chunks, args.explain)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        client.close()
