"""Search over papers_high_level / paper_chunks, against an in-memory Qdrant with fake query encoders."""

from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

from src import retrieval
from src.retrieval import ChunkHit, PaperHit, RetrievalError, search_chunks, search_papers
from src.vector_store import (
    CHUNK_COLLECTION,
    DENSE_VECTOR,
    HIGH_LEVEL_COLLECTION,
    SPARSE_VECTOR,
    ensure_ingestion_collections,
    point_id,
)

QUESTION_TERM = 7  # the one BM25 term every fake question has


def axis(i: int) -> list[float]:
    """A 768-dimension vector pointing along axis i, so cosine similarity is easy to predict."""
    vector = [0.0] * 768
    vector[i] = 1.0
    return vector


def blend(i: int, j: int, weight_j: float) -> list[float]:
    vector = axis(i)
    vector[j] = weight_j
    return vector


@pytest.fixture
def client():
    client = QdrantClient(":memory:")
    ensure_ingestion_collections(client)
    return client


@pytest.fixture
def question_vector(monkeypatch):
    """Every question embeds to axis 0; records the questions it was given."""
    asked = []

    def fake_embed_query(question):
        asked.append(question)
        return axis(0)

    monkeypatch.setattr(retrieval, "embed_query", fake_embed_query)
    monkeypatch.setattr(retrieval, "embed_sparse_query",
                        lambda question: SparseVector(indices=[QUESTION_TERM], values=[1.0]))
    return asked


def add_paper(client, arxiv_id, vector, **payload):
    payload = {"arxiv_id": arxiv_id, "title": f"Title {arxiv_id}", "content": f"Abstract {arxiv_id}", **payload}
    client.upsert(HIGH_LEVEL_COLLECTION, [PointStruct(id=point_id(arxiv_id), vector={DENSE_VECTOR: vector},
                                                      payload=payload)])


class TestSearchPapers:
    def test_ranks_papers_by_similarity_to_the_question(self, client, question_vector):
        add_paper(client, "far", axis(1))
        add_paper(client, "close", blend(0, 1, 0.2))
        add_paper(client, "middle", blend(0, 1, 1.0))

        hits = search_papers(client, "How do I route queries?")

        assert [h.arxiv_id for h in hits] == ["close", "middle", "far"]
        assert [h.rank for h in hits] == [1, 2, 3]
        assert hits[0].score > hits[1].score > hits[2].score
        assert question_vector == ["How do I route queries?"]

    def test_result_carries_the_stored_paper_fields(self, client, question_vector):
        add_paper(client, "2610.13767v1", axis(0), title="DualQuant", content="We introduce DualQuant.",
                  authors=["A. Author"], publish_date="2026-10-15", categories=["cs.LG"])
        [hit] = search_papers(client, "quantization")
        assert hit == PaperHit(arxiv_id="2610.13767v1", title="DualQuant", abstract="We introduce DualQuant.",
                               score=pytest.approx(1.0), rank=1, authors=["A. Author"], publish_date="2026-10-15",
                               categories=["cs.LG"])

    def test_top_k_limits_the_results(self, client, question_vector):
        for i in range(1, 15):
            add_paper(client, f"p{i}", blend(0, i, 0.1 * i))
        assert len(search_papers(client, "q")) == 10
        assert [h.arxiv_id for h in search_papers(client, "q", top_k=3)] == ["p1", "p2", "p3"]

    def test_missing_payload_fields_become_empty(self, client, question_vector):
        client.upsert(HIGH_LEVEL_COLLECTION, [PointStruct(id=point_id("x"), vector={DENSE_VECTOR: axis(0)},
                                                          payload={})])
        [hit] = search_papers(client, "q")
        assert (hit.arxiv_id, hit.title, hit.abstract) == ("", None, None)
        assert (hit.authors, hit.publish_date, hit.categories) == ([], None, [])

    def test_empty_collection_gives_no_results(self, client, question_vector):
        assert search_papers(client, "q") == []

    def test_blank_question_gives_no_results_without_embedding(self, client, question_vector):
        add_paper(client, "p", axis(0))
        assert search_papers(client, "   ") == []
        assert question_vector == []

    def test_missing_collection_raises(self, question_vector):
        with pytest.raises(RetrievalError, match="make ingest"):
            search_papers(QdrantClient(":memory:"), "q")

    def test_qdrant_failure_raises(self, question_vector):
        broken = MagicMock()
        broken.collection_exists.return_value = True
        broken.query_points.side_effect = ConnectionError("connection refused")
        with pytest.raises(RetrievalError, match="connection refused"):
            search_papers(broken, "q")

    def test_unreachable_qdrant_raises(self, question_vector):
        broken = MagicMock()
        broken.collection_exists.side_effect = ConnectionError("connection refused")
        with pytest.raises(RetrievalError, match="Couldn't reach Qdrant"):
            search_papers(broken, "q")


def add_chunk(client, name, dense, term_weight=None, **payload):
    """term_weight: how strongly the chunk contains the question's BM25 term (None: not at all)."""
    sparse = SparseVector(indices=[QUESTION_TERM], values=[term_weight]) if term_weight else \
        SparseVector(indices=[8], values=[1.0])
    payload = {"chunk_id": name, "arxiv_id": f"paper-{name}", "title": f"Title {name}", "content": f"Text {name}",
               "breadcrumb": ["2 Method"], "chunk_index": 0, "content_type": "text", **payload}
    client.upsert(CHUNK_COLLECTION, [PointStruct(id=point_id(name), vector={DENSE_VECTOR: dense, SPARSE_VECTOR: sparse},
                                                 payload=payload)])


@pytest.fixture
def chunks(client):
    """Dense order: dense_only, both, filler, sparse_only, neither. BM25 order: both, sparse_only."""
    add_chunk(client, "dense_only", axis(0))
    add_chunk(client, "both", blend(0, 1, 0.5), term_weight=2.0)
    add_chunk(client, "filler", blend(0, 1, 1.0))
    add_chunk(client, "sparse_only", blend(0, 1, 2.0), term_weight=1.0)
    add_chunk(client, "neither", axis(1))
    return client


class TestSearchChunks:
    def test_fuses_dense_and_sparse_rankings(self, chunks, question_vector):
        hits = search_chunks(chunks, "q")
        # sparse_only is 4th by meaning alone, but 2nd once its keyword match counts
        assert [h.chunk_id for h in hits] == ["both", "sparse_only", "dense_only", "filler", "neither"]
        assert [h.rank for h in hits] == [1, 2, 3, 4, 5]

    def test_score_is_the_rrf_score(self, chunks, question_vector):
        both, sparse_only, dense_only = search_chunks(chunks, "q")[:3]
        assert both.score == pytest.approx(1 / 61 + 1 / 60)         # 2nd dense, 1st sparse
        assert sparse_only.score == pytest.approx(1 / 63 + 1 / 61)  # 4th dense, 2nd sparse
        assert dense_only.score == pytest.approx(1 / 60)            # 1st dense only

    def test_rrf_k_is_configurable(self, chunks, question_vector):
        [top] = search_chunks(chunks, "q", top_k=1, rrf_k=2)
        assert top.score == pytest.approx(1 / 3 + 1 / 2)

    def test_searches_every_paper(self, chunks, question_vector):
        assert len({h.arxiv_id for h in search_chunks(chunks, "q")}) == 5

    def test_result_carries_the_stored_chunk_fields(self, client, question_vector):
        add_chunk(client, "2610.13767v1_chunk_3", axis(0), arxiv_id="2610.13767v1", title="DualQuant",
                  content="We quantize weights.", breadcrumb=["4 Method", "4.2 Scales"], chunk_index=3,
                  authors=["A. Author"], publish_date="2026-10-15", categories=["cs.LG"])
        [hit] = search_chunks(client, "q")
        assert hit == ChunkHit(chunk_id="2610.13767v1_chunk_3", arxiv_id="2610.13767v1", title="DualQuant",
                               content="We quantize weights.", breadcrumb=["4 Method", "4.2 Scales"],
                               chunk_index=3, content_type="text", score=pytest.approx(1 / 60), rank=1,
                               authors=["A. Author"], publish_date="2026-10-15", categories=["cs.LG"])

    def test_debug_fields_stay_empty_without_explain(self, chunks, question_vector):
        hit = search_chunks(chunks, "q")[0]
        assert (hit.dense_rank, hit.dense_score, hit.sparse_rank, hit.sparse_score) == (None, None, None, None)

    def test_explain_records_each_searchs_rank_and_score(self, chunks, question_vector):
        hits = {h.chunk_id: h for h in search_chunks(chunks, "q", explain=True)}
        assert (hits["both"].dense_rank, hits["both"].sparse_rank) == (2, 1)
        assert (hits["sparse_only"].dense_rank, hits["sparse_only"].sparse_rank) == (4, 2)
        assert (hits["dense_only"].dense_rank, hits["dense_only"].sparse_rank) == (1, None)
        assert hits["dense_only"].dense_score == pytest.approx(1.0)  # cosine similarity
        assert hits["dense_only"].sparse_score is None
        assert hits["both"].sparse_score > hits["sparse_only"].sparse_score > 0

    def test_explain_does_not_change_the_results(self, chunks, question_vector):
        plain = search_chunks(chunks, "q")
        explained = search_chunks(chunks, "q", explain=True)
        assert [(h.chunk_id, h.score) for h in plain] == [(h.chunk_id, h.score) for h in explained]

    def test_only_each_searchs_top_candidates_are_fused(self, chunks, question_vector):
        hits = search_chunks(chunks, "q", candidates=2)
        assert {h.chunk_id for h in hits} == {"dense_only", "both", "sparse_only"}

    def test_top_k_limits_the_results(self, chunks, question_vector):
        assert [h.chunk_id for h in search_chunks(chunks, "q", top_k=2)] == ["both", "sparse_only"]

    def test_question_without_bm25_terms_falls_back_to_dense(self, chunks, question_vector, monkeypatch):
        monkeypatch.setattr(retrieval, "embed_sparse_query", lambda q: SparseVector(indices=[], values=[]))
        hits = search_chunks(chunks, "the of and", explain=True)
        assert [h.chunk_id for h in hits] == ["dense_only", "both", "filler", "sparse_only", "neither"]
        assert all(h.sparse_rank is None for h in hits)

    def test_missing_payload_fields_become_empty(self, client, question_vector):
        client.upsert(CHUNK_COLLECTION, [PointStruct(id=point_id("x"), vector={DENSE_VECTOR: axis(0)}, payload={})])
        [hit] = search_chunks(client, "q")
        assert (hit.chunk_id, hit.arxiv_id, hit.content, hit.breadcrumb) == ("", "", "", [])
        assert (hit.title, hit.chunk_index, hit.content_type) == (None, None, None)
        assert (hit.authors, hit.publish_date, hit.categories) == ([], None, [])

    def test_empty_collection_gives_no_results(self, client, question_vector):
        assert search_chunks(client, "q") == []

    def test_blank_question_gives_no_results_without_embedding(self, chunks, question_vector):
        assert search_chunks(chunks, "") == []
        assert question_vector == []

    def test_missing_collection_raises(self, question_vector):
        with pytest.raises(RetrievalError, match="make ingest"):
            search_chunks(QdrantClient(":memory:"), "q")

    def test_qdrant_failure_raises(self, question_vector):
        broken = MagicMock()
        broken.collection_exists.return_value = True
        broken.query_points.side_effect = ConnectionError("connection refused")
        with pytest.raises(RetrievalError, match="connection refused"):
            search_chunks(broken, "q")
