"""Storage in papers_high_level / paper_chunks, against an in-memory Qdrant with fake encoders."""

import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Modifier, SparseVector

from src import vector_store
from src.chunking import Chunk
from src.schema import Paper
from src.vector_store import (
    CHUNK_COLLECTION,
    DENSE_VECTOR,
    HIGH_LEVEL_COLLECTION,
    SPARSE_VECTOR,
    ensure_ingestion_collections,
    paper_is_stored,
    point_id,
    store_chunks,
    store_paper_record,
)

PAPER = Paper(
    paper_id="2610.13767v1",
    title="DualQuant",
    abstract="We introduce DualQuant.",
    authors=["A. Author", "B. Author"],
    publish_date="2026-10-15",
    categories=["cs.LG", "cs.CL"],
)
METADATA = {"authors": ["A. Author", "B. Author"], "publish_date": "2026-10-15", "categories": ["cs.LG", "cs.CL"]}
OTHER = Paper(paper_id="2608.20316v1", title="Pandora's Router", abstract="We route models.")


@pytest.fixture
def client():
    return QdrantClient(":memory:")


@pytest.fixture
def encoded(monkeypatch):
    """Replaces both encoders and records the texts they were given."""
    texts = {"dense": [], "sparse": []}

    def fake_dense(batch):
        texts["dense"] += batch
        return [[1.0] * 768 for _ in batch]

    def fake_sparse(batch):
        texts["sparse"] += batch
        return [SparseVector(indices=[1, 2], values=[1.0, 0.5]) for _ in batch]

    monkeypatch.setattr(vector_store, "embed_documents", fake_dense)
    monkeypatch.setattr(vector_store, "embed_sparse", fake_sparse)
    return texts


def make_chunks(n: int) -> list[Chunk]:
    return [
        Chunk(text=f"Chunk {i} text.", breadcrumb=["4 Method", "4.2 Scales"], chunk_index=i,
              first_block=i, last_block=i, token_count=3)
        for i in range(n)
    ]


def chunk_payloads(client, arxiv_id=None) -> list[dict]:
    points, _ = client.scroll(CHUNK_COLLECTION, limit=1000)
    payloads = [p.payload for p in points if arxiv_id is None or p.payload["arxiv_id"] == arxiv_id]
    return sorted(payloads, key=lambda p: (p["arxiv_id"], p["chunk_index"]))


class TestCollections:
    def test_creates_both_collections_with_their_vectors(self, client):
        ensure_ingestion_collections(client)

        high = client.get_collection(HIGH_LEVEL_COLLECTION).config.params
        assert high.vectors[DENSE_VECTOR].size == 768
        assert high.vectors[DENSE_VECTOR].distance == Distance.COSINE
        assert not high.sparse_vectors

        chunks = client.get_collection(CHUNK_COLLECTION).config.params
        assert chunks.vectors[DENSE_VECTOR].size == 768
        assert chunks.sparse_vectors[SPARSE_VECTOR].modifier == Modifier.IDF

    def test_running_again_keeps_existing_data(self, client, encoded):
        store_paper_record(client, PAPER)
        ensure_ingestion_collections(client)
        assert client.count(HIGH_LEVEL_COLLECTION).count == 1


class TestPaperRecord:
    def test_stores_title_and_abstract_under_a_stable_id(self, client, encoded):
        store_paper_record(client, PAPER)

        [point] = client.retrieve(HIGH_LEVEL_COLLECTION, [point_id("2610.13767v1")], with_vectors=True)
        assert point.payload == {
            "arxiv_id": "2610.13767v1",
            "title": "DualQuant",
            **METADATA,
            "content": "We introduce DualQuant.",
        }
        assert len(point.vector[DENSE_VECTOR]) == 768
        assert encoded["dense"] == ["DualQuant\n\nWe introduce DualQuant."]
        assert point_id("2610.13767v1") == str(uuid.uuid5(uuid.NAMESPACE_DNS, "2610.13767v1"))

    def test_missing_metadata_is_stored_empty(self, client, encoded):
        store_paper_record(client, OTHER)

        [point] = client.retrieve(HIGH_LEVEL_COLLECTION, [point_id("2608.20316v1")])
        assert point.payload["authors"] == []
        assert point.payload["publish_date"] is None
        assert point.payload["categories"] == []

    def test_storing_again_overwrites(self, client, encoded):
        store_paper_record(client, PAPER)
        store_paper_record(client, PAPER)
        assert client.count(HIGH_LEVEL_COLLECTION).count == 1


class TestChunks:
    def test_payload_has_paper_and_chunk_fields(self, client, encoded):
        store_chunks(client, PAPER, make_chunks(2))

        assert chunk_payloads(client)[1] == {
            "arxiv_id": "2610.13767v1",
            "title": "DualQuant",
            **METADATA,
            "content_type": "text",
            "content": "Chunk 1 text.",
            "breadcrumb": ["4 Method", "4.2 Scales"],
            "chunk_index": 1,
            "chunk_id": "2610.13767v1_chunk_1",
            "record_id": "2610.13767v1_chunk_1",
        }

    def test_each_point_has_dense_and_sparse_vectors(self, client, encoded):
        store_chunks(client, PAPER, make_chunks(1))

        [point] = client.retrieve(CHUNK_COLLECTION, [point_id("2610.13767v1_chunk_0")], with_vectors=True)
        assert len(point.vector[DENSE_VECTOR]) == 768
        assert point.vector[SPARSE_VECTOR].indices == [1, 2]

    def test_chunk_inputs_have_section_but_no_title(self, client, encoded):
        store_chunks(client, PAPER, make_chunks(1))

        assert encoded["dense"] == ["Section: 4 Method > 4.2 Scales\n\nChunk 0 text."]
        assert encoded["sparse"] == ["4 Method > 4.2 Scales\n\nChunk 0 text."]

    def test_storing_again_overwrites(self, client, encoded):
        store_chunks(client, PAPER, make_chunks(3))
        store_chunks(client, PAPER, make_chunks(3))
        assert client.count(CHUNK_COLLECTION).count == 3

    def test_storing_fewer_chunks_removes_the_leftovers(self, client, encoded):
        store_chunks(client, PAPER, make_chunks(5))
        store_chunks(client, PAPER, make_chunks(2))
        assert [p["chunk_index"] for p in chunk_payloads(client)] == [0, 1]

    def test_other_papers_chunks_are_untouched(self, client, encoded):
        store_chunks(client, OTHER, make_chunks(2))
        store_chunks(client, PAPER, make_chunks(5))
        store_chunks(client, PAPER, make_chunks(1))

        assert len(chunk_payloads(client, "2608.20316v1")) == 2
        assert len(chunk_payloads(client, "2610.13767v1")) == 1

    def test_paper_counts_as_stored_only_with_record_and_chunks(self, client, encoded):
        assert not paper_is_stored(client, PAPER.paper_id)
        store_paper_record(client, PAPER)
        assert not paper_is_stored(client, PAPER.paper_id)  # record written, chunks not yet
        store_chunks(client, OTHER, make_chunks(1))
        assert not paper_is_stored(client, PAPER.paper_id)  # another paper's chunks don't count
        store_chunks(client, PAPER, make_chunks(1))
        assert paper_is_stored(client, PAPER.paper_id)

    def test_storing_no_chunks_removes_all_of_the_papers_chunks(self, client, encoded):
        store_chunks(client, OTHER, make_chunks(2))
        store_chunks(client, PAPER, make_chunks(3))
        store_chunks(client, PAPER, [])

        assert chunk_payloads(client, "2610.13767v1") == []
        assert len(chunk_payloads(client, "2608.20316v1")) == 2
