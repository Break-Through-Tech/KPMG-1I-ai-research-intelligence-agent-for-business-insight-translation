"""
Tests for scraping arXiv papers into the vector store (abstracts only).

Runs offline: arXiv search results and the embedding model are faked.
The vector-store tests use a real in-memory Qdrant.

Section parsing is tested separately in tests/test_arxiv_html.py.

Run from the repo root:
    python -m unittest tests.test_scrape_papers -v
"""

import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import scripts.scrape_papers as scrape_papers


class FakeAuthor:
    def __init__(self, name):
        self.name = name


class FakeResult:
    """Looks like an arxiv.Result."""

    def __init__(self, arxiv_id):
        self.entry_id = f"http://arxiv.org/abs/{arxiv_id}"
        self.pdf_url = f"http://arxiv.org/pdf/{arxiv_id}"
        self.title = f"Title of {arxiv_id}\n"
        self.authors = [FakeAuthor("Ada Lovelace"), FakeAuthor("Alan Turing")]
        self.summary = f"Abstract of\n{arxiv_id}."
        self.categories = ["cs.AI"]
        self.published = datetime(2026, 9, 30, tzinfo=timezone.utc)
        self.comment = None


class FakeVectorStore:
    """In-memory stand-in for Qdrant: {paper_id: Paper}"""

    def __init__(self):
        self.papers = {}
        self.fail_for = set()

    def store(self, client, paper):
        if paper.paper_id in self.fail_for:
            raise ConnectionError("simulated Qdrant outage")
        self.papers[paper.paper_id] = paper


# ---------- scraping / rescraping ----------

@patch("scripts.scrape_papers.arxiv.Client")
class TestScraper(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.data_dir = Path(self.tmp)
        patch.object(scrape_papers, "DATA_DIR", self.data_dir).start()
        patch.object(scrape_papers, "LOG_FILE", self.data_dir / "seen_papers.json").start()
        self.store = FakeVectorStore()

    def tearDown(self):
        patch.stopall()
        shutil.rmtree(self.tmp)

    def run_scraper(self, mock_client, ids):
        mock_client.return_value.results.return_value = [FakeResult(i) for i in ids]
        return scrape_papers.scrape_new_papers(vector_client=object(), store=self.store.store)

    def test_stores_abstract_and_metadata(self, mock_client):
        self.run_scraper(mock_client, ["2610.00001v1"])
        paper = self.store.papers["2610.00001v1"]
        self.assertEqual(paper.abstract, "Abstract of 2610.00001v1.")
        self.assertEqual(paper.title, "Title of 2610.00001v1")
        self.assertEqual(paper.authors, ["Ada Lovelace", "Alan Turing"])

    def test_rescrape_adds_only_new_papers(self, mock_client):
        self.run_scraper(mock_client, ["2610.00001v1", "2610.00002v1"])
        new_count, _ = self.run_scraper(mock_client, ["2610.00003v1", "2610.00001v1", "2610.00002v1"])
        self.assertEqual(new_count, 1)
        self.assertEqual(len(self.store.papers), 3)

    def test_same_results_twice_is_a_no_op(self, mock_client):
        self.run_scraper(mock_client, ["2610.00001v1", "2610.00002v1"])
        new_count, _ = self.run_scraper(mock_client, ["2610.00001v1", "2610.00002v1"])
        self.assertEqual(new_count, 0)

    def test_failed_paper_is_retried_next_run(self, mock_client):
        self.store.fail_for = {"2610.00002v1"}
        new_count, failed = self.run_scraper(mock_client, ["2610.00001v1", "2610.00002v1"])
        self.assertEqual((new_count, failed), (1, ["2610.00002v1"]))
        self.assertNotIn("2610.00002v1", scrape_papers.load_seen_ids())

        self.store.fail_for = set()
        new_count, failed = self.run_scraper(mock_client, ["2610.00001v1", "2610.00002v1"])
        self.assertEqual((new_count, failed), (1, []))
        self.assertEqual(scrape_papers.load_seen_ids(), {"2610.00001v1", "2610.00002v1"})

    def test_no_pdfs_written(self, mock_client):
        self.run_scraper(mock_client, ["2610.00001v1"])
        self.assertEqual(list(self.data_dir.glob("*.pdf")), [])

    def test_record_only_marks_seen_without_storing(self, mock_client):
        mock_client.return_value.results.return_value = [FakeResult("2610.00001v1")]
        with patch.object(scrape_papers, "_default_vector_client") as default_client:
            new_count, failed = scrape_papers.scrape_new_papers(record_only=True, store=self.store.store)
        default_client.assert_not_called()
        self.assertEqual((new_count, failed), (1, []))
        self.assertEqual(self.store.papers, {})
        self.assertEqual(scrape_papers.load_seen_ids(), {"2610.00001v1"})


# ---------- the real vector store (in-memory Qdrant) ----------

class TestVectorStore(unittest.TestCase):

    def setUp(self):
        from qdrant_client import QdrantClient
        from src import vector_store
        from src.schema import Paper

        self.vs = vector_store
        self.client = QdrantClient(":memory:")
        self.paper = Paper(paper_id="2610.00001v1", title="T", authors=["A"], abstract="Abs.")
        fake_embed = lambda texts: [[0.1] * vector_store.VECTOR_SIZE for _ in texts]
        patch("src.vector_store.embed_chunks", side_effect=fake_embed).start()

    def tearDown(self):
        patch.stopall()

    def all_points(self):
        points, _ = self.client.scroll(self.vs.COLLECTION_NAME, limit=1000, with_payload=True)
        return points

    def test_abstract_is_tagged_with_paper_id(self):
        self.vs.generate_and_store_embeddings(self.client, self.paper)
        (point,) = self.all_points()
        self.assertEqual(point.payload["paper_id"], "2610.00001v1")
        self.assertEqual(point.payload["chunk_text"], "Abs.")
        self.assertTrue(self.vs.paper_exists(self.client, "2610.00001v1"))

    def test_storing_a_paper_twice_does_not_duplicate(self):
        self.vs.generate_and_store_embeddings(self.client, self.paper)
        self.vs.generate_and_store_embeddings(self.client, self.paper)
        self.assertEqual(len(self.all_points()), 1)


if __name__ == "__main__":
    unittest.main()
