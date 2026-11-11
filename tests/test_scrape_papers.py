"""
Tests for scraping and backfilling arXiv papers into the vector store.

Runs offline: arXiv search results, HTML pages and the encoders are faked.
The end-to-end tests use a real in-memory Qdrant.

Parsing, chunking, encoders and storage are tested on their own in
tests/test_arxiv_blocks.py, test_chunking.py, test_embeddings.py and test_vector_store.py.

Run from the repo root:
    python -m pytest tests/test_scrape_papers.py -v
"""

import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import scripts.backfill_papers as backfill_papers
import scripts.scrape_papers as scrape_papers
from src.arxiv_html import TextBlock

FIXTURES = Path(__file__).parent / "fixtures" / "arxiv_html"


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
    """Stands in for store_paper_record / store_chunks: {paper_id: Paper} and {paper_id: chunks}."""

    def __init__(self):
        self.records = {}
        self.chunks = {}
        self.fail_for = set()  # chunk writes fail for these, after the record was written

    def store_paper_record(self, client, paper):
        self.records[paper.paper_id] = paper

    def store_chunks(self, client, paper, chunks):
        if paper.paper_id in self.fail_for:
            raise ConnectionError("simulated Qdrant outage")
        self.chunks[paper.paper_id] = chunks


def words(text):
    return len(text.split())


# ---------- scraping / rescraping ----------

@patch("scripts.scrape_papers.arxiv.Client")
class TestScraper(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.data_dir = Path(self.tmp)
        patch.object(scrape_papers, "DATA_DIR", self.data_dir).start()
        patch.object(scrape_papers, "LOG_FILE", self.data_dir / "seen_papers.json").start()

        self.store = FakeVectorStore()
        patch("src.vector_store.store_paper_record", self.store.store_paper_record).start()
        patch("src.vector_store.store_chunks", self.store.store_chunks).start()

        self.no_html = set()
        patch("src.arxiv_html.fetch_blocks", side_effect=self.fake_fetch_blocks).start()
        patch("src.embeddings.count_tokens", side_effect=words).start()

    def tearDown(self):
        patch.stopall()
        shutil.rmtree(self.tmp)

    def fake_fetch_blocks(self, arxiv_id):
        if arxiv_id in self.no_html:
            return None
        return [TextBlock(f"Body of {arxiv_id}.", ["1 Intro"], 0, ("1 Intro",))]

    def run_scraper(self, mock_client, ids):
        mock_client.return_value.results.return_value = [FakeResult(i) for i in ids]
        return scrape_papers.scrape_new_papers(vector_client=object())

    def test_stores_paper_record_and_chunks(self, mock_client):
        self.run_scraper(mock_client, ["2610.00001v1"])
        paper = self.store.records["2610.00001v1"]
        self.assertEqual(paper.abstract, "Abstract of 2610.00001v1.")
        self.assertEqual(paper.title, "Title of 2610.00001v1")
        self.assertEqual(paper.authors, ["Ada Lovelace", "Alan Turing"])
        [chunk] = self.store.chunks["2610.00001v1"]
        self.assertEqual((chunk.text, chunk.breadcrumb), ("Body of 2610.00001v1.", ["1 Intro"]))

    def test_rescrape_adds_only_new_papers(self, mock_client):
        self.run_scraper(mock_client, ["2610.00001v1", "2610.00002v1"])
        new_count, _ = self.run_scraper(mock_client, ["2610.00003v1", "2610.00001v1", "2610.00002v1"])
        self.assertEqual(new_count, 1)
        self.assertEqual(len(self.store.chunks), 3)

    def test_same_results_twice_is_a_no_op(self, mock_client):
        self.run_scraper(mock_client, ["2610.00001v1", "2610.00002v1"])
        new_count, _ = self.run_scraper(mock_client, ["2610.00001v1", "2610.00002v1"])
        self.assertEqual(new_count, 0)

    def test_failed_paper_is_retried_next_run(self, mock_client):
        self.store.fail_for = {"2610.00002v1"}
        new_count, failed = self.run_scraper(mock_client, ["2610.00001v1", "2610.00002v1"])
        self.assertEqual((new_count, failed), (1, ["2610.00002v1"]))
        # The high-level record was written before the chunk write failed; it's still not "seen".
        self.assertIn("2610.00002v1", self.store.records)
        self.assertNotIn("2610.00002v1", scrape_papers.load_seen_ids())

        self.store.fail_for = set()
        new_count, failed = self.run_scraper(mock_client, ["2610.00001v1", "2610.00002v1"])
        self.assertEqual((new_count, failed), (1, []))
        self.assertEqual(scrape_papers.load_seen_ids(), {"2610.00001v1", "2610.00002v1"})

    def test_paper_without_html_is_skipped_and_retried(self, mock_client):
        self.no_html = {"2610.00002v1"}
        new_count, failed = self.run_scraper(mock_client, ["2610.00001v1", "2610.00002v1"])
        self.assertEqual((new_count, failed), (1, []))  # skipped, not failed
        self.assertNotIn("2610.00002v1", self.store.records)  # nothing stored at all
        self.assertNotIn("2610.00002v1", scrape_papers.load_seen_ids())

        self.no_html = set()
        new_count, _ = self.run_scraper(mock_client, ["2610.00001v1", "2610.00002v1"])
        self.assertEqual(new_count, 1)
        self.assertIn("2610.00002v1", self.store.chunks)

    def test_no_pdfs_written(self, mock_client):
        self.run_scraper(mock_client, ["2610.00001v1"])
        self.assertEqual(list(self.data_dir.glob("*.pdf")), [])

    def test_record_only_marks_seen_without_storing(self, mock_client):
        mock_client.return_value.results.return_value = [FakeResult("2610.00001v1")]
        with patch.object(scrape_papers, "_default_vector_client") as default_client:
            new_count, failed = scrape_papers.scrape_new_papers(record_only=True)
        default_client.assert_not_called()
        self.assertEqual((new_count, failed), (1, []))
        self.assertEqual(self.store.records, {})
        self.assertEqual(scrape_papers.load_seen_ids(), {"2610.00001v1"})


# ---------- one real paper through parse, chunk and store ----------

class TestIngestEndToEnd(unittest.TestCase):
    """Real parser, chunker and in-memory Qdrant; only the HTML download and encoders are faked."""

    def setUp(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import SparseVector

        self.html = (FIXTURES / "dualquant_2610.13767v1.html").read_text()
        patch("src.arxiv_html.fetch_html", return_value=self.html).start()
        patch("src.embeddings.count_tokens", side_effect=words).start()
        patch("src.vector_store.embed_documents", side_effect=lambda t: [[1.0] * 768 for _ in t]).start()
        patch("src.vector_store.embed_sparse",
              side_effect=lambda t: [SparseVector(indices=[1], values=[1.0]) for _ in t]).start()
        self.client = QdrantClient(":memory:")

    def tearDown(self):
        patch.stopall()

    def test_paper_is_fully_stored(self):
        from src import vector_store
        from src.arxiv_html import parse_blocks
        from src.chunking import chunk_blocks

        paper = scrape_papers.ingest_paper(FakeResult("2610.13767v1"), self.client)

        [record] = self.client.retrieve(vector_store.HIGH_LEVEL_COLLECTION, [vector_store.point_id(paper.paper_id)])
        self.assertEqual(record.payload["title"], "Title of 2610.13767v1")
        self.assertEqual(record.payload["content"], "Abstract of 2610.13767v1.")

        points, _ = self.client.scroll(vector_store.CHUNK_COLLECTION, limit=1000)
        expected = chunk_blocks(parse_blocks(self.html), count_tokens=words)
        self.assertEqual(sorted(p.payload["content"] for p in points), sorted(c.text for c in expected))
        self.assertTrue(all(p.payload["arxiv_id"] == "2610.13767v1" for p in points))
        self.assertTrue(vector_store.paper_is_stored(self.client, "2610.13767v1"))


# ---------- backfill ----------

class TestBackfill(unittest.TestCase):

    def setUp(self):
        self.known = ["2610.00001v1", "2610.00002v1", "2610.00003v1"]
        self.stored = {"2610.00001v1"}
        self.no_html = set()
        self.ingested = []

        patch.object(backfill_papers, "corpus_ids", lambda: self.known).start()
        patch.object(backfill_papers, "all_known_ids", lambda: self.known + ["2611.00009v1"]).start()
        patch.object(backfill_papers, "_default_vector_client", MagicMock).start()
        patch.object(backfill_papers, "ingest_paper", self.fake_ingest).start()
        patch("src.vector_store.paper_is_stored", lambda client, i: i in self.stored).start()
        arxiv_client = patch("scripts.backfill_papers.arxiv.Client").start()
        arxiv_client.return_value.results.side_effect = lambda search: [FakeResult(i) for i in search.id_list]

    def tearDown(self):
        patch.stopall()

    def fake_ingest(self, result, vector_client):
        paper_id = scrape_papers.get_paper_id(result)
        if paper_id in self.no_html:
            return None
        self.ingested.append(paper_id)
        return object()

    def test_skips_papers_already_stored(self):
        done, failed, _ = backfill_papers.main()
        self.assertEqual(self.ingested, ["2610.00002v1", "2610.00003v1"])
        self.assertEqual((done, failed), (2, []))

    def test_force_restores_every_paper(self):
        backfill_papers.main(force=True)
        self.assertEqual(self.ingested, self.known)

    def test_papers_without_html_are_reported_separately(self):
        self.no_html = {"2610.00003v1"}
        done, failed, no_html = backfill_papers.main()
        self.assertEqual((done, failed, no_html), (1, [], ["2610.00003v1"]))

    def test_all_uses_every_seen_paper_instead_of_the_corpus(self):
        backfill_papers.main(use_all=True)
        self.assertEqual(self.ingested, ["2610.00002v1", "2610.00003v1", "2611.00009v1"])


class TestCorpusFile(unittest.TestCase):

    def test_reads_ids_and_ignores_comments_and_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus_ids.txt"
            path.write_text("# eval papers\n2610.00002v1\n\n  2610.00001v1  \n# others\n2610.00001v1\n")
            with patch.object(backfill_papers, "CORPUS_FILE", path):
                self.assertEqual(backfill_papers.corpus_ids(), ["2610.00001v1", "2610.00002v1"])

    def test_committed_corpus_has_100_papers(self):
        self.assertEqual(len(backfill_papers.corpus_ids()), 100)


if __name__ == "__main__":
    unittest.main()
