"""
Tests for issue #15: scraping/rescraping -> object storage -> vector storage.

Uses a real in-memory Qdrant and a fake in-memory object store, and patches
out arXiv, the PDF download and the embedding model, so they run offline with
no B2 credentials.

Run from the repo root:
    python -m unittest tests.test_scrape_papers -v
"""

import os
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from qdrant_client import QdrantClient

import scripts.scrape_papers as scrape_papers
from src import vector_store
from src.vector_store import COLLECTION_NAME, VECTOR_SIZE, generate_and_store_embeddings


# ---------- fakes ----------

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
        self.categories = ["cs.AI", "cs.CL"]
        self.published = datetime(2026, 9, 30, tzinfo=timezone.utc)
        self.comment = None


class FakeObjectStorage:
    """In-memory stand-in for B2Storage with the same interface."""

    def __init__(self):
        self.objects = {}  # object_key -> bytes
        self.upload_calls = 0
        self.fail_for = set()

    @staticmethod
    def object_key(paper_id):
        return f"raw/{paper_id}.pdf"

    def pdf_exists(self, paper_id):
        return self.object_key(paper_id) in self.objects

    def upload_pdf(self, pdf_path, paper_id):
        if paper_id in self.fail_for:
            raise ConnectionError("simulated B2 outage")
        self.upload_calls += 1
        with open(pdf_path, "rb") as f:
            self.objects[self.object_key(paper_id)] = f.read()
        return self.object_key(paper_id)


def fake_download(url, path):
    with open(path, "wb") as f:
        f.write(b"%PDF-1.4 fake pdf for " + url.encode())


def fake_embed(chunks):
    return [[0.1] * VECTOR_SIZE for _ in chunks]


# ---------- tests ----------

@patch("src.vector_store.embed_chunks", side_effect=fake_embed)
@patch("scripts.scrape_papers.urlretrieve", side_effect=fake_download)
@patch("scripts.scrape_papers.arxiv.Client")
class TestDualStorageScraper(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.data_dir = Path(self.tmp)
        patch.object(scrape_papers, "DATA_DIR", self.data_dir).start()
        patch.object(scrape_papers, "LOG_FILE", self.data_dir / "seen_papers.json").start()
        self.storage = FakeObjectStorage()
        self.qdrant = QdrantClient(":memory:")
        # the scraper closes the client at the end of a run; keep it open for asserts
        self.qdrant_close = patch.object(self.qdrant, "close").start()

    def tearDown(self):
        patch.stopall()
        shutil.rmtree(self.tmp)

    def run_scraper(self, mock_client, ids):
        mock_client.return_value.results.return_value = [FakeResult(i) for i in ids]
        return scrape_papers.scrape_new_papers(
            storage=self.storage,
            vector_client=self.qdrant,
            store_embeddings=generate_and_store_embeddings,
        )

    def all_points(self):
        points, _ = self.qdrant.scroll(COLLECTION_NAME, limit=1000, with_payload=True)
        return points

    # Test the full scraping -> object storage -> vector storage workflow end-to-end
    def test_end_to_end(self, mock_client, mock_dl, mock_embed):
        new_count, failed = self.run_scraper(mock_client, ["2610.00001v1", "2610.00002v1"])

        self.assertEqual((new_count, failed), (2, []))
        self.assertEqual(len(self.storage.objects), 2)
        self.assertEqual(len(self.all_points()), 2)

    # Verify that each scraped paper has a corresponding PDF in object storage
    def test_each_paper_has_pdf_in_object_storage(self, mock_client, mock_dl, mock_embed):
        ids = ["2610.00001v1", "2610.00002v1", "2610.00003v1"]
        self.run_scraper(mock_client, ids)

        for pid in ids:
            self.assertTrue(self.storage.pdf_exists(pid), pid)
            self.assertTrue(self.storage.objects[f"raw/{pid}.pdf"].startswith(b"%PDF"))

    # Verify the vector entry has the correct metadata and an object ID that
    # references the corresponding PDF in object storage
    def test_vector_entry_metadata_and_object_id(self, mock_client, mock_dl, mock_embed):
        self.run_scraper(mock_client, ["2610.00001v1"])

        (point,) = self.all_points()
        p = point.payload
        self.assertEqual(p["paper_id"], "2610.00001v1")
        self.assertEqual(p["title"], "Title of 2610.00001v1")
        self.assertEqual(p["authors"], ["Ada Lovelace", "Alan Turing"])
        self.assertEqual(p["abstract"], "Abstract of 2610.00001v1.")
        self.assertEqual(p["object_storage_key"], "raw/2610.00001v1.pdf")
        self.assertIn(p["object_storage_key"], self.storage.objects)

    # Test rescraping: new papers are added without creating duplicate entries
    def test_rescrape_adds_new_without_duplicates(self, mock_client, mock_dl, mock_embed):
        self.run_scraper(mock_client, ["2610.00001v1", "2610.00002v1"])
        # next run: one brand-new paper on top, the two old ones still listed
        new_count, _ = self.run_scraper(
            mock_client, ["2610.00003v1", "2610.00001v1", "2610.00002v1"]
        )

        self.assertEqual(new_count, 1)
        self.assertEqual(self.storage.upload_calls, 3)
        self.assertEqual(len(self.all_points()), 3)
        self.assertEqual(mock_dl.call_count, 3)  # old papers not downloaded again

    # Confirm existing papers are handled correctly on following runs
    def test_same_results_twice_is_a_no_op(self, mock_client, mock_dl, mock_embed):
        ids = ["2610.00001v1", "2610.00002v1"]
        self.run_scraper(mock_client, ids)
        new_count, _ = self.run_scraper(mock_client, ids)

        self.assertEqual(new_count, 0)
        self.assertEqual(len(self.all_points()), 2)
        self.assertEqual(self.storage.upload_calls, 2)

    def test_failed_paper_is_retried_next_run_without_duplicates(self, mock_client, mock_dl, mock_embed):
        self.storage.fail_for = {"2610.00002v1"}
        new_count, failed = self.run_scraper(mock_client, ["2610.00001v1", "2610.00002v1"])
        self.assertEqual((new_count, failed), (1, ["2610.00002v1"]))
        self.assertNotIn("2610.00002v1", scrape_papers.load_seen_ids())

        self.storage.fail_for = set()  # outage over
        new_count, failed = self.run_scraper(mock_client, ["2610.00001v1", "2610.00002v1"])
        self.assertEqual((new_count, failed), (1, []))
        self.assertEqual(len(self.all_points()), 2)
        self.assertEqual(scrape_papers.load_seen_ids(), {"2610.00001v1", "2610.00002v1"})

    def test_pdf_already_in_storage_is_not_reuploaded(self, mock_client, mock_dl, mock_embed):
        # e.g. upload succeeded but the vector step crashed on a previous run
        self.storage.objects["raw/2610.00001v1.pdf"] = b"%PDF existing"
        self.run_scraper(mock_client, ["2610.00001v1"])

        self.assertEqual(self.storage.upload_calls, 0)
        mock_dl.assert_not_called()
        self.assertEqual(self.all_points()[0].payload["object_storage_key"], "raw/2610.00001v1.pdf")

    # Ensure we no longer store the PDFs in the repo / GitHub
    def test_no_pdfs_written_to_data_dir(self, mock_client, mock_dl, mock_embed):
        self.run_scraper(mock_client, ["2610.00001v1", "2610.00002v1"])
        self.assertEqual(list(self.data_dir.glob("*.pdf")), [])
        self.assertEqual(os.listdir(self.data_dir), ["seen_papers.json"])


class TestVectorStoreHelpers(unittest.TestCase):

    def test_paper_exists(self):
        client = QdrantClient(":memory:")
        from src.schema import Paper

        self.assertFalse(vector_store.paper_exists(client, "2610.00001v1"))
        with patch("src.vector_store.embed_chunks", side_effect=fake_embed):
            generate_and_store_embeddings(
                client,
                Paper(paper_id="2610.00001v1", title="t", abstract="a",
                      object_storage_key="raw/2610.00001v1.pdf"),
            )
        self.assertTrue(vector_store.paper_exists(client, "2610.00001v1"))

    def test_get_client_uses_remote_url_when_set(self):
        with patch.dict(os.environ, {"QDRANT_URL": "https://example.qdrant.io", "QDRANT_API_KEY": "k"}), \
             patch("src.vector_store.QdrantClient") as mock_cls:
            vector_store.get_client()
        mock_cls.assert_called_once_with(url="https://example.qdrant.io", api_key="k")


if __name__ == "__main__":
    unittest.main()
