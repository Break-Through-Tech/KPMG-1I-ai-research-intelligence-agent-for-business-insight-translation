"""
Tests for scraping arXiv papers into the vector store, section by section.

Runs offline: arXiv search results, the HTML download and the embedding model
are faked. The vector-store tests use a real in-memory Qdrant.

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
from src.arxiv_html import Section, parse_sections, split_long_text


# A trimmed-down copy of what an arxiv.org/html/<id> page looks like.
SAMPLE_HTML = """
<html><body><article class="ltx_document">
  <h1 class="ltx_title ltx_title_document">Pandora's AI Model Routing Box</h1>
  <div class="ltx_abstract"><h6 class="ltx_title">Abstract</h6>
    <p class="ltx_p">Routing requires estimating each specialist's value.</p></div>

  <section class="ltx_section" id="S1">
    <h2 class="ltx_title ltx_title_section"><span class="ltx_tag ltx_tag_section">1 </span>Introduction</h2>
    <div class="ltx_para"><p class="ltx_p">Heterogeneous AI systems route queries to specialists.</p></div>
  </section>

  <section class="ltx_section" id="S2">
    <h2 class="ltx_title ltx_title_section"><span class="ltx_tag ltx_tag_section">2 </span>Method</h2>
    <div class="ltx_para"><p class="ltx_p">We model the cost as
      <math alttext="c_{i}" display="inline"><semantics><msub><mi>c</mi><mi>i</mi></msub></semantics></math>
      for each specialist.</p></div>
    <figure class="ltx_table"><table><tr><td>99.9</td></tr></table>
      <figcaption>Table 1: results</figcaption></figure>
    <section class="ltx_subsection" id="S2.SS1">
      <h3 class="ltx_title ltx_title_subsection">2.1 Gaussian signals</h3>
      <div class="ltx_para"><p class="ltx_p">Under a Gaussian model we get closed forms.</p></div>
    </section>
  </section>

  <section class="ltx_bibliography" id="bib">
    <h2 class="ltx_title ltx_title_bibliography">References</h2>
    <ul><li>[1] Some cited paper.</li></ul>
  </section>
</article></body></html>
"""


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
    """In-memory stand-in for Qdrant: {paper_id: [(section_title, text), ...]}"""

    def __init__(self):
        self.papers = {}
        self.fail_for = set()

    def store(self, client, paper, chunks):
        if paper.paper_id in self.fail_for:
            raise ConnectionError("simulated Qdrant outage")
        self.papers[paper.paper_id] = list(chunks)
        self.last_paper = paper

    def delete(self, client, paper_id):
        self.papers.pop(paper_id, None)

    def total_chunks(self):
        return sum(len(c) for c in self.papers.values())


def fake_fetch(arxiv_id):
    if arxiv_id.endswith("nohtml"):
        return None
    return [Section("Introduction", f"Intro of {arxiv_id}."), Section("Method", f"Method of {arxiv_id}.")]


# ---------- parsing the arXiv HTML ----------

class TestParseSections(unittest.TestCase):

    def setUp(self):
        self.sections = parse_sections(SAMPLE_HTML)

    def test_one_section_per_h2_without_numbers(self):
        self.assertEqual([s.title for s in self.sections], ["Introduction", "Method"])

    def test_skips_references(self):
        self.assertNotIn("References", [s.title for s in self.sections])

    def test_section_text_includes_subsections(self):
        method = self.sections[1].text
        self.assertIn("We model the cost as", method)
        self.assertIn("Gaussian model", method)

    def test_math_becomes_latex_and_tables_are_dropped(self):
        method = self.sections[1].text
        self.assertIn("c_{i}", method)
        self.assertNotIn("99.9", method)

    def test_split_long_text(self):
        text = " ".join(f"Sentence number {i}." for i in range(200))
        pieces = split_long_text(text, max_chars=300)
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(len(p) <= 300 for p in pieces))
        self.assertEqual(" ".join(pieces), text)  # nothing lost


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
        return scrape_papers.scrape_new_papers(
            vector_client=object(),
            fetch=fake_fetch,
            store=self.store.store,
            delete=self.store.delete,
        )

    def test_stores_abstract_plus_sections(self, mock_client):
        self.run_scraper(mock_client, ["2610.00001v1"])
        self.assertEqual(
            self.store.papers["2610.00001v1"],
            [
                ("Abstract", "Abstract of 2610.00001v1."),
                ("Introduction", "Intro of 2610.00001v1."),
                ("Method", "Method of 2610.00001v1."),
            ],
        )

    def test_metadata_is_kept(self, mock_client):
        self.run_scraper(mock_client, ["2610.00001v1"])
        paper = self.store.last_paper
        self.assertEqual(paper.paper_id, "2610.00001v1")
        self.assertEqual(paper.title, "Title of 2610.00001v1")
        self.assertEqual(paper.authors, ["Ada Lovelace", "Alan Turing"])

    def test_no_html_falls_back_to_abstract(self, mock_client):
        self.run_scraper(mock_client, ["2610.00002nohtml"])
        self.assertEqual(
            self.store.papers["2610.00002nohtml"], [("Abstract", "Abstract of 2610.00002nohtml.")]
        )

    def test_rescrape_adds_only_new_papers(self, mock_client):
        self.run_scraper(mock_client, ["2610.00001v1", "2610.00002v1"])
        new_count, _ = self.run_scraper(mock_client, ["2610.00003v1", "2610.00001v1", "2610.00002v1"])
        self.assertEqual(new_count, 1)
        self.assertEqual(len(self.store.papers), 3)
        self.assertEqual(self.store.total_chunks(), 9)

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

    def test_chunks_are_tagged_with_paper_id_and_section(self):
        self.vs.store_paper_chunks(
            self.client, self.paper, [("Abstract", "Abs."), ("Introduction", "Intro.")]
        )
        payloads = sorted((p.payload["chunk_index"], p.payload["section_title"], p.payload["paper_id"])
                          for p in self.all_points())
        self.assertEqual(payloads, [(0, "Abstract", "2610.00001v1"), (1, "Introduction", "2610.00001v1")])
        self.assertTrue(self.vs.paper_exists(self.client, "2610.00001v1"))

    def test_restoring_a_paper_does_not_duplicate(self):
        chunks = [("Abstract", "Abs."), ("Introduction", "Intro."), ("Method", "M.")]
        self.vs.store_paper_chunks(self.client, self.paper, chunks)
        self.vs.delete_paper(self.client, self.paper.paper_id)
        self.vs.store_paper_chunks(self.client, self.paper, chunks[:2])
        self.assertEqual(len(self.all_points()), 2)


if __name__ == "__main__":
    unittest.main()
