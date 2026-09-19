import unittest
from unittest.mock import patch
import scrape_papers


class FakePaper:
    def __init__(self, arxiv_id):
        self.entry_id = f"http://arxiv.org/abs/{arxiv_id}"
        self.pdf_url = f"http://arxiv.org/pdf/{arxiv_id}"


class TestScraper(unittest.TestCase):

    @patch("scrape_papers.urlretrieve")
    @patch("scrape_papers.save_seen_ids")
    @patch("scrape_papers.load_seen_ids")
    @patch("scrape_papers.arxiv.Client")
    def test_only_downloads_new_papers(self, mock_client, mock_load, mock_save, mock_dl):
        mock_load.return_value = {"old1"}
        mock_client.return_value.results.return_value = [
            FakePaper("new1"), FakePaper("new2"), FakePaper("old1"), FakePaper("older1")
        ]

        scrape_papers.scrape_new_papers()

        self.assertEqual(mock_dl.call_count, 2)  # only new1 and new2

    @patch("scrape_papers.urlretrieve")
    @patch("scrape_papers.save_seen_ids")
    @patch("scrape_papers.load_seen_ids")
    @patch("scrape_papers.arxiv.Client")
    def test_no_new_papers_downloads_nothing(self, mock_client, mock_load, mock_save, mock_dl):
        mock_load.return_value = {"paper1"}
        mock_client.return_value.results.return_value = [FakePaper("paper1")]

        scrape_papers.scrape_new_papers()

        mock_dl.assert_not_called()


if __name__ == "__main__":
    unittest.main()