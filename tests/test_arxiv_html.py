from unittest.mock import MagicMock, patch

import requests

from src.agent import retrieve_sections
from src.arxiv_html import get_h2_sections

SAMPLE_HTML = """
<html>
<body>
<h2>1 Introduction</h2>
<p>Routing decides which model answers a query.</p>
<p>We study cost and quality together.</p>
<h2>2 Methods</h2>
<div><p>We train a router on public benchmarks.</p></div>
<h1>References</h1>
<p>This reference text belongs to the next heading.</p>
</body>
</html>
"""

NO_SECTIONS_HTML = "<html><body><p>No headings here.</p></body></html>"


def _response(text=None, error=None):
    response = MagicMock()
    if error:
        response.raise_for_status.side_effect = error
    else:
        response.text = text
        response.raise_for_status.return_value = None
    return response


def test_get_h2_sections_returns_header_and_full_content():
    with patch("src.arxiv_html.requests.get", return_value=_response(SAMPLE_HTML)) as get:
        sections = get_h2_sections("2501.12345")

    get.assert_called_once_with("https://arxiv.org/html/2501.12345", timeout=15)
    assert sections == [
        {
            "header": "1 Introduction",
            "content": "Routing decides which model answers a query. We study cost and quality together.",
        },
        {
            "header": "2 Methods",
            "content": "We train a router on public benchmarks.",
        },
    ]


def test_retrieve_sections_skips_failed_and_empty_papers():
    def fake_get(url, timeout=15):
        arxiv_id = url.rsplit("/", 1)[-1]
        if arxiv_id == "2501.00001":
            return _response(error=requests.HTTPError("404"))
        if arxiv_id == "2501.00002":
            return _response(NO_SECTIONS_HTML)
        return _response(SAMPLE_HTML)

    state = {
        "document_ids": ["2501.00001", "2501.00002", "2501.00003"],
    }
    with patch("src.arxiv_html.requests.get", side_effect=fake_get):
        result = retrieve_sections(state)

    assert list(result["paper_sections"]) == ["2501.00003"]
    assert result["paper_sections"]["2501.00003"][0]["header"] == "1 Introduction"
