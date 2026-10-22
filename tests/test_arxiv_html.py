from unittest.mock import patch

import requests

from src.agent import format_sections, retrieve_sections
from src.arxiv_html import Section, parse_sections

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


def test_one_section_per_h2_without_numbers():
    assert [s.title for s in parse_sections(SAMPLE_HTML)] == ["Introduction", "Method"]


def test_skips_references():
    assert "References" not in [s.title for s in parse_sections(SAMPLE_HTML)]


def test_section_text_includes_subsections():
    method = parse_sections(SAMPLE_HTML)[1].text
    assert "We model the cost as" in method
    assert "Gaussian model" in method


def test_math_becomes_latex_and_tables_are_dropped():
    method = parse_sections(SAMPLE_HTML)[1].text
    assert "c_{i}" in method
    assert "99.9" not in method


def test_retrieve_sections_skips_failed_and_missing_papers():
    def fake_fetch(arxiv_id):
        if arxiv_id == "2501.00001":
            raise requests.HTTPError("503")
        if arxiv_id == "2501.00002":
            return None
        return [Section("Introduction", "Routing decides which model answers.")]

    state = {"document_ids": ["2501.00001", "2501.00002", "2501.00003"]}
    with patch("src.agent.fetch_sections", side_effect=fake_fetch):
        result = retrieve_sections(state)

    assert result["paper_sections"] == {
        "2501.00003": [{"title": "Introduction", "text": "Routing decides which model answers."}]
    }


def test_format_sections_stays_within_budget():
    sections = [{"title": f"Section {i}", "text": "x" * 500} for i in range(10)]
    text = format_sections(sections, budget=1200)
    assert len(text) == 1200
    assert text.startswith("Section 0\n")
