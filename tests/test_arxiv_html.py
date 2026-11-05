from unittest.mock import patch

import numpy as np
import requests

from src.agent import _format_sections, retrieve_sections
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


# Papers split into parts: the part gets the <h2> and the real sections drop to <h3>.
PARTS_HTML = """
<html><body><article class="ltx_document">
  <section class="ltx_section" id="S1">
    <h3 class="ltx_title ltx_title_section"><span class="ltx_tag">1 </span>Introduction</h3>
    <div class="ltx_para"><p>Agents can tamper with their traces.</p></div>
  </section>
  <section class="ltx_section" id="S2">
    <h3 class="ltx_title ltx_title_section"><span class="ltx_tag">2 </span>Results</h3>
    <div class="ltx_para"><p>All tested harnesses failed.</p></div>
  </section>
  <section class="ltx_part" id="Pt1">
    <h2 class="ltx_title ltx_title_part"><span class="ltx_tag">Part I </span>Appendix</h2>
    <section class="ltx_appendix" id="A1">
      <h3 class="ltx_title ltx_title_appendix"><span class="ltx_tag">Appendix A </span>Prompts</h3>
      <div class="ltx_para"><p>The prompts we used.</p></div>
    </section>
  </section>
</article></body></html>
"""


def test_sections_under_h3_in_papers_with_parts():
    sections = parse_sections(PARTS_HTML)
    assert [s.title for s in sections] == ["Introduction", "Results", "Prompts"]
    assert sections[1].text == "All tested harnesses failed."


def test_wrongly_nested_sections_are_split_without_repeating_text():
    html = """
    <article>
      <section class="ltx_section"><h2>1 Method</h2><p>Outer text.</p>
        <section class="ltx_section"><h2>2 Results</h2><p>Inner text.</p></section>
      </section>
    </article>
    """
    sections = parse_sections(html)
    assert [(s.title, s.text) for s in sections] == [("1 Method", "Outer text."), ("2 Results", "Inner text.")]


def test_sections_swallowed_by_a_figure_are_kept():
    html = """
    <article>
      <section class="ltx_section"><h2>1 Method</h2><p>Method text.</p>
        <figure class="ltx_table"><table><tr><td>42.0</td></tr></table>
          <section class="ltx_section"><h2>2 Results</h2><p>Results text.</p></section>
        </figure>
      </section>
    </article>
    """
    sections = parse_sections(html)
    assert [(s.title, s.text) for s in sections] == [("1 Method", "Method text."), ("2 Results", "Results text.")]


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


TOPICS = ["routing", "weather", "quantum"]


def fake_embed(texts):
    """One dimension per topic word, so similarity is just topic overlap."""
    vectors = np.array([[text.lower().count(w) for w in TOPICS] for text in texts], dtype=float) + 1e-6
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


PAPER = [
    {"title": "Introduction", "text": "weather " * 100},
    {"title": "Router Design", "text": "routing " * 100},
    {"title": "Quantum Baselines", "text": "quantum " * 100},
    {"title": "Conclusion", "text": "weather " * 100},
]


def test_format_sections_keeps_conclusion_and_most_similar_section():
    with patch("src.agent._embed", side_effect=fake_embed):
        text = _format_sections(PAPER, "How does routing work?", budget=1500)
    assert "Router Design\n" in text
    assert "Conclusion\n" in text
    assert "Introduction\n" not in text
    assert "Quantum Baselines\n" not in text


def test_format_sections_keeps_paper_order():
    with patch("src.agent._embed", side_effect=fake_embed):
        text = _format_sections(PAPER, "routing", budget=1500)
    assert text.index("Router Design\n") < text.index("Conclusion\n")


def test_format_sections_stays_within_budget():
    sections = [{"title": f"Section {i}", "text": "routing " * 500} for i in range(10)]
    with patch("src.agent._embed", side_effect=fake_embed):
        text = _format_sections(sections, "routing", budget=1200)
    assert len(text) == 1200
