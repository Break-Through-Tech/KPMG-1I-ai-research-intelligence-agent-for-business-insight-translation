"""
Tests for parse_blocks(): paragraph-level blocks with heading breadcrumbs.

The fixtures in tests/fixtures/arxiv_html/ are trimmed excerpts of real
arxiv.org/html pages: a handful of real elements (paragraphs, an equation, a
theorem, an algorithm, a table, a figure, ...) with the headings around them.
Math is reduced to its alttext and unused attributes are stripped.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from bs4 import BeautifulSoup

from src.arxiv_html import fetch_blocks, parse_blocks

FIXTURES = Path(__file__).parent / "fixtures" / "arxiv_html"


def load(name):
    return (FIXTURES / name).read_text()


def all_text(blocks):
    return "\n".join(b.text for b in blocks)


@pytest.fixture(scope="module")
def dualquant():
    return parse_blocks(load("dualquant_2610.13767v1.html"))


@pytest.fixture(scope="module")
def pandora():
    return parse_blocks(load("pandora_2608.20316v1.html"))


@pytest.fixture(scope="module")
def tampering():
    return parse_blocks(load("trace_tampering_2609.30266v1.html"))


# ---------- structure: order and breadcrumbs ----------

def test_blocks_are_in_document_order(dualquant):
    assert [b.order for b in dualquant] == list(range(len(dualquant)))
    assert [b.breadcrumb[0] for b in dualquant] == [
        "Abstract",
        "1 Introduction",
        "1 Introduction",
        "4 Methodology",
        "4 Methodology",
        "4 Methodology",
        "4 Methodology",
        "5 Experiments",
        "Appendix C Proof of Theorem 1",
    ]


def test_breadcrumb_keeps_full_hierarchy_with_section_numbers(dualquant):
    assert dualquant[3].breadcrumb == ["4 Methodology", "4.1 Problem description"]
    assert dualquant[6].breadcrumb == [
        "4 Methodology",
        "4.3 MSE optimization via block coordinate descent",
        "Convergence analysis",
    ]


def test_paragraph_headings_are_breadcrumbs_not_text(dualquant):
    assert dualquant[6].breadcrumb[-1] == "Convergence analysis"
    assert "Convergence analysis" not in dualquant[6].text


def test_section_path_stops_at_top_level_section(dualquant):
    assert dualquant[6].section_path == ("4 Methodology",)


def test_unusual_heading_levels_and_parts(tampering):
    # This paper is split into parts: sections use h3, subsections h4,
    # subsubsections h5 and paragraph headings h6.
    assert tampering[0].breadcrumb == ["1 Introduction"]
    assert tampering[1].breadcrumb == [
        "Part I Appendix",
        "Appendix A Attack Settings and Prompt Details",
        "A.1 Direct User Prompting",
        "Fabricated container-reset event",
    ]
    assert tampering[2].breadcrumb[-1] == "A.3.1 Summary Anonymization"
    assert tampering[1].section_path == ("Part I Appendix", "Appendix A Attack Settings and Prompt Details")


def test_abstract_is_its_own_section(dualquant):
    assert dualquant[0].breadcrumb == ["Abstract"]
    assert dualquant[0].text.startswith("Low-bit inference on accelerators")


def test_multiple_paragraphs_in_a_section_stay_separate(dualquant):
    intro = [b for b in dualquant if b.breadcrumb == ["1 Introduction"]]
    assert len(intro) == 2
    assert intro[0].text.startswith("Large Language Models (LLMs)")
    assert intro[1].text.startswith("In this work, we introduce DualQuant")


# ---------- content ----------

def test_inline_math_becomes_dollar_latex(dualquant):
    assert "computing $\\mathbf{X}\\mathbf{W}^{\\top}$ with activation matrix" in dualquant[3].text


def test_display_equation_stays_in_its_paragraph_with_its_number(dualquant):
    text = dualquant[3].text
    assert "can be written as $$" in text
    assert "$$ (1)" in text
    assert text.index("$$ (1)") < text.index("where the activation error")


def test_theorem_title_prefixes_its_text(dualquant):
    assert dualquant[6].text.startswith("Assumption 1 (Non-degenerate quantization). For all iterations $t$")


def test_proof_title_prefixes_its_text(pandora):
    assert pandora[2].text.startswith("Proof. By admissibility")


def test_algorithm_is_kept_as_captioned_lines(dualquant):
    lines = dualquant[4].text.split("\n")
    assert lines[0] == "Algorithm 1: DualQuant: Dual-Scale Quantization via Coordinate Descent"
    assert lines[1].startswith("1: Weight tile $\\mathbf{W}$")


def test_bullet_list_items_become_lines(pandora):
    lines = pandora[1].text.split("\n")
    assert lines[0].endswith("compare with the following methods:")
    assert lines[1].startswith("- ${f}$")
    assert all(line.startswith("- ") for line in lines[1:])


# ---------- removed content ----------

def test_tables_and_figures_are_removed(dualquant):
    text = all_text(dualquant)
    assert "Llama-3.1-8B downstream accuracy" not in text  # table caption
    assert "HellaSwag" not in text                          # table cell
    assert "Per-element quantization error" not in text     # figure caption


def test_listings_inside_figures_are_removed(tampering):
    assert "GPT-5.6-Sol" not in all_text(tampering)


def test_footnotes_are_dropped(dualquant, pandora, tampering):
    for name, blocks in [
        ("dualquant_2610.13767v1.html", dualquant),
        ("pandora_2608.20316v1.html", pandora),
        ("trace_tampering_2609.30266v1.html", tampering),
    ]:
        notes = BeautifulSoup(load(name), "html.parser").select(".ltx_note_content")
        for note in notes:
            note_text = " ".join(note.get_text(" ").split()[-4:])
            assert note_text not in all_text(blocks)


def test_references_and_acknowledgements_are_skipped(pandora, dualquant):
    assert all("Acknowledgements" not in b.breadcrumb for b in pandora)
    assert all("References" not in b.breadcrumb for b in dualquant)


def section(title: str, number: str = "") -> str:
    tag = f'<span class="ltx_tag ltx_tag_section">{number} </span>' if number else ""
    return (f'<section class="ltx_section"><h2 class="ltx_title">{tag}{title}</h2>'
            f'<div class="ltx_para"><p>Text of {title}.</p></div></section>')


@pytest.mark.parametrize("title", [
    "Acknowledgment", "Reproducibility Statement", "Funding", "Author Contributions",
    "CRediT authorship contribution statement", "Declaration of competing interest", "Declarations",
    "Data and Code Availability", "AI Use Statement", "Usage of AI", "AI-Assistance Disclosure",
    "Declaration of Generative AI and AI-assisted Technologies", "Ethics and Generative-AI Use Disclosure",
])
def test_administrative_sections_are_skipped(title):
    blocks = parse_blocks(f"<article>{section('Results', '1')}{section(title, '9')}</article>")
    assert [b.breadcrumb for b in blocks] == [["1 Results"]]


@pytest.mark.parametrize("title", [
    "Ethics Statement", "Impact Statement", "Our Contributions", "Reproducibility and Evidence Provenance",
    "Use of LLMs as Judges", "How Developers Use AI Assistants", "Token Usage and Billing",
])
def test_similar_looking_content_sections_are_kept(title):
    [block] = parse_blocks(f"<article>{section(title)}</article>")
    assert block.breadcrumb == [title]


# ---------- malformed or unusual HTML ----------

def test_wrongly_nested_sections_get_their_own_breadcrumbs():
    html = """
    <article>
      <section class="ltx_section"><h2>1 Method</h2><div class="ltx_para"><p>Outer text.</p></div>
        <section class="ltx_section"><h2>2 Results</h2><div class="ltx_para"><p>Inner text.</p></div></section>
      </section>
    </article>
    """
    blocks = parse_blocks(html)
    assert [(b.breadcrumb, b.text) for b in blocks] == [(["1 Method"], "Outer text."), (["2 Results"], "Inner text.")]


def test_sections_swallowed_by_a_figure_are_kept():
    html = """
    <article>
      <section class="ltx_section"><h2>1 Method</h2><div class="ltx_para"><p>Method text.</p></div>
        <figure class="ltx_table"><table><tr><td>42.0</td></tr></table>
          <section class="ltx_section"><h2>2 Results</h2><div class="ltx_para"><p>Results text.</p></div></section>
        </figure>
      </section>
    </article>
    """
    blocks = parse_blocks(html)
    assert [(b.breadcrumb, b.text) for b in blocks] == [(["1 Method"], "Method text."), (["2 Results"], "Results text.")]


def test_pages_without_latexml_classes_fall_back_to_heading_tags():
    html = """
    <article>
      <section><h2>Intro</h2><p>First.</p>
        <section><h3>Details</h3><p>Second.</p></section>
      </section>
    </article>
    """
    blocks = parse_blocks(html)
    assert [(b.breadcrumb, b.text) for b in blocks] == [(["Intro"], "First."), (["Intro", "Details"], "Second.")]


def test_text_outside_any_section_has_an_empty_breadcrumb():
    blocks = parse_blocks("<article><div class='ltx_para'><p>Loose text.</p></div></article>")
    assert [(b.breadcrumb, b.section_path, b.text) for b in blocks] == [([], (), "Loose text.")]


def test_section_without_heading_inherits_parent_breadcrumb():
    html = """
    <article>
      <section class="ltx_section"><h2>1 Method</h2>
        <section class="ltx_subsection"><div class="ltx_para"><p>No heading here.</p></div></section>
      </section>
    </article>
    """
    assert parse_blocks(html)[0].breadcrumb == ["1 Method"]


def test_empty_paragraphs_and_missing_alttext_are_skipped():
    html = """
    <article>
      <div class="ltx_para"><p>   </p></div>
      <div class="ltx_para"><p>Value <math></math> here.</p></div>
    </article>
    """
    blocks = parse_blocks(html)
    assert [(b.order, b.text) for b in blocks] == [(0, "Value here.")]


def test_numbered_list_keeps_its_labels():
    html = """
    <article><div class="ltx_para"><p>Steps:</p>
      <ol class="ltx_enumerate">
        <li><span class="ltx_tag ltx_tag_item">1.</span><div class="ltx_para"><p>Fetch.</p></div></li>
        <li><span class="ltx_tag ltx_tag_item">2.</span><div class="ltx_para"><p>Parse.</p></div></li>
      </ol>
    </div></article>
    """
    assert parse_blocks(html)[0].text == "Steps:\n1. Fetch.\n2. Parse."


def test_fetch_blocks_returns_none_without_html_version():
    with patch("src.arxiv_html.fetch_html", return_value=None):
        assert fetch_blocks("2609.27746v1") is None
