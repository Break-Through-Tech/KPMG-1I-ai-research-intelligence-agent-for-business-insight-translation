"""
Get a paper's text from arXiv's HTML version, split into its real sections.

Most new arXiv papers have a web-page version at https://arxiv.org/html/<arxiv_id>.
In that page every main section ("1 Introduction", "2 Related Work", ...) is a
<section class="ltx_section"> (appendices: "ltx_appendix"). Its heading is
usually an <h2>, but papers split into parts push sections down to <h3>, so we
go by the class and return one Section per top-level section.

If arXiv has no HTML version for a paper, fetch_sections() returns None.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

ARXIV_HTML_URL = "https://arxiv.org/html/{arxiv_id}"
USER_AGENT = "KPMG-1I-research-agent (Break Through Tech student project)"
REQUEST_DELAY_SECONDS = 3  # arXiv asks for ~1 request every 3 seconds

# Sections that add noise instead of useful content.
SKIP_SECTION_TITLES = {"references", "bibliography", "acknowledgments", "acknowledgements"}

SECTION_CLASSES = ["ltx_section", "ltx_appendix"]
HEADING_TAGS = ["h2", "h3", "h4", "h5", "h6"]

_last_request_time = 0.0


@dataclass
class Section:
    title: str  # e.g. "Introduction"
    text: str   # all the text in that section, subsections included


def fetch_html(arxiv_id: str) -> str | None:
    """Download the paper's HTML page. Returns None if arXiv has no HTML version."""
    global _last_request_time
    wait = REQUEST_DELAY_SECONDS - (time.time() - _last_request_time)
    if wait > 0:
        time.sleep(wait)

    response = requests.get(
        ARXIV_HTML_URL.format(arxiv_id=arxiv_id),
        headers={"User-Agent": USER_AGENT},
        timeout=60,
    )
    _last_request_time = time.time()

    if response.status_code == 404:
        return None  # arXiv couldn't convert this paper to HTML
    response.raise_for_status()
    return response.text


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _heading_title(heading) -> str:
    """'<h2><span class="ltx_tag">1 </span>Introduction</h2>' -> 'Introduction'"""
    tag = heading.find(class_="ltx_tag")
    if tag:
        tag.extract()  # drop the section number
    return _clean(heading.get_text(" "))


def _section_containers(soup) -> list:
    sections = soup.find_all("section", class_=SECTION_CLASSES)
    if sections:
        return sections
    # A few papers don't use these classes; fall back to the <section> around each <h2>.
    return [h2.find_parent("section") for h2 in soup.find_all("h2") if h2.find_parent("section")]


def parse_sections(html: str) -> list[Section]:
    """Split an arXiv HTML page into its sections (subsections stay in their section)."""
    soup = BeautifulSoup(html, "html.parser")

    # Math is stored as MathML (lots of junk symbols); replace it with its LaTeX text.
    for math in soup.find_all("math"):
        math.replace_with(f" {math.get('alttext', '')} ")

    # Figures, tables and equation blocks don't read well as plain text. Broken
    # conversions sometimes swallow later sections into a figure, so move those out first.
    for el in soup.select("figure, table, .ltx_equation, .ltx_equationgroup"):
        if el.parent is None:
            continue  # already removed with an enclosing figure
        for section in reversed(el.find_all("section", class_=SECTION_CLASSES)):
            el.insert_after(section.extract())
        el.decompose()

    # Broken conversions sometimes nest sections inside each other. Going from the last
    # section to the first and removing each one after reading it means a section's text
    # never includes the sections nested inside it.
    sections: list[Section] = []
    for container in reversed(_section_containers(soup)):
        heading = container.find(HEADING_TAGS)
        title = _heading_title(heading.extract()) if heading else ""
        text = _clean(container.extract().get_text(" "))
        if title and text and title.lower() not in SKIP_SECTION_TITLES:
            sections.append(Section(title=title, text=text))

    return sections[::-1]


def fetch_sections(arxiv_id: str) -> list[Section] | None:
    """Sections of a paper, or None if arXiv has no HTML version for it."""
    html = fetch_html(arxiv_id)
    if html is None:
        return None
    sections = parse_sections(html)
    return sections or None
