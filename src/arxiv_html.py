"""
Get a paper's text from arXiv's HTML version, split into its real sections.

Most new arXiv papers have a web-page version at https://arxiv.org/html/<arxiv_id>.
In that page every main section ("1 Introduction", "2 Related Work", ...) is
marked with an <h2> header. We read the text under each <h2> and return it as
one Section per header.

If arXiv has no HTML version for a paper, fetch_sections() returns None and the
caller falls back to storing only the abstract.
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

# Sections that add noise to search results instead of useful content.
SKIP_SECTION_TITLES = {"references", "bibliography", "acknowledgments", "acknowledgements"}

# all-MiniLM-L6-v2 only reads the first ~256 tokens (~1000 characters) of a text,
# so long sections are split into pieces about this size or the rest is ignored.
MAX_PIECE_CHARS = 1000

_last_request_time = 0.0


@dataclass
class Section:
    title: str  # e.g. "Introduction"
    text: str   # all the text under that <h2>


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


def _heading_title(h2) -> str:
    """'<h2><span class="ltx_tag">1 </span>Introduction</h2>' -> 'Introduction'"""
    tag = h2.find(class_="ltx_tag")
    if tag:
        tag.extract()  # drop the section number
    return _clean(h2.get_text(" "))


def parse_sections(html: str) -> list[Section]:
    """Split an arXiv HTML page into sections, one per <h2> header."""
    soup = BeautifulSoup(html, "html.parser")

    # Math is stored as MathML (lots of junk symbols); replace it with its LaTeX text.
    for math in soup.find_all("math"):
        math.replace_with(f" {math.get('alttext', '')} ")

    # Figures, tables and equation blocks don't read well as plain text.
    for el in soup.select("figure, table, .ltx_equation, .ltx_equationgroup"):
        el.decompose()

    sections: list[Section] = []
    for h2 in soup.find_all("h2"):
        title = _heading_title(h2)
        if not title or title.lower() in SKIP_SECTION_TITLES:
            continue

        # The section's text is everything in its <section> except the <h2> itself.
        container = h2.find_parent("section")
        if container is None:
            continue
        h2.extract()
        text = _clean(container.get_text(" "))
        if text:
            sections.append(Section(title=title, text=text))

    return sections


def fetch_sections(arxiv_id: str) -> list[Section] | None:
    """Sections of a paper, or None if arXiv has no HTML version for it."""
    html = fetch_html(arxiv_id)
    if html is None:
        return None
    sections = parse_sections(html)
    return sections or None


def split_long_text(text: str, max_chars: int = MAX_PIECE_CHARS) -> list[str]:
    """Split text into pieces of at most ~max_chars, breaking between sentences."""
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces, current = [], ""
    for sentence in sentences:
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)

    # A single sentence longer than max_chars is cut into fixed-size slices.
    result = []
    for piece in pieces:
        result.extend(piece[i : i + max_chars] for i in range(0, len(piece), max_chars))
    return result
