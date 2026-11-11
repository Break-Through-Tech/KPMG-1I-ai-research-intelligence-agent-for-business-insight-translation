"""
Get a paper's text from arXiv's HTML version, split into its real sections.

Most new arXiv papers have a web-page version at https://arxiv.org/html/<arxiv_id>.
In that page every main section ("1 Introduction", "2 Related Work", ...) is a
<section class="ltx_section"> (appendices: "ltx_appendix"). Its heading is
usually an <h2>, but papers split into parts push sections down to <h3>, so we
go by the class and return one Section per top-level section.

Two parsers live here:
  - parse_sections(): one flattened Section per top-level section. Used by the
    agent at answer time.
  - parse_blocks(): ordered paragraph-level TextBlocks with the full heading
    breadcrumb. Used by ingestion, which chunks and indexes them.

If arXiv has no HTML version for a paper, fetch_sections() and fetch_blocks() return None.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

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


# ---------- Paragraph-level blocks for ingestion ----------

# Nesting level of each section class, outermost first. Classes are more reliable
# than h2/h3/...: papers split into parts push every heading down a level.
SECTION_LEVELS = {
    "ltx_part": 0,
    "ltx_chapter": 1,
    "ltx_section": 2,
    "ltx_appendix": 2,
    "ltx_subsection": 3,
    "ltx_subsubsection": 4,
    "ltx_paragraph": 5,
    "ltx_subparagraph": 6,
}
TOP_SECTION_LEVEL = 2
ALL_HEADING_TAGS = ["h1", *HEADING_TAGS]

# Removed before parsing. Footnotes (.ltx_note) are dropped rather than left mid-sentence;
# .ltx_ERROR spans only hold LaTeX macros the converter couldn't render.
DROPPED_SELECTORS = ".ltx_bibliography, nav, .ltx_TOC, .ltx_authors, .ltx_dates, .ltx_keywords, .ltx_note, .ltx_ERROR, svg"

LIST_ITEM_BREAK = "\x00"  # marks list-item line breaks while the rest of the whitespace is collapsed


@dataclass
class TextBlock:
    text: str                       # one paragraph, with math as $...$ and $$...$$ (n)
    breadcrumb: list[str]           # e.g. ["4 Method", "4.2 Dual scales", "Column scales as activation rescalings"]
    order: int                      # position among the paper's blocks
    section_path: tuple[str, ...]   # breadcrumb down to the top-level section; chunks never cross it


def _tidy(text: str) -> str:
    """Collapse whitespace and drop the spaces get_text(" ") leaves around punctuation."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;:!?)\]])", r"\1", text)
    return re.sub(r"([(\[])\s+", r"\1", text)


def _convert_math(root: Tag):
    """Display equations become '$$latex$$ (n)' in place; inline math becomes '$latex$'."""
    for equation in root.select("table.ltx_equation, table.ltx_equationgroup"):
        if equation.parent is None:
            continue  # already replaced along with an enclosing equation group
        rows = []
        for row in equation.find_all("tr"):
            latex = " ".join(m.get("alttext", "") for m in row.find_all("math")).strip()
            if not latex:
                continue
            number = row.find(class_="ltx_tag_equation")
            rows.append(f"$${latex}$$" + (f" {_tidy(number.get_text())}" if number else ""))
        equation.replace_with(NavigableString(f" {' '.join(rows)} "))

    for math in root.find_all("math"):
        latex = math.get("alttext", "").strip()
        math.replace_with(NavigableString(f" ${latex}$ " if latex else " "))


def _is_kept_listing(figure: Tag) -> bool:
    """Algorithm floats are kept as text; listings drawn inside regular figures go with the figure."""
    classes = set(figure.get("class", []))
    return not classes & {"ltx_figure", "ltx_table"} and figure.find(class_="ltx_listing") is not None


def _remove_figures_and_tables(root: Tag):
    for el in root.select("figure, table"):
        if el.parent is None or (el.name == "figure" and _is_kept_listing(el)):
            continue
        # Broken conversions sometimes swallow later sections into a figure; move those out first.
        for section in reversed(el.find_all("section")):
            el.insert_after(section.extract())
        el.decompose()


def _section_level(el: Tag) -> int | None:
    levels = [SECTION_LEVELS[c] for c in el.get("class", []) if c in SECTION_LEVELS]
    if levels:
        return min(levels)
    if el.name == "section":  # pages without LaTeXML classes: fall back to the heading tag
        heading = el.find(ALL_HEADING_TAGS, recursive=False)
        if heading:
            return int(heading.name[1])
    return None


def _section_title(section: Tag) -> tuple[str, str]:
    """(title with its number, title without it), e.g. ('4.2 Dual scales', 'Dual scales')."""
    heading = section.find(ALL_HEADING_TAGS, recursive=False)
    if heading is None:
        return "", ""
    full = _tidy(heading.get_text(" ")).removesuffix(".")
    number = heading.find(class_="ltx_tag")
    plain = _tidy(full.removeprefix(_tidy(number.get_text(" ")))) if number else full
    return full, plain


def _paragraph_text(el: Tag) -> str:
    for item in el.find_all("li"):
        marker = item.find(class_="ltx_tag_item")
        label = _tidy(marker.get_text(" ")) if marker else ""
        if marker:
            marker.decompose()
        bullet = "-" if label in {"", "•", "∙", "◦", "-", "–", "*"} else label
        item.insert(0, NavigableString(f"{LIST_ITEM_BREAK}{bullet} "))
    lines = el.get_text(" ").split(LIST_ITEM_BREAK)
    return "\n".join(line for line in map(_tidy, lines) if line)


def _listing_text(el: Tag) -> str:
    """Caption (e.g. 'Algorithm 1: Pandora's Router') followed by one line per listing line."""
    lines = []
    caption = el.find("figcaption")
    if caption:
        tag = caption.find(class_="ltx_tag")
        label = _tidy(tag.extract().get_text(" ")) if tag else ""
        rest = _tidy(caption.get_text(" "))
        lines.append(f"{label}: {rest}" if label and rest else label or rest)
    lines += [_tidy(line.get_text(" ")) for line in el.find_all(class_="ltx_listingline")]
    return "\n".join(line for line in lines if line)


def _walk(el: Tag, crumbs: list[tuple[int, str]], blocks: list[TextBlock]):
    for child in el.children:
        if not isinstance(child, Tag):
            continue
        classes = set(child.get("class", []))
        level = _section_level(child)

        if level is not None:
            title, plain = _section_title(child)
            if plain.lower() in SKIP_SECTION_TITLES:
                continue
            # Keep only outer levels, so wrongly nested sections don't stack on each other.
            inner = [c for c in crumbs if c[0] < level]
            _walk(child, inner + [(level, title)] if title else inner, blocks)
        elif "ltx_abstract" in classes:
            _walk(child, [(TOP_SECTION_LEVEL, "Abstract")], blocks)
        elif "ltx_para" in classes or child.name == "p":
            _add_block(blocks, _paragraph_text(child), crumbs)
        elif classes & {"ltx_theorem", "ltx_proof"}:
            heading = child.find(ALL_HEADING_TAGS, recursive=False)
            title = _tidy(heading.extract().get_text(" ")) if heading else ""
            first = len(blocks)
            _walk(child, crumbs, blocks)
            if title and len(blocks) > first:
                blocks[first].text = f"{title} {blocks[first].text}"
        elif child.name == "figure" or "ltx_listing" in classes:
            _add_block(blocks, _listing_text(child), crumbs)
        elif child.name not in ALL_HEADING_TAGS:
            _walk(child, crumbs, blocks)


def _add_block(blocks: list[TextBlock], text: str, crumbs: list[tuple[int, str]]):
    if not text:
        return
    breadcrumb = [title for _, title in crumbs]
    top = next((i for i, (level, _) in enumerate(crumbs) if level >= TOP_SECTION_LEVEL), len(crumbs) - 1)
    blocks.append(TextBlock(
        text=text,
        breadcrumb=breadcrumb,
        order=len(blocks),
        section_path=tuple(breadcrumb[: top + 1]),
    ))


def parse_blocks(html: str) -> list[TextBlock]:
    """Split an arXiv HTML page into ordered paragraph blocks with their heading breadcrumbs.

    Tables and figures are removed; algorithm listings are kept as text.
    """
    soup = BeautifulSoup(html, "html.parser")
    root = soup.find("article") or soup.body or soup
    for el in root.select(DROPPED_SELECTORS):
        if el.parent is not None:
            el.decompose()
    _convert_math(root)
    _remove_figures_and_tables(root)

    blocks: list[TextBlock] = []
    _walk(root, [], blocks)
    return blocks


def parse_title(html: str) -> str | None:
    heading = BeautifulSoup(html, "html.parser").find(class_="ltx_title_document")
    return _tidy(heading.get_text(" ")) if heading else None


def fetch_blocks(arxiv_id: str) -> list[TextBlock] | None:
    """A paper's paragraph blocks, or None if arXiv has no HTML version for it."""
    html = fetch_html(arxiv_id)
    if html is None:
        return None
    return parse_blocks(html) or None


if __name__ == "__main__":
    # python -m src.arxiv_html <arxiv_id | path/to/page.html>
    import sys
    from pathlib import Path

    source = sys.argv[1]
    html = Path(source).read_text() if source.endswith(".html") else fetch_html(source)
    if html is None:
        sys.exit(f"arXiv has no HTML version of {source}")
    for block in parse_blocks(html):
        print(f"[{block.order}] {' > '.join(block.breadcrumb)}")
        print(f"{block.text}\n")
