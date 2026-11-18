"""
Get a paper's text from arXiv's HTML version as ordered paragraph blocks.

Most new arXiv papers have a web-page version at https://arxiv.org/html/<arxiv_id>.
parse_blocks() turns that page into TextBlocks, each one paragraph with its full
heading breadcrumb, which ingestion chunks and indexes.

If arXiv has no HTML version for a paper, fetch_blocks() returns None.
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

# Sections that add noise instead of useful content: references and administrative statements.
# Matched against the whole heading (number removed, lowercased), so e.g. "Our Contributions" or
# "Reproducibility and Evidence Provenance" are kept. Ethics and impact statements are kept too.
SKIP_SECTION = re.compile(r"""^(?:
    references | bibliography | acknowledge?ments?
  | reproducibility\ statement
  | funding(?:\ statement)?
  | (?:author\ )?declarations | declaration\ of\ (?:competing\ )?interests? | competing\ interests?
  | conflicts?\ of\ interests?
  | (?:credit\ )?(?:authors?'?|authorship)\ (?:authorship\ )?contributions?(?:\ statement)?
  | (?:data|code|artifact|software|materials)(?:\ and\ (?:data|code|materials))?\ availability(?:\ statement)?
)$""", re.X)
# Statements of how the authors used AI tools, e.g. "AI Use Statement", "Declaration of Generative AI
# and AI-assisted Technologies". Anchored phrasings only: many papers study AI use in their content.
AI_USE_DISCLOSURE = re.compile(r"""^(?=.*\b(?:use|usage|disclosure|declaration|assist))
    (?:ethics\ and\ )?(?:declaration\ (?:of|on)\ )?(?:the\ )?(?:use\ of\ |usage\ of\ )?(?:generative[-\ ])?
    (?:ai|llms?)(?:[-\ ]assist(?:ance|ants?|ed)?)?(?:\ and\ ai-assisted\ technologies)?
    (?:\ use|\ usage)?(?:\ statement|\ disclosure)?$""", re.X)

HEADING_TAGS = ["h2", "h3", "h4", "h5", "h6"]

_last_request_time = 0.0


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


def _is_skipped_section(plain_title: str) -> bool:
    title = plain_title.lower()
    return bool(SKIP_SECTION.match(title) or AI_USE_DISCLOSURE.match(title))


def _walk(el: Tag, crumbs: list[tuple[int, str]], blocks: list[TextBlock]):
    for child in el.children:
        if not isinstance(child, Tag):
            continue
        classes = set(child.get("class", []))
        level = _section_level(child)

        if level is not None:
            title, plain = _section_title(child)
            if _is_skipped_section(plain):
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
