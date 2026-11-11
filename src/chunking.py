"""Split a paper's paragraph blocks (from arxiv_html.parse_blocks) into chunks for embedding.

- Chunks never cross a top-level section (TextBlock.section_path); a section that fits in
  MAX_TOKENS is one chunk.
- Longer sections are cut at sentence boundaries into evenly sized chunks of about
  TARGET_TOKENS, preferring to end at a paragraph, so no section ends in a tiny leftover chunk.
- Each chunk after the first in a section starts with the previous chunk's last sentences,
  up to OVERLAP_TOKENS.
- Tokens are counted on the chunk text with the dense embedding model's tokenizer.
"""

import itertools
import math
import re
from dataclasses import dataclass
from typing import Callable

import pysbd

from src.arxiv_html import TextBlock

TARGET_TOKENS = 500
MAX_TOKENS = 600
OVERLAP_TOKENS = 60
# A last sentence longer than OVERLAP_TOKENS is still carried over if it's at most this long.
MAX_SENTENCE_OVERLAP_TOKENS = 150
# Once a chunk holds this fraction of its share of the section, it ends at the next paragraph end.
PARAGRAPH_END_FRACTION = 0.8

# pysbd splits after these when the next word isn't capitalised, e.g. "Eq. (2)" or "on avg. across".
_ABBREVIATION_END = re.compile(
    r"\b(?:eqs?|secs?|tabs?|figs?|refs?|algs?|thm|def|app|avg|approx|resp|cf)\.$", re.IGNORECASE
)
_MATH = re.compile(r"\$\$.*?\$\$|\$[^$]*\$")
_MATH_PLACEHOLDER = re.compile(r"MATHSPAN(\d+)X")

_segmenter: pysbd.Segmenter | None = None


@dataclass
class Chunk:
    text: str              # sentences joined back as written: paragraphs separated by blank lines
    breadcrumb: list[str]  # headings shared by every paragraph in the chunk
    chunk_index: int       # position among the paper's chunks
    first_block: int       # TextBlock.order of the first and last paragraphs covered, overlap included
    last_block: int
    token_count: int


@dataclass
class _Sentence:
    text: str
    tokens: int
    block: TextBlock
    line: int            # line within the block; list items and algorithm lines are separate lines
    paragraph_end: bool  # last sentence of its block


def _split_line(line: str) -> list[str]:
    global _segmenter
    if _segmenter is None:
        _segmenter = pysbd.Segmenter(language="en", clean=False)

    # Hide math so periods inside it can't end a sentence.
    maths: list[str] = []

    def hide(match: re.Match) -> str:
        maths.append(match.group())
        return f"MATHSPAN{len(maths) - 1}X"

    merged: list[str] = []
    for piece in (p.strip() for p in _segmenter.segment(_MATH.sub(hide, line))):
        if not piece:
            continue
        if merged and _ABBREVIATION_END.search(merged[-1]) and not piece[0].isupper():
            merged[-1] += " " + piece
        else:
            merged.append(piece)
    return [_MATH_PLACEHOLDER.sub(lambda m: maths[int(m.group(1))], s) for s in merged]


def split_sentences(text: str) -> list[str]:
    """Sentences of one paragraph. Lines are split separately, so a sentence never spans lines."""
    return [sentence for line in text.split("\n") for sentence in _split_line(line)]


def _sentences(blocks: list[TextBlock], count_tokens: Callable[[str], int]) -> list[_Sentence]:
    sentences = []
    for block in blocks:
        start = len(sentences)
        for line_number, line in enumerate(block.text.split("\n")):
            for text in _split_line(line):
                sentences.append(_Sentence(text, count_tokens(text), block, line_number, False))
        if len(sentences) > start:
            sentences[-1].paragraph_end = True
    return sentences


def _chunk_count(tokens: int, target_tokens: int, max_new_tokens: int) -> int:
    """Number of chunks closest to target_tokens each, without any going over max_new_tokens."""
    return max(1, math.floor(tokens / target_tokens + 0.5), math.ceil(tokens / max_new_tokens))


def _overlap(previous: list[_Sentence], overlap_tokens: int) -> list[_Sentence]:
    """The previous chunk's trailing sentences that fit in overlap_tokens, or at least its last
    sentence if that's no longer than MAX_SENTENCE_OVERLAP_TOKENS."""
    tail: list[_Sentence] = []
    used = 0
    for sentence in reversed(previous):
        if used + sentence.tokens > overlap_tokens:
            break
        tail.insert(0, sentence)
        used += sentence.tokens
    if not tail and previous[-1].tokens <= MAX_SENTENCE_OVERLAP_TOKENS:
        tail = [previous[-1]]
    return tail


def _pack_section(sentences: list[_Sentence], target_tokens: int, max_tokens: int,
                  overlap_tokens: int) -> list[list[_Sentence]]:
    if sum(s.tokens for s in sentences) <= max_tokens:
        return [sentences]

    groups: list[list[_Sentence]] = []
    i = 0
    while i < len(sentences):
        overlap = _overlap(groups[-1], overlap_tokens) if groups else []
        overlap_size = sum(s.tokens for s in overlap)
        remaining = sum(s.tokens for s in sentences[i:])
        if remaining + overlap_size <= max_tokens:
            groups.append(overlap + sentences[i:])
            break

        # Re-split what's left after every chunk, so an early paragraph end doesn't leave a tiny tail.
        budget = min(remaining / _chunk_count(remaining, target_tokens, max_tokens - overlap_tokens),
                     max_tokens - overlap_size)
        new: list[_Sentence] = []
        size = 0
        while i < len(sentences):
            sentence = sentences[i]
            if new and size + sentence.tokens > budget:
                break
            new.append(sentence)
            size += sentence.tokens
            i += 1
            if sentence.paragraph_end and size >= PARAGRAPH_END_FRACTION * budget:
                break
        # A sentence longer than the budget gets a chunk to itself, without overlap if that would exceed the max.
        groups.append(overlap + new if overlap_size + size <= max_tokens else new)
    return groups


def _join(sentences: list[_Sentence]) -> str:
    parts = [sentences[0].text]
    for previous, sentence in itertools.pairwise(sentences):
        if sentence.block is not previous.block:
            parts.append("\n\n")
        elif sentence.line != previous.line:
            parts.append("\n")
        else:
            parts.append(" ")
        parts.append(sentence.text)
    return "".join(parts)


def _common_prefix(breadcrumbs: list[list[str]]) -> list[str]:
    prefix = []
    for titles in zip(*breadcrumbs):
        if any(title != titles[0] for title in titles):
            break
        prefix.append(titles[0])
    return prefix


def chunk_blocks(
    blocks: list[TextBlock],
    count_tokens: Callable[[str], int] | None = None,
    target_tokens: int = TARGET_TOKENS,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunks for one paper, in reading order. count_tokens defaults to the dense model's tokenizer."""
    if count_tokens is None:
        from src.embeddings import count_tokens  # loads the dense model, so only when it's needed
    chunks: list[Chunk] = []
    for _, section in itertools.groupby(blocks, key=lambda b: b.section_path):
        sentences = _sentences(list(section), count_tokens)
        if not sentences:
            continue
        for group in _pack_section(sentences, target_tokens, max_tokens, overlap_tokens):
            covered = list({id(s.block): s.block for s in group}.values())
            chunks.append(Chunk(
                text=_join(group),
                breadcrumb=_common_prefix([b.breadcrumb for b in covered]),
                chunk_index=len(chunks),
                first_block=group[0].block.order,
                last_block=group[-1].block.order,
                token_count=sum(s.tokens for s in group),
            ))
    return chunks


if __name__ == "__main__":
    # python -m src.chunking <arxiv_id | path/to/page.html>
    import sys
    from pathlib import Path

    from src.arxiv_html import fetch_html, parse_blocks

    source = sys.argv[1]
    html = Path(source).read_text() if source.endswith(".html") else fetch_html(source)
    if html is None:
        sys.exit(f"arXiv has no HTML version of {source}")
    chunks = chunk_blocks(parse_blocks(html))
    for chunk in chunks:
        print(f"[{chunk.chunk_index}] {' > '.join(chunk.breadcrumb)}  "
              f"({chunk.token_count} tokens, blocks {chunk.first_block}-{chunk.last_block})")
        print(f"{chunk.text}\n")
    sizes = sorted(c.token_count for c in chunks)
    print(f"{len(chunks)} chunks; tokens min {sizes[0]}, median {sizes[len(sizes) // 2]}, max {sizes[-1]}")
