from pathlib import Path

from src.arxiv_html import TextBlock, parse_blocks
from src.chunking import chunk_blocks, split_sentences

FIXTURES = Path(__file__).parent / "fixtures" / "arxiv_html"

# Small sizes keep the examples readable; tokens are counted as words so tests don't need the tokenizer.
SIZES = {"target_tokens": 50, "max_tokens": 60, "overlap_tokens": 10}


def words(text: str) -> int:
    return len(text.split())


def sentence(label: str, length: int = 10) -> str:
    return " ".join([label] + ["word"] * (length - 2) + ["end."])


def make_blocks(*paragraphs: tuple[list[str], list[str]], section_depth: int = 1) -> list[TextBlock]:
    """paragraphs: (breadcrumb, sentences) pairs, in reading order."""
    return [
        TextBlock(
            text=" ".join(sentences),
            breadcrumb=breadcrumb,
            order=order,
            section_path=tuple(breadcrumb[:section_depth]),
        )
        for order, (breadcrumb, sentences) in enumerate(paragraphs)
    ]


def chunk(blocks: list[TextBlock], **sizes):
    return chunk_blocks(blocks, count_tokens=words, **{**SIZES, **sizes})


def labels(n: int, prefix: str = "S") -> list[str]:
    return [sentence(f"{prefix}{i}") for i in range(n)]


class TestSplitSentences:
    def test_abbreviation_before_lowercase_or_number_does_not_end_a_sentence(self):
        text = "See Eq. (2) for details. Accuracy rose on avg. across tasks."
        assert split_sentences(text) == ["See Eq. (2) for details.", "Accuracy rose on avg. across tasks."]

    def test_abbreviation_before_capital_still_ends_a_sentence(self):
        assert split_sentences("Results improve on avg. The gap closes later.") == [
            "Results improve on avg.",
            "The gap closes later.",
        ]

    def test_periods_inside_math_do_not_end_a_sentence(self):
        text = "We get $12.54\\times$ less error. Then $$a=b.c$$ (1) holds for all $x.y$ here. Done."
        assert split_sentences(text) == [
            "We get $12.54\\times$ less error.",
            "Then $$a=b.c$$ (1) holds for all $x.y$ here.",
            "Done.",
        ]

    def test_each_line_is_split_on_its_own(self):
        assert split_sentences("Two kinds:\n- first item\n- second item") == [
            "Two kinds:",
            "- first item",
            "- second item",
        ]


class TestChunkBlocks:
    def test_no_blocks_gives_no_chunks(self):
        assert chunk([]) == []

    def test_short_section_is_one_chunk_with_paragraphs_kept_apart(self):
        a, b = sentence("Alpha"), sentence("Beta")
        chunks = chunk(make_blocks((["1 Intro"], [a]), (["1 Intro"], [b])))

        assert len(chunks) == 1
        assert chunks[0].text == f"{a}\n\n{b}"
        assert chunks[0].breadcrumb == ["1 Intro"]
        assert (chunks[0].first_block, chunks[0].last_block, chunks[0].token_count) == (0, 1, 20)

    def test_list_lines_stay_on_their_own_lines(self):
        chunks = chunk(make_blocks((["1 Intro"], ["Two kinds:\n- first item\n- second item"])))
        assert chunks[0].text == "Two kinds:\n- first item\n- second item"

    def test_chunks_never_cross_top_level_sections(self):
        intro, method = sentence("Intro"), sentence("Method")
        chunks = chunk(make_blocks((["1 Intro"], [intro]), (["2 Method"], [method])))

        assert [c.text for c in chunks] == [intro, method]
        assert [c.breadcrumb for c in chunks] == [["1 Intro"], ["2 Method"]]
        assert [c.chunk_index for c in chunks] == [0, 1]

    def test_long_paragraph_is_cut_between_sentences_within_max(self):
        sentences = labels(12)
        chunks = chunk(make_blocks((["1 Intro"], sentences)))

        assert len(chunks) > 1
        assert all(c.token_count <= 60 for c in chunks)
        pieces = [s for c in chunks for s in split_sentences(c.text)]
        assert set(pieces) == set(sentences)  # whole sentences only, and none lost

    def test_section_is_split_evenly_without_a_tiny_last_chunk(self):
        # 130 words: three chunks of ~43 new words, not 50 + 50 + 30.
        chunks = chunk(make_blocks((["1 Intro"], labels(13))))
        new_sizes = [chunks[0].token_count] + [c.token_count - 10 for c in chunks[1:]]

        assert len(chunks) == 3
        assert max(new_sizes) - min(new_sizes) <= 10

    def test_next_chunk_starts_with_the_previous_chunks_last_sentence(self):
        chunks = chunk(make_blocks((["1 Intro"], labels(12))))

        for previous, current in zip(chunks, chunks[1:]):
            assert split_sentences(current.text)[0] == split_sentences(previous.text)[-1]

    def test_last_sentence_longer_than_overlap_is_still_carried_over(self):
        sentences = [sentence("Long", 20), sentence("Long2", 20), sentence("Long3", 20), sentence("Long4", 20)]
        chunks = chunk(make_blocks((["1 Intro"], sentences)))

        assert len(chunks) == 2
        assert split_sentences(chunks[1].text)[0] == split_sentences(chunks[0].text)[-1]

    def test_chunk_prefers_to_end_at_a_paragraph(self):
        p1 = labels(3, "A")                       # 30 words
        p2 = [sentence("Bshort", 3)] + labels(3, "B")  # 33 words
        p3 = labels(4, "C")                       # 40 words
        chunks = chunk(make_blocks((["1 Intro"], p1), (["1 Intro"], p2), (["1 Intro"], p3)))

        # The 3-word sentence would still fit in the first chunk, but it ends at paragraph 1 instead.
        assert chunks[0].text == " ".join(p1)
        assert chunks[1].text == f"{p1[-1]}\n\n{' '.join(p2)}"

    def test_sentence_longer_than_max_is_kept_whole_in_its_own_chunk(self):
        huge = sentence("Huge", 160)  # also too long (> 150) to be carried into the next chunk
        chunks = chunk(make_blocks((["1 Intro"], labels(3, "A") + [huge] + labels(3, "B"))))

        assert huge in [c.text for c in chunks]
        assert all(huge not in c.text or c.text == huge for c in chunks)

    def test_breadcrumb_is_what_all_paragraphs_in_the_chunk_share(self):
        chunks = chunk(make_blocks(
            (["4 Method", "4.1 Setup"], labels(3, "A")),
            (["4 Method", "4.2 Scales"], labels(4, "B")),
            (["4 Method", "4.2 Scales"], labels(4, "C")),
        ))

        assert [c.breadcrumb for c in chunks] == [
            ["4 Method", "4.1 Setup"],
            ["4 Method"],                # overlap sentence from 4.1, then 4.2
            ["4 Method", "4.2 Scales"],
        ]


def test_real_paper_chunks_cover_every_sentence_without_crossing_sections():
    blocks = parse_blocks((FIXTURES / "dualquant_2610.13767v1.html").read_text())
    sizes = {"target_tokens": 100, "max_tokens": 120, "overlap_tokens": 15}
    chunks = chunk_blocks(blocks, count_tokens=words, **sizes)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert c.token_count <= 120 or len(split_sentences(c.text)) == 1
        covered = blocks[c.first_block : c.last_block + 1]
        assert len({b.section_path for b in covered}) == 1
        assert all(b.breadcrumb[: len(c.breadcrumb)] == c.breadcrumb for b in covered)

    chunk_sentences = {s for c in chunks for s in split_sentences(c.text)}
    for block in blocks:
        assert set(split_sentences(block.text)) <= chunk_sentences
