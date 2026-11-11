"""The model tests load the real Nomic and BM25 models (downloaded on first run, ~0.5 GB)."""

import numpy as np

from src.embeddings import (
    DENSE_VECTOR_SIZE,
    count_tokens,
    dense_chunk_text,
    embed_documents,
    embed_query,
    embed_sparse,
    embed_sparse_query,
    get_dense_model,
    paper_text,
    sparse_chunk_text,
)

TITLE = "DualQuant: Calibration-Free Dual-Scale Quantization"
BREADCRUMB = ["4 Methodology", "4.2 Dual scales"]
TEXT = "We quantize both weights and activations to 4 bits."


class TestEmbeddingInputs:
    def test_dense_chunk_text_adds_paper_and_section(self):
        assert dense_chunk_text(TITLE, BREADCRUMB, TEXT) == (
            f"Paper: {TITLE}\nSection: 4 Methodology > 4.2 Dual scales\n\n{TEXT}"
        )

    def test_dense_chunk_text_without_breadcrumb_has_no_section_line(self):
        assert dense_chunk_text(TITLE, [], TEXT) == f"Paper: {TITLE}\n\n{TEXT}"

    def test_sparse_chunk_text_adds_section_but_not_title(self):
        assert sparse_chunk_text(BREADCRUMB, TEXT) == f"4 Methodology > 4.2 Dual scales\n\n{TEXT}"
        assert sparse_chunk_text([], TEXT) == TEXT

    def test_paper_text_is_title_then_abstract(self):
        assert paper_text(TITLE, "We introduce DualQuant.") == f"{TITLE}\n\nWe introduce DualQuant."


def cosine(a, b) -> float:
    return float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))


class TestDenseEncoder:
    def test_one_768_dimension_vector_per_text(self):
        vectors = embed_documents([TEXT, "Another document."])
        assert len(vectors) == 2
        assert all(len(v) == DENSE_VECTOR_SIZE and isinstance(v[0], float) for v in vectors)

    def test_documents_are_embedded_with_the_document_prefix(self):
        [vector] = embed_documents([TEXT])
        [expected] = get_dense_model().embed([f"search_document: {TEXT}"])
        assert np.allclose(vector, expected)

    def test_vectors_come_back_in_input_order_whatever_the_lengths(self):
        texts = [f"Paragraph about topic {i}. " * (20 - 2 * i) for i in range(10)]  # longest first
        expected = [get_dense_model().embed([f"search_document: {t}"]).__next__() for t in texts]
        assert all(np.allclose(v, e, atol=1e-5) for v, e in zip(embed_documents(texts), expected))

    def test_related_documents_are_closer_than_unrelated_ones(self):
        quant, quant2, cooking = embed_documents([
            "Post-training quantization of LLM weights to 4-bit integers.",
            "Low-bit weight quantization for large language model inference.",
            "Simmer the tomatoes with garlic and basil for twenty minutes.",
        ])
        assert cosine(quant, quant2) > cosine(quant, cooking)


class TestQueryEncoders:
    QUESTION = "How can LLM weights be quantized to 4 bits?"

    def test_question_is_embedded_with_the_query_prefix(self):
        [expected] = get_dense_model().embed([f"search_query: {self.QUESTION}"])
        vector = embed_query(self.QUESTION)
        assert len(vector) == DENSE_VECTOR_SIZE and isinstance(vector[0], float)
        assert np.allclose(vector, expected)
        assert not np.allclose(vector, embed_documents([self.QUESTION])[0], atol=1e-3)

    def test_question_is_closer_to_a_relevant_document(self):
        relevant, unrelated = embed_documents([
            "Post-training quantization of LLM weights to 4-bit integers.",
            "Simmer the tomatoes with garlic and basil for twenty minutes.",
        ])
        question = embed_query(self.QUESTION)
        assert cosine(question, relevant) > cosine(question, unrelated)

    def test_sparse_question_terms_all_weigh_one_however_often_they_repeat(self):
        once = embed_sparse_query("quantization of weights")
        repeated = embed_sparse_query("quantization quantization quantization of weights")
        assert sorted(once.indices) == sorted(repeated.indices)
        assert once.values == [1.0] * len(once.indices)

    def test_sparse_question_terms_match_document_terms(self):
        [document] = embed_sparse(["Activation quantization for language models."])
        assert set(embed_sparse_query("quantization").indices) <= set(document.indices)


class TestSparseEncoder:
    def test_shared_words_share_indices(self):
        a, b = embed_sparse(["Quantization of weights.", "Activation quantization."])
        assert set(a.indices) & set(b.indices)
        assert all(v > 0 for v in a.values + b.values)

    def test_stopwords_only_text_has_no_terms(self):
        [vector] = embed_sparse(["the of and"])
        assert vector.indices == []


class TestTokenCount:
    def test_counts_dense_tokens_without_special_tokens(self):
        assert count_tokens("") == 0
        text = "For a linear layer computing $\\mathbf{X}\\mathbf{W}^{\\top}$ with activation matrix."
        assert count_tokens(text) == 28  # LaTeX is token-heavy: \mathbf{X} alone is 5 tokens
