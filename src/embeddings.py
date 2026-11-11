"""Text encoders for the vector store.

- Dense: Nomic Embed Text v1.5 (768 dimensions), run locally through fastembed.
- Sparse: BM25 through fastembed. It only produces term frequencies; the Qdrant collection
  must apply IDF to its sparse vector.

The MiniLM model below still backs the old `papers` collection and the agent until they move over.
"""

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client.models import SparseVector
from sentence_transformers import SentenceTransformer

DENSE_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
DENSE_VECTOR_SIZE = 768
# Nomic is trained with these task prefixes; documents and queries must use the matching one.
DOCUMENT_PREFIX = "search_document: "
SPARSE_MODEL_NAME = "Qdrant/bm25"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_dense_model: TextEmbedding | None = None
_sparse_model: SparseTextEmbedding | None = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_chunks(chunks: list[str]):
    model = get_model()
    return model.encode(chunks, show_progress_bar=False).tolist()


def get_dense_model() -> TextEmbedding:
    global _dense_model
    if _dense_model is None:
        _dense_model = TextEmbedding(DENSE_MODEL_NAME)
    return _dense_model


def get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(SPARSE_MODEL_NAME)
    return _sparse_model


def count_tokens(text: str) -> int:
    """Tokens the dense model sees for text, without special tokens or prefix."""
    return len(get_dense_model().model.tokenizer.encode(text, add_special_tokens=False).ids)


def paper_text(title: str, abstract: str) -> str:
    """Dense input for a paper's high-level record."""
    return f"{title}\n\n{abstract}"


def dense_chunk_text(title: str, breadcrumb: list[str], text: str) -> str:
    """Dense input for a chunk: the paper and section give it context the chunk text may lack."""
    header = f"Paper: {title}"
    if breadcrumb:
        header += f"\nSection: {' > '.join(breadcrumb)}"
    return f"{header}\n\n{text}"


def sparse_chunk_text(breadcrumb: list[str], text: str) -> str:
    """Sparse input for a chunk. No title: BM25 would match every chunk of a paper on its title words."""
    return f"{' > '.join(breadcrumb)}\n\n{text}" if breadcrumb else text


def embed_documents(texts: list[str]) -> list[list[float]]:
    return [vector.tolist() for vector in get_dense_model().embed([DOCUMENT_PREFIX + t for t in texts])]


def embed_sparse(texts: list[str]) -> list[SparseVector]:
    return [
        SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
        for e in get_sparse_model().embed(texts)
    ]


if __name__ == "__main__":
    # python -m src.embeddings <arxiv_id | path/to/page.html> [number of chunks to show]
    import sys
    from pathlib import Path

    from bs4 import BeautifulSoup

    from src.arxiv_html import fetch_html, parse_blocks
    from src.chunking import chunk_blocks

    source = sys.argv[1]
    shown = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    html = Path(source).read_text() if source.endswith(".html") else fetch_html(source)
    if html is None:
        sys.exit(f"arXiv has no HTML version of {source}")
    heading = BeautifulSoup(html, "html.parser").find(class_="ltx_title_document")
    title = " ".join(heading.get_text(" ").split()) if heading else source

    for chunk in chunk_blocks(parse_blocks(html))[:shown]:
        dense_input = dense_chunk_text(title, chunk.breadcrumb, chunk.text)
        sparse_input = sparse_chunk_text(chunk.breadcrumb, chunk.text)
        [dense] = embed_documents([dense_input])
        [sparse] = embed_sparse([sparse_input])
        print(f"===== chunk {chunk.chunk_index} =====")
        print(f"--- dense input ({count_tokens(DOCUMENT_PREFIX + dense_input)} tokens, prefix included) ---")
        print(f"{DOCUMENT_PREFIX}{dense_input}\n")
        print(f"--- sparse input ---\n{sparse_input}\n")
        print(f"dense output: {len(dense)} floats, first 5 {[round(x, 3) for x in dense[:5]]}")
        print(f"sparse output: {len(sparse.indices)} terms, e.g. "
              f"{list(zip(sparse.indices[:3], [round(v, 3) for v in sparse.values[:3]]))}\n")
