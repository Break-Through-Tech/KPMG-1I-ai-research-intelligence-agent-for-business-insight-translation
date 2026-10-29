from time import perf_counter

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None

def get_model(timings: dict[str, float] | None = None):
    global _model
    if _model is None:
        started = perf_counter()
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        if timings is not None:
            timings["embedding_model_initialization_seconds"] = perf_counter() - started
    return _model

def chunk_paper(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def embed_chunks(chunks: list[str], timings: dict[str, float] | None = None):
    model = get_model(timings)
    return model.encode(chunks, show_progress_bar=False).tolist()