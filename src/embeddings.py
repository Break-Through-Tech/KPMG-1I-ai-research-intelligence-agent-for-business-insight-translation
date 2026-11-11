from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model

def embed_chunks(chunks: list[str]):
    model = get_model()
    return model.encode(chunks, show_progress_bar=False).tolist()