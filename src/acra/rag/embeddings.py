"""Embedding function for ChromaDB using local sentence-transformers (no API required)."""

from chromadb.utils import embedding_functions


def get_embedding_function() -> embedding_functions.SentenceTransformerEmbeddingFunction:
    """Create a local sentence-transformer embedding function for ChromaDB."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
    )
