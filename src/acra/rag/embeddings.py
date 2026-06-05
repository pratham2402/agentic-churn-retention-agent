"""Embedding functions for ChromaDB using local sentence-transformers.

Provides two interfaces:
    1. ChromaDB-native: `get_embedding_function()` → ChromaDB EmbeddingFunction
    2. LangChain-compatible: `get_langchain_embeddings()` → LCEL Embeddings

Uses all-MiniLM-L6-v2 - a 384-dimensional model that runs entirely
on CPU with no API calls.
"""

from chromadb.utils import embedding_functions
from langchain_core.embeddings import Embeddings


def get_embedding_function() -> embedding_functions.SentenceTransformerEmbeddingFunction:
    """Create a ChromaDB-native sentence-transformer embedding function.

    Returns a callable compatible with ChromaDB's collection operations
    (add, query, update, upsert).
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
    )


class LangChainEmbeddingAdapter(Embeddings):
    """Adapter that wraps a ChromaDB embedding function for LangChain compatibility.

    LangChain vectorstores (like langchain_chroma.Chroma) require an Embeddings
    object with embed_documents() and embed_query(). ChromaDB's native
    SentenceTransformerEmbeddingFunction uses a different interface.
    This adapter bridges the two.
    """

    def __init__(self, chromadb_ef=None):
        if chromadb_ef is None:
            chromadb_ef = get_embedding_function()
        self._ef = chromadb_ef

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents using the ChromaDB embedding function.

        ChromaDB's __call__ accepts a list and returns a list of embeddings.
        """
        results = self._ef(texts)
        if isinstance(results, list) and len(results) > 0:
            if hasattr(results[0], 'tolist'):
                return [r.tolist() if hasattr(r, 'tolist') else r for r in results]
        return results

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        embeddings = self.embed_documents([text])
        result = embeddings[0]
        if hasattr(result, 'tolist'):
            return result.tolist()
        return result


def get_langchain_embeddings() -> LangChainEmbeddingAdapter:
    """Create a LangChain-compatible embedding function.

    Use this with langchain_chroma.Chroma and other LangChain vectorstores.
    """
    return LangChainEmbeddingAdapter(get_embedding_function())
