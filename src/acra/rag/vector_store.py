"""ChromaDB vector store setup with multi-collection architecture.

Collections:
    customer_profiles  — exact ID lookup for customer data (key-value)
    policy_children    — multi-vector embeddings of policy summaries and
                          hypothetical questions (dense vector search)

Parent Document Store:
    InMemoryByteStore (pickle-persisted) — stores full policy text,
    keyed by policy_id. Retrieved after child vector match.
"""

import os
import pickle
import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction

from acra.rag.embeddings import get_embedding_function


def _resolve_persist_dir() -> str:
    """Resolve the ChromaDB persist directory.

    If CHROMA_PERSIST_DIR is set as an absolute path, use it directly.
    If it's relative (or not set), resolve relative to the project root
    (directory containing pyproject.toml), not the current working directory.
    """
    env_dir = os.getenv("CHROMA_PERSIST_DIR")
    if env_dir and os.path.isabs(env_dir):
        return env_dir

    # Find project root: directory containing pyproject.toml
    # Walk up from this file's location
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        if os.path.exists(os.path.join(current, "pyproject.toml")):
            break
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    relative = env_dir if env_dir else "./chroma_data"
    return os.path.normpath(os.path.join(current, relative))


PERSIST_DIR = _resolve_persist_dir()

BYTE_STORE_PATH = os.path.join(PERSIST_DIR, "parent_policy_store.pkl")


def get_chroma_client() -> chromadb.PersistentClient:
    """Get or create a persistent ChromaDB client."""
    os.makedirs(PERSIST_DIR, exist_ok=True)
    return chromadb.PersistentClient(
        path=PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collections(
    embedding_fn: EmbeddingFunction | None = None,
):
    """Get or create all ChromaDB collections and the parent document store.

    Returns:
        tuple of (client, profile_collection, policy_children_collection, parent_store)
    """
    client = get_chroma_client()

    if embedding_fn is None:
        embedding_fn = get_embedding_function()

    profile_collection = client.get_or_create_collection(
        name="customer_profiles",
        embedding_function=embedding_fn,
        metadata={"description": "Customer account profiles for retention analysis"},
    )

    policy_children_collection = client.get_or_create_collection(
        name="policy_children",
        embedding_function=embedding_fn,
        metadata={
            "description": (
                "Multi-vector policy children: summaries and hypothetical "
                "questions that link to parent policy documents"
            ),
        },
    )

    parent_store = load_parent_store()

    return client, profile_collection, policy_children_collection, parent_store


def load_parent_store() -> dict[str, str]:
    """Load the parent policy document store from disk.

    Returns a plain dict[str, str] mapping policy_id -> full policy text.
    This is the canonical parent document store used by the retriever.
    """
    if os.path.exists(BYTE_STORE_PATH):
        try:
            with open(BYTE_STORE_PATH, "rb") as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError, OSError):
            pass
    return {}


def save_parent_store(store: dict[str, str]) -> None:
    """Persist the parent policy document store to disk."""
    os.makedirs(os.path.dirname(BYTE_STORE_PATH), exist_ok=True)
    with open(BYTE_STORE_PATH, "wb") as f:
        pickle.dump(store, f)


def reset_collections():
    """Delete and recreate all collections and the parent store.

    Useful for re-seeding with fresh data.
    """
    client = get_chroma_client()
    for name in ("customer_profiles", "policy_children"):
        try:
            client.delete_collection(name)
        except Exception:
            pass
    if os.path.exists(BYTE_STORE_PATH):
        os.remove(BYTE_STORE_PATH)
    return get_or_create_collections()
