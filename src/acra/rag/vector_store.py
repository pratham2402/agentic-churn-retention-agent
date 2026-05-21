"""ChromaDB vector store setup and management for ACRA."""

import os
import chromadb
from chromadb.config import Settings

from acra.rag.embeddings import get_embedding_function


def get_chroma_client() -> chromadb.PersistentClient:
    """Get or create a persistent ChromaDB client."""
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collections():
    """Get or create the customer profiles and company playbook collections."""
    client = get_chroma_client()
    embedding_fn = get_embedding_function()

    profile_collection = client.get_or_create_collection(
        name="customer_profiles",
        embedding_function=embedding_fn,
        metadata={"description": "Customer account profiles for retention analysis"},
    )

    playbook_collection = client.get_or_create_collection(
        name="company_playbook",
        embedding_function=embedding_fn,
        metadata={"description": "Company retention policies, discount rules, and guardrails"},
    )

    return client, profile_collection, playbook_collection


def reset_collections():
    """Delete and recreate both collections. Useful for re-seeding."""
    client = get_chroma_client()
    try:
        client.delete_collection("customer_profiles")
    except Exception:
        pass
    try:
        client.delete_collection("company_playbook")
    except Exception:
        pass
    return get_or_create_collections()
