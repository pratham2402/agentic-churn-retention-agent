"""Multi-vector RAG retriever for company retention policies.

Architecture (true multi-vector, per the LangChain textbook pattern):

    Each policy document is a PARENT. For each parent, we generate
    multiple CHILD documents (1 factual summary + 3 hypothetical questions).
    Children are embedded in ChromaDB. Parents are stored in a
    pickle-persisted byte store.

    At retrieval time:
        1. The query is embedded and matched against CHILD vectors
        2. Matching children reveal their parent policy_ids
        3. Full parent policy texts are fetched from the byte store
        4. Deduplication ensures each policy is returned once

This is the canonical pattern for multi-vector RAG where small,
targeted chunks (children) drive retrieval accuracy while large,
context-rich documents (parents) provide LLM context.

Customer profiles are retrieved via exact ID lookup (key-value),
which is the correct pattern for structured entity data where
semantic search adds no value.
"""

import json
import os
from langchain_core.documents import Document
from langchain_chroma import Chroma

from acra.rag.vector_store import (
    get_or_create_collections,
    load_parent_store,
    save_parent_store,
)
from acra.rag.embeddings import (
    get_embedding_function,
    get_langchain_embeddings,
)


class MultiVectorPolicyRetriever:
    """Multi-vector RAG retriever for the company retention playbook.

    Stores:
        - Children (summaries + hypothetical questions) → ChromaDB
        - Parents (full policy text) → dict[str, str] persisted via pickle

    Retrieves:
        - Vector similarity search against children
        - Maps matched children to parent policy_ids
        - Returns deduplicated full parent texts
    """

    def __init__(self):
        embedding_fn = get_embedding_function()
        (
            self._client,
            self._profile_collection,
            self._policy_children_collection,
            self._parent_store,
        ) = get_or_create_collections(embedding_fn=embedding_fn)

        lc_embeddings = get_langchain_embeddings()
        self._langchain_chroma = Chroma(
            client=self._client,
            collection_name="policy_children",
            embedding_function=lc_embeddings,
        )

    # ── Policy ingestion ──────────────────────────────────────────

    def add_policy(
        self,
        policy_id: str,
        parent_text: str,
        child_documents: list[Document],
    ) -> None:
        """Store a policy's parent text and its child documents.

        Args:
            policy_id: unique policy identifier (e.g., "POL-001")
            parent_text: full policy text returned on retrieval
            child_documents: list of Document objects, each with
                metadata containing at minimum {"policy_id": policy_id}
        """
        # Store parent in memory + persist
        self._parent_store[policy_id] = parent_text
        save_parent_store(self._parent_store)

        # Add children to ChromaDB via langchain_chroma
        if child_documents:
            self._langchain_chroma.add_documents(child_documents)

    # ── Policy retrieval ──────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Multi-vector retrieval: match children, return parents.

        Args:
            query: natural language query (e.g., cancellation reason + context)
            top_k: max number of child matches to consider (higher = broader recall)

        Returns:
            list of dicts with keys: policy_id, content, chunk_type
            Each policy appears at most once (deduplicated by policy_id).
        """
        if not self._parent_store:
            return []

        # Step 1: similarity search against children in ChromaDB
        matched_docs = self._langchain_chroma.similarity_search(
            query,
            k=min(top_k, len(self._parent_store) * 4),
        )

        # Step 2: extract unique policy_ids from matched children
        seen_ids: set[str] = set()
        unique_matches: list[tuple[str, str]] = []  # (policy_id, chunk_type)

        for doc in matched_docs:
            pid = doc.metadata.get("policy_id", "")
            if pid and pid not in seen_ids and pid in self._parent_store:
                seen_ids.add(pid)
                unique_matches.append(
                    (pid, doc.metadata.get("chunk_type", "unknown"))
                )

        # Step 3: fetch parent texts from byte store
        results: list[dict] = []
        for pid, chunk_type in unique_matches:
            parent_text = self._parent_store.get(pid, "")
            if parent_text:
                results.append({
                    "policy_id": pid,
                    "content": parent_text,
                    "chunk_type": chunk_type,
                })

        return results

    # ── Customer profile lookup (key-value, not vector) ───────────

    def lookup_customer(self, customer_id: str) -> dict:
        """Retrieve a customer profile by exact ID from ChromaDB.

        This is a key-value lookup, not a vector search. Customer profiles
        are structured entities — exact match is the correct access pattern.
        """
        result = self._profile_collection.get(
            ids=[f"profile-{customer_id}"],
            include=["documents", "metadatas"],
        )
        if result["ids"] and result["documents"]:
            doc = result["documents"][0]
            return json.loads(doc)
        return {}

    # ── Properties ─────────────────────────────────────────────────

    @property
    def parent_store(self) -> dict[str, str]:
        return self._parent_store

    @property
    def policy_count(self) -> int:
        return len(self._parent_store)

    @property
    def child_count(self) -> int:
        """Approximate count of child documents in the vector store."""
        try:
            return self._policy_children_collection.count()
        except Exception:
            return 0
