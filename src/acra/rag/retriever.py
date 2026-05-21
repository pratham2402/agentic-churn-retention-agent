"""Multi-vector RAG retriever for simultaneous profile and playbook lookup."""

import json
from acra.rag.vector_store import get_or_create_collections


class RetentionRetriever:
    """Retrieves customer profiles and company playbook policies simultaneously.

    This implements a multi-vector RAG approach: the customer profile JSON
    and the text playbook are stored in separate ChromaDB collections and
    queried in parallel to give the Strategist agent a complete picture.
    """

    def __init__(self):
        _, self.profiles, self.playbook = get_or_create_collections()

    def lookup_customer(self, customer_id: str) -> dict:
        """Retrieve a customer profile by exact ID match from ChromaDB."""
        result = self.profiles.get(
            ids=[f"profile-{customer_id}"],
            include=["documents", "metadatas"],
        )
        if result["ids"] and result["documents"]:
            doc = result["documents"][0]
            return json.loads(doc)
        return {}

    def search_playbook(self, query: str, n_results: int = 5) -> list[dict]:
        """Search the company playbook for relevant retention policies."""
        results = self.playbook.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas"],
        )
        policies = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                policies.append({
                    "policy_id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                })
        return policies

    def retrieve(self, customer_id: str, cancellation_reason: str) -> tuple[dict, list[dict]]:
        """Multi-vector retrieval: fetch profile and playbook in parallel."""
        profile = self.lookup_customer(customer_id)

        search_query = (
            f"Customer cancellation reason: {cancellation_reason}. "
            f"Customer tenure: {profile.get('tenure_months', 0)} months. "
            f"Customer plan: {profile.get('plan_name', 'unknown')}. "
            f"LTV: ${profile.get('lifetime_value_usd', 0)}."
        )
        policies = self.search_playbook(search_query)

        return profile, policies
