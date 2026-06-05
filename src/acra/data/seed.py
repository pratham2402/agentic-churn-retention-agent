"""Seed script to populate ChromaDB with multi-vector policy chunks and
customer profiles.

Multi-vector seeding process:
    1. For each policy, use an LLM to generate 4 child documents:
       - 1 factual summary
       - 3 hypothetical questions
    2. Store children in the policy_children ChromaDB collection
       (each child gets its own embedding vector)
    3. Store full parent policy text in the pickle-persisted byte store
    4. Store customer profiles in the customer_profiles collection
       (exact ID lookup - not vector search)
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from acra.rag.vector_store import reset_collections, save_parent_store
from acra.rag.retriever import MultiVectorPolicyRetriever
from acra.rag.chunker import generate_policy_children
from acra.data.customer_profiles import CUSTOMER_PROFILES
from acra.data.playbook import (
    RETENTION_PLAYBOOK,
    HIGH_VALUE_CUSTOMER_POLICIES,
    RISKY_CUSTOMER_FLAGS,
)


def seed_customer_profiles(retriever: MultiVectorPolicyRetriever) -> None:
    """Seed the customer profiles collection with sample data.

    Profiles are stored as JSON documents with exact ID lookup.
    Each profile document has one embedding vector (used only for
    the collection metadata, not for semantic search).
    """
    ids = []
    documents = []
    metadatas = []

    for profile in CUSTOMER_PROFILES:
        ids.append(f"profile-{profile['customer_id']}")
        documents.append(json.dumps(profile))
        metadatas.append({
            "customer_id": profile["customer_id"],
            "tenure_months": profile["tenure_months"],
            "plan_name": profile["plan_name"],
            "lifetime_value_usd": profile["lifetime_value_usd"],
        })

    retriever._profile_collection.add(
        ids=ids, documents=documents, metadatas=metadatas
    )
    print(f"  Seeded {len(ids)} customer profiles.")


def seed_policies(retriever: MultiVectorPolicyRetriever) -> None:
    """Seed the multi-vector policy store.

    For each policy:
        1. Use LLM to generate 1 summary + 3 hypothetical questions
        2. Store children in ChromaDB (each gets its own embedding)
        3. Store parent text in the byte store
    """
    all_policies = (
        RETENTION_PLAYBOOK + HIGH_VALUE_CUSTOMER_POLICIES + RISKY_CUSTOMER_FLAGS
    )

    if not os.getenv("DEEPSEEK_API_KEY"):
        print("  Warning: DEEPSEEK_API_KEY not set. Using basic chunking fallback.")
        llm = None
    else:
        llm = ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            temperature=0.1,
            max_tokens=512,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )

    total_children = 0
    for policy in all_policies:
        policy_id = policy["policy_id"]
        policy_title = policy.get("title", policy_id)
        parent_text = policy["content"]

        try:
            children = generate_policy_children(
                policy_id=policy_id,
                policy_title=policy_title,
                policy_content=parent_text,
                llm=llm,
            )
        except Exception as e:
            print(f"  Warning: LLM chunking failed for {policy_id}: {e}")
            # Fallback: use the policy title as summary and generate basic questions
            from langchain_core.documents import Document
            children = [
                Document(
                    page_content=f"[{policy_id}] {policy_title}: {parent_text[:300]}",
                    metadata={"policy_id": policy_id, "chunk_type": "summary",
                              "policy_title": policy_title},
                ),
                Document(
                    page_content=f"[{policy_id}] What are the rules for {policy_title.lower()}?",
                    metadata={"policy_id": policy_id, "chunk_type": "hypothetical_question",
                              "question_index": 0, "policy_title": policy_title},
                ),
                Document(
                    page_content=f"[{policy_id}] What limits does {policy_id} impose?",
                    metadata={"policy_id": policy_id, "chunk_type": "hypothetical_question",
                              "question_index": 1, "policy_title": policy_title},
                ),
                Document(
                    page_content=f"[{policy_id}] Who is eligible under {policy_id}?",
                    metadata={"policy_id": policy_id, "chunk_type": "hypothetical_question",
                              "question_index": 2, "policy_title": policy_title},
                ),
            ]

        retriever.add_policy(policy_id, parent_text, children)
        total_children += len(children)
        print(f"  {policy_id}: {len(children)} children generated "
              f"({', '.join(d.metadata.get('chunk_type', '?') for d in children)})")

    print(f"  Total: {len(all_policies)} policies, {total_children} child vectors")


def main():
    """Reset and seed all collections with multi-vector RAG architecture."""
    print("Resetting collections...")
    reset_collections()

    # Create retriever (initializes all collections and byte store)
    retriever = MultiVectorPolicyRetriever()

    print("\nSeeding customer profiles...")
    seed_customer_profiles(retriever)

    print("\nGenerating multi-vector policy chunks...")
    seed_policies(retriever)

    print(f"\nDatabase seeding complete.")
    print(f"  Policies in store: {retriever.policy_count}")
    print(f"  Child vectors: ~{retriever.child_count}")
    print(f"  Profiles: {len(CUSTOMER_PROFILES)}")


if __name__ == "__main__":
    main()
