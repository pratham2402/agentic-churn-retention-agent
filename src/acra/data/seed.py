"""Seed script to populate ChromaDB with customer profiles and company playbook."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv

load_dotenv()

from acra.rag.vector_store import reset_collections
from acra.data.customer_profiles import CUSTOMER_PROFILES
from acra.data.playbook import (
    RETENTION_PLAYBOOK,
    HIGH_VALUE_CUSTOMER_POLICIES,
    RISKY_CUSTOMER_FLAGS,
)


def seed_customer_profiles(collection) -> None:
    """Seed the customer profiles collection with sample data."""
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

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Seeded {len(ids)} customer profiles into 'customer_profiles' collection.")


def seed_company_playbook(collection) -> None:
    """Seed the company playbook collection with retention policies."""
    all_policies = RETENTION_PLAYBOOK + HIGH_VALUE_CUSTOMER_POLICIES + RISKY_CUSTOMER_FLAGS

    ids = []
    documents = []
    metadatas = []

    for policy in all_policies:
        ids.append(policy["policy_id"])
        documents.append(policy["content"])
        metadatas.append({
            "policy_id": policy["policy_id"],
            "category": policy.get("category", "general"),
            "title": policy.get("title", ""),
        })

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Seeded {len(ids)} policies into 'company_playbook' collection.")


def main():
    """Reset and seed all ChromaDB collections."""
    client, profile_collection, playbook_collection = reset_collections()
    seed_customer_profiles(profile_collection)
    seed_company_playbook(playbook_collection)
    print("Database seeding complete.")


if __name__ == "__main__":
    main()
