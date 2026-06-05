"""LangChain @tool functions for the ACRA agent system.

These tools are exposed to the Strategist agent, enabling it to
autonomously gather customer context and search company policies.

Each tool is a typed, self-documenting function that the LLM can
invoke via function calling. The docstrings serve as the tool
descriptions that guide the agent's decision-making.
"""

from langchain_core.tools import tool
from acra.rag.retriever import MultiVectorPolicyRetriever

# Module-level retriever instance - created once and reused.
# The retriever manages its own ChromaDB connections internally.
_retriever: MultiVectorPolicyRetriever | None = None


def _get_retriever() -> MultiVectorPolicyRetriever:
    """Lazy-initialize and return the shared retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = MultiVectorPolicyRetriever()
    return _retriever


@tool
def get_customer_profile(customer_id: str) -> str:
    """Retrieve the full customer account profile by customer ID.

    Use this tool whenever you need to look up a customer's plan,
    tenure, monthly cost, LTV, feature usage, payment history, or
    any other account details. The customer_id is typically provided
    in the cancellation request.

    Args:
        customer_id: The customer's unique identifier (e.g., 'CUST-001')

    Returns:
        JSON string with all customer profile fields, or an empty
        object string '{}' if the customer is not found.
    """
    import json

    retriever = _get_retriever()
    profile = retriever.lookup_customer(customer_id)
    return json.dumps(profile, indent=2) if profile else "{}"


@tool
def search_retention_policies(query: str) -> str:
    """Search the company retention playbook for relevant policies.

    Uses multi-vector RAG to find policies relevant to the query.
    The search matches against policy summaries and hypothetical
    questions (not raw policy text), then returns the full text of
    matched parent policies.

    Use this tool to discover:
    - Maximum discount limits by customer tenure
    - LTV protection rules and revenue floors
    - Free month eligibility criteria
    - Plan downgrade preferences
    - Feature gap handling protocols
    - High-value customer treatment rules
    - Competitor price match policies

    Args:
        query: A natural language description of what policies you
               need (e.g., 'discount limits for 8-month customer'
               or 'what offers can I make to a high-LTV customer')

    Returns:
        Formatted string containing the full text of each matched
        policy, separated by delimiters. Returns empty string if
        no policies were matched.
    """
    retriever = _get_retriever()
    results = retriever.retrieve(query, top_k=5)

    if not results:
        return "No matching policies found."

    parts: list[str] = []
    for i, policy in enumerate(results, 1):
        parts.append(
            f"--- Policy {i}: {policy['policy_id']} ---\n"
            f"Match type: {policy['chunk_type']}\n"
            f"{policy['content']}\n"
        )

    return "\n".join(parts)
