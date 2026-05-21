"""LangGraph state schema for the ACRA retention workflow."""

from typing import TypedDict


class RetentionState(TypedDict):
    """State that flows through the Strategist -> Auditor -> (loop) graph."""

    customer_id: str
    customer_name: str
    customer_email: str
    cancellation_reason: str
    customer_profile: dict
    playbook_policies: list[dict]
    proposed_offer: dict
    audit_approved: bool
    audit_feedback: str
    iteration_count: int
    final_email: str
    final_json: dict
