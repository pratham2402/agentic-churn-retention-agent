"""LangGraph state schema for the ACRA retention workflow.

Uses MessagesState as the base, which provides `messages` with the
`add_messages` reducer for automatic conversation history management.
The extended fields track the workflow-specific data flowing through
the Strategist -> Auditor -> (loop) graph.
"""

from typing import Annotated, Any
from langgraph.graph import MessagesState


class RetentionState(MessagesState):
    """State that flows through the ACRA retention agent graph.

    Inherits from MessagesState:
        messages: list[AnyMessage]  — full conversation history with
            automatic add_messages reducer for append-only semantics.

    Extended fields for workflow orchestration:
        customer_id:          target customer identifier
        cancellation_reason:  the reason the customer gave for cancelling
        customer_profile:     full profile dict from ChromaDB lookup
        proposed_offer:       RetentionOffer.model_dump() from Strategist
        audit_approved:       set by Auditor after programmatic checks
        audit_feedback:       violation details when rejected
        iteration_count:      incremented each time Auditor rejects
        final_email:          the email draft for the final offer
        final_result:         assembled RetentionPayload at workflow end
    """

    customer_id: str
    cancellation_reason: str
    customer_profile: dict
    proposed_offer: dict
    audit_approved: bool
    audit_feedback: str
    iteration_count: int
    final_email: str
    final_result: dict
