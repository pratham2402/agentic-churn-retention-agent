"""LangGraph workflow: 2-node cycle (Strategist <-> Auditor) with loop guard.

Workflow:
    START -> Strategist -> Auditor -> [approved?]
                                        YES -> END
                                        NO  -> Strategist (retry, max N iterations)
"""

import os
from typing import Literal

from langgraph.graph import StateGraph, START, END

from acra.agent.state import RetentionState
from acra.agent.strategist import create_strategist_node
from acra.agent.auditor import create_auditor_node
from acra.rag.retriever import RetentionRetriever


MAX_ITERATIONS = int(os.getenv("MAX_RETENTION_LOOP_ITERATIONS", "3"))


def create_retrieval_node():
    """Node that performs RAG lookup before the strategist runs."""
    retriever = RetentionRetriever()

    def retrieval_node(state: RetentionState) -> dict:
        profile, policies = retriever.retrieve(
            customer_id=state["customer_id"],
            cancellation_reason=state["cancellation_reason"],
        )
        if not profile:
            raise ValueError(
                f"Customer profile not found for ID: {state['customer_id']}"
            )
        return {
            "customer_profile": profile,
            "playbook_policies": policies,
            "customer_name": profile.get("name", ""),
            "customer_email": profile.get("email", ""),
        }

    return retrieval_node


def build_workflow() -> StateGraph:
    """Build the ACRA retention workflow graph.

    Graph topology:
        START -> retrieval -> strategist -> auditor
                           ^                  |
                           |     (rejected)   |
                           +------------------+
                                  (approved) -> END
    """
    workflow = StateGraph(RetentionState)

    retrieval = create_retrieval_node()
    strategist = create_strategist_node()
    auditor = create_auditor_node()

    workflow.add_node("retrieval", retrieval)
    workflow.add_node("strategist", strategist)
    workflow.add_node("auditor", auditor)

    workflow.add_edge(START, "retrieval")
    workflow.add_edge("retrieval", "strategist")
    workflow.add_edge("strategist", "auditor")

    workflow.add_conditional_edges(
        "auditor",
        route_after_audit,
        {
            "strategist": "strategist",
            "finalize": "finalize",
        },
    )

    workflow.add_node("finalize", finalize_node)
    workflow.add_edge("finalize", END)

    return workflow


def route_after_audit(state: RetentionState) -> Literal["strategist", "finalize"]:
    """Conditional routing: if offer is rejected and under max iterations, loop back."""
    if state.get("audit_approved", False):
        return "finalize"

    iteration = state.get("iteration_count", 0)
    if iteration >= MAX_ITERATIONS - 1:
        return "finalize"

    return "strategist"


def finalize_node(state: RetentionState) -> dict:
    """Prepare the final output: email and JSON payload for the database."""
    return {
        "final_email": state.get("proposed_offer", {}).get("email_draft", ""),
        "final_json": {
            "customer_id": state["customer_id"],
            "customer_name": state.get("customer_name", ""),
            "original_reason": state.get("cancellation_reason", ""),
            "final_offer": state.get("proposed_offer", {}),
            "audit_approved": state.get("audit_approved", False),
            "iterations_used": state.get("iteration_count", 0) + 1,
            "db_update": {
                "applied_discount": state.get("proposed_offer", {}).get("discount_percent", 0),
                "discount_duration_months": state.get("proposed_offer", {}).get("duration_months", 0),
                "offer_type": state.get("proposed_offer", {}).get("offer_type", ""),
            },
        },
    }


def compile_graph():
    """Build and compile the LangGraph workflow."""
    workflow = build_workflow()
    return workflow.compile()
