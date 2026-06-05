"""LangGraph workflow: autonomous 2-node actor-critic agent loop.

Graph topology:
    START → retrieval → strategist ⟷ tools      (ReAct loop)
                            ↓
                         [no tool_calls]
                            ↓
                         auditor → [approved?]
                            │           ├─ yes → finalize → END
                            │           └─ no → [under max?]
                            │                ├─ yes → strategist (retry)
                            │                └─ no  → finalize → END
                            ↑                    │
                            └────────────────────┘ (rejected, under max)

The Strategist is a fully autonomous tool-calling agent (ReAct pattern).
The Auditor is a programmatic policy enforcement node (zero LLM).
The conditional edges route based on tool_call presence and audit results.
"""

import os
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from acra.agent.state import RetentionState
from acra.agent.strategist import create_strategist_node, STRATEGIST_TOOLS
from acra.agent.auditor import create_auditor_node
from acra.agent.tools import _get_retriever

MAX_ITERATIONS = int(os.getenv("MAX_RETENTION_LOOP_ITERATIONS", "3"))


# ── Retrieval node ────────────────────────────────────────────────


def create_retrieval_node():
    """Node that performs initial data loading before the Strategist runs.

    Loads the customer profile and publishes it to state so the Auditor
    has access to it without needing to call tools.
    """
    def retrieval_node(state: RetentionState) -> dict:
        customer_id = state.get("customer_id", "")
        retriever = _get_retriever()

        profile = retriever.lookup_customer(customer_id)
        if not profile:
            raise ValueError(
                f"Customer profile not found for ID: {customer_id}. "
                f"Ensure the database has been seeded with: make seed"
            )

        return {
            "customer_profile": profile,
        }

    return retrieval_node


# ── Routing functions ─────────────────────────────────────────────


def route_after_strategist(
    state: RetentionState,
) -> Literal["tools", "auditor"]:
    """Route based on whether the Strategist's last message has tool calls.

    - tool_calls present → route to tools node (ReAct loop continues)
    - no tool_calls → route to auditor (agent is done, ready for review)
    """
    messages = state.get("messages", [])
    if not messages:
        return "auditor"

    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "auditor"


def route_after_auditor(
    state: RetentionState,
) -> Literal["strategist", "finalize"]:
    """Route based on audit result and iteration count.

    - approved → finalize
    - rejected AND under max iterations → strategist (retry with feedback)
    - rejected AND at max iterations → finalize (best available offer)
    """
    if state.get("audit_approved", False):
        return "finalize"

    iteration = state.get("iteration_count", 0)
    if iteration >= MAX_ITERATIONS:
        return "finalize"

    return "strategist"


# ── Finalize node ─────────────────────────────────────────────────


def finalize_node(state: RetentionState) -> dict:
    """Assemble the final RetentionPayload for database dispatch."""
    profile = state.get("customer_profile", {})
    offer = state.get("proposed_offer", {})

    final_json = {
        "customer_id": state.get("customer_id", ""),
        "customer_name": profile.get("name", ""),
        "original_reason": state.get("cancellation_reason", ""),
        "final_offer": offer,
        "audit_approved": state.get("audit_approved", False),
        "iterations_used": state.get("iteration_count", 0) + 1,
        "db_update": {
            "applied_discount": offer.get("discount_percent", 0),
            "discount_duration_months": offer.get("duration_months", 0),
            "offer_type": offer.get("offer_type", ""),
            "email_draft": offer.get("email_draft", ""),
            "justification": offer.get("justification", ""),
            "agent_reasoning": offer.get("reasoning", ""),
        },
    }

    final_email = offer.get("email_draft", "")

    return {
        "final_email": final_email,
        "final_result": final_json,
    }


# ── Graph construction ────────────────────────────────────────────


def build_workflow() -> StateGraph:
    """Build the ACRA retention workflow as a StateGraph.

    Nodes:
        retrieval   - loads customer profile into state
        strategist  - autonomous tool-calling ReAct agent
        tools       - ToolNode executing Strategist's tool calls
        auditor     - programmatic policy enforcement (9 checks)
        finalize    - assembles RetentionPayload

    Edges:
        START → retrieval → strategist
        strategist → tools (if tool_calls) or auditor (if done)
        tools → strategist (ReAct loop)
        auditor → finalize (approved/at cap) or strategist (retry)
        finalize → END
    """
    workflow = StateGraph(RetentionState)

    # Create nodes
    retrieval = create_retrieval_node()
    strategist = create_strategist_node()
    tools_node = ToolNode(STRATEGIST_TOOLS)
    auditor = create_auditor_node()

    # Add nodes
    workflow.add_node("retrieval", retrieval)
    workflow.add_node("strategist", strategist)
    workflow.add_node("tools", tools_node)
    workflow.add_node("auditor", auditor)
    workflow.add_node("finalize", finalize_node)

    # Add edges
    workflow.add_edge(START, "retrieval")
    workflow.add_edge("retrieval", "strategist")

    # Strategist's conditional routing (ReAct loop or proceed to audit)
    workflow.add_conditional_edges(
        "strategist",
        route_after_strategist,
        {
            "tools": "tools",
            "auditor": "auditor",
        },
    )
    workflow.add_edge("tools", "strategist")

    # Auditor's conditional routing (approve or loop back)
    workflow.add_conditional_edges(
        "auditor",
        route_after_auditor,
        {
            "strategist": "strategist",
            "finalize": "finalize",
        },
    )

    workflow.add_edge("finalize", END)

    return workflow


def compile_graph():
    """Build and compile the LangGraph workflow.

    Returns a compiled graph ready for .invoke() or .stream().
    """
    workflow = build_workflow()
    return workflow.compile()
