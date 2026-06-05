"""Tests for the LangGraph workflow, routing logic, and finalization."""

import pytest
from langchain_core.messages import AIMessage

from acra.agent.graph import (
    route_after_strategist,
    route_after_auditor,
    build_workflow,
    finalize_node,
)
from acra.agent.state import RetentionState


class TestStrategistRouting:
    def test_route_with_tool_calls_goes_to_tools(self):
        msg = AIMessage(
            content="",
            tool_calls=[{
                "name": "get_customer_profile",
                "args": {"customer_id": "CUST-001"},
                "id": "call_1",
            }],
        )
        state: RetentionState = {
            "messages": [msg],
            "customer_id": "CUST-001",
            "cancellation_reason": "Too expensive",
            "customer_profile": {},
            "proposed_offer": {},
            "audit_approved": False,
            "audit_feedback": "",
            "iteration_count": 0,
            "final_email": "",
            "final_result": {},
        }
        assert route_after_strategist(state) == "tools"

    def test_route_without_tool_calls_goes_to_auditor(self):
        msg = AIMessage(content='{"discount_percent": 30, "duration_months": 6, '
                         '"offer_type": "discount", "justification": "ok", '
                         '"reasoning": "thought", "email_draft": "Hi"}')
        state: RetentionState = {
            "messages": [msg],
            "customer_id": "CUST-001",
            "cancellation_reason": "Too expensive",
            "customer_profile": {},
            "proposed_offer": {},
            "audit_approved": False,
            "audit_feedback": "",
            "iteration_count": 0,
            "final_email": "",
            "final_result": {},
        }
        assert route_after_strategist(state) == "auditor"

    def test_route_empty_messages_goes_to_auditor(self):
        state: RetentionState = {
            "messages": [],
            "customer_id": "CUST-001",
            "cancellation_reason": "Too expensive",
            "customer_profile": {},
            "proposed_offer": {},
            "audit_approved": False,
            "audit_feedback": "",
            "iteration_count": 0,
            "final_email": "",
            "final_result": {},
        }
        assert route_after_strategist(state) == "auditor"


class TestAuditorRouting:
    def test_route_approved_goes_to_finalize(self):
        state: RetentionState = {
            "messages": [],
            "customer_id": "CUST-001",
            "cancellation_reason": "Too expensive",
            "customer_profile": {},
            "proposed_offer": {},
            "audit_approved": True,
            "audit_feedback": "Approved",
            "iteration_count": 0,
            "final_email": "",
            "final_result": {},
        }
        assert route_after_auditor(state) == "finalize"

    def test_route_rejected_loops_back(self):
        state: RetentionState = {
            "messages": [],
            "customer_id": "CUST-001",
            "cancellation_reason": "Too expensive",
            "customer_profile": {},
            "proposed_offer": {},
            "audit_approved": False,
            "audit_feedback": "Discount too high",
            "iteration_count": 0,
            "final_email": "",
            "final_result": {},
        }
        assert route_after_auditor(state) == "strategist"

    def test_route_max_iterations_forces_finalize(self):
        state: RetentionState = {
            "messages": [],
            "customer_id": "CUST-001",
            "cancellation_reason": "Too expensive",
            "customer_profile": {},
            "proposed_offer": {},
            "audit_approved": False,
            "audit_feedback": "Still invalid",
            "iteration_count": 3,
            "final_email": "",
            "final_result": {},
        }
        assert route_after_auditor(state) == "finalize"


class TestFinalizeNode:
    def test_finalize_assembles_payload(self):
        state: RetentionState = {
            "messages": [],
            "customer_id": "CUST-001",
            "cancellation_reason": "Too expensive",
            "customer_profile": {
                "name": "Alice Johnson",
                "email": "alice@example.com",
            },
            "proposed_offer": {
                "discount_percent": 30,
                "duration_months": 6,
                "offer_type": "discount",
                "justification": "Within limits",
                "reasoning": "test reasoning",
                "email_draft": "Dear Alice...",
            },
            "audit_approved": True,
            "audit_feedback": "Approved",
            "iteration_count": 0,
            "final_email": "",
            "final_result": {},
        }

        result = finalize_node(state)

        assert result["final_email"] == "Dear Alice..."
        assert result["final_result"]["customer_id"] == "CUST-001"
        assert result["final_result"]["audit_approved"] is True
        assert result["final_result"]["db_update"]["applied_discount"] == 30

    def test_finalize_handles_empty_offer(self):
        state: RetentionState = {
            "messages": [],
            "customer_id": "CUST-002",
            "cancellation_reason": "Unknown",
            "customer_profile": {},
            "proposed_offer": {},
            "audit_approved": False,
            "audit_feedback": "Failed",
            "iteration_count": 3,
            "final_email": "",
            "final_result": {},
        }

        result = finalize_node(state)
        assert result["final_result"] is not None
        assert result["final_result"]["audit_approved"] is False
        assert result["final_result"]["iterations_used"] == 4


class TestGraphConstruction:
    def test_builds_workflow(self):
        workflow = build_workflow()
        assert workflow is not None

    def test_compiled_graph_has_expected_nodes(self):
        workflow = build_workflow()
        graph = workflow.compile()
        nodes = graph.get_graph().nodes
        node_names = {n if isinstance(n, str) else n.name for n in nodes}
        assert "retrieval" in node_names
        assert "strategist" in node_names
        assert "tools" in node_names
        assert "auditor" in node_names
        assert "finalize" in node_names
