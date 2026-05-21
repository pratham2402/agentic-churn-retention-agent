"""Tests for the LangGraph workflow and routing logic."""

import pytest
from acra.agent.graph import route_after_audit, build_workflow
from acra.agent.state import RetentionState


class TestRouting:
    def test_route_approved_goes_to_finalize(self):
        state: RetentionState = {
            "customer_id": "CUST-001",
            "customer_name": "Alice",
            "customer_email": "alice@example.com",
            "cancellation_reason": "Too expensive",
            "customer_profile": {},
            "playbook_policies": [],
            "proposed_offer": {},
            "audit_approved": True,
            "audit_feedback": "Approved",
            "iteration_count": 0,
            "final_email": "",
            "final_json": {},
        }
        assert route_after_audit(state) == "finalize"

    def test_route_rejected_loops_back(self):
        state: RetentionState = {
            "customer_id": "CUST-001",
            "customer_name": "Alice",
            "customer_email": "alice@example.com",
            "cancellation_reason": "Too expensive",
            "customer_profile": {},
            "playbook_policies": [],
            "proposed_offer": {},
            "audit_approved": False,
            "audit_feedback": "Discount too high",
            "iteration_count": 0,
            "final_email": "",
            "final_json": {},
        }
        assert route_after_audit(state) == "strategist"

    def test_route_max_iterations_forces_finalize(self):
        state: RetentionState = {
            "customer_id": "CUST-001",
            "customer_name": "Alice",
            "customer_email": "alice@example.com",
            "cancellation_reason": "Too expensive",
            "customer_profile": {},
            "playbook_policies": [],
            "proposed_offer": {},
            "audit_approved": False,
            "audit_feedback": "Still invalid",
            "iteration_count": 2,
            "final_email": "",
            "final_json": {},
        }
        assert route_after_audit(state) == "finalize"


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
        assert "auditor" in node_names
        assert "finalize" in node_names
