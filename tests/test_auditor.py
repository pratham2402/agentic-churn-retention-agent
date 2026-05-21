"""Tests for the Financial Auditor agent node."""

import pytest
from unittest.mock import MagicMock, patch

from acra.agent.auditor import build_auditor_prompt, create_auditor_node
from acra.models import AuditResult


class TestAuditorPrompt:
    def test_builds_prompt_with_offer_and_profile(self):
        state = {
            "customer_id": "CUST-001",
            "customer_name": "Alice",
            "customer_email": "alice@example.com",
            "cancellation_reason": "Too expensive",
            "customer_profile": {"tenure_months": 14, "plan_name": "Pro", "lifetime_value_usd": 686},
            "playbook_policies": [],
            "proposed_offer": {"discount_percent": 50, "duration_months": 6, "offer_type": "discount"},
            "audit_approved": False,
            "audit_feedback": "",
            "iteration_count": 0,
            "final_email": "",
            "final_json": {},
        }
        prompt = build_auditor_prompt(state)
        assert "50" in prompt
        assert "Pro" in prompt or "14" in prompt
        assert "discount" in prompt


class TestAuditorNode:
    @patch("acra.agent.auditor.ChatOpenAI")
    def test_approves_valid_offer(self, mock_chat):
        mock_result = AuditResult(
            approved=True,
            feedback="Offer is compliant with all policies.",
            policy_violations=[],
        )

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_result
        mock_llm.with_structured_output.return_value = mock_structured
        mock_chat.return_value = mock_llm

        state = {
            "customer_id": "CUST-001",
            "customer_name": "Alice",
            "customer_email": "alice@example.com",
            "cancellation_reason": "Too expensive",
            "customer_profile": {"tenure_months": 14, "plan_name": "Professional", "lifetime_value_usd": 686},
            "playbook_policies": [],
            "proposed_offer": {"discount_percent": 30, "duration_months": 3, "offer_type": "discount"},
            "audit_approved": False,
            "audit_feedback": "",
            "iteration_count": 0,
            "final_email": "",
            "final_json": {},
        }

        node = create_auditor_node()
        result = node(state)

        assert result["audit_approved"] is True

    @patch("acra.agent.auditor.ChatOpenAI")
    def test_rejects_invalid_offer(self, mock_chat):
        mock_result = AuditResult(
            approved=False,
            feedback="Discount of 70% exceeds maximum of 50%.",
            policy_violations=["POL-001"],
        )

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_result
        mock_llm.with_structured_output.return_value = mock_structured
        mock_chat.return_value = mock_llm

        state = {
            "customer_id": "CUST-002",
            "customer_name": "Bob",
            "customer_email": "bob@example.com",
            "cancellation_reason": "Too expensive",
            "customer_profile": {"tenure_months": 3, "plan_name": "Starter", "lifetime_value_usd": 57},
            "playbook_policies": [],
            "proposed_offer": {"discount_percent": 70, "duration_months": 12, "offer_type": "discount"},
            "audit_approved": False,
            "audit_feedback": "",
            "iteration_count": 0,
            "final_email": "",
            "final_json": {},
        }

        node = create_auditor_node()
        result = node(state)

        assert result["audit_approved"] is False
        assert "70%" in result["audit_feedback"] or "exceeds" in result["audit_feedback"].lower()
