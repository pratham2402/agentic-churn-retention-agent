"""Tests for the Retention Strategist agent node."""

import pytest
from unittest.mock import MagicMock, patch

from acra.agent.strategist import build_strategist_prompt, create_strategist_node
from acra.models import RetentionOffer


class TestStrategistPrompt:
    def test_builds_prompt_with_profile_and_policies(self):
        state = {
            "customer_id": "CUST-001",
            "customer_name": "Alice",
            "customer_email": "alice@example.com",
            "cancellation_reason": "Too expensive",
            "customer_profile": {"tenure_months": 14, "plan_name": "Pro", "lifetime_value_usd": 686},
            "playbook_policies": [
                {"policy_id": "POL-001", "content": "Max 40% for 12-24mo tenure"},
            ],
            "proposed_offer": {},
            "audit_approved": False,
            "audit_feedback": "",
            "iteration_count": 0,
            "final_email": "",
            "final_json": {},
        }
        prompt = build_strategist_prompt(state)
        assert "CUST-001" in prompt or "14" in prompt
        assert "Too expensive" in prompt
        assert "POL-001" in prompt
        assert "Max 40% for 12-24mo tenure" in prompt

    def test_includes_previous_audit_feedback(self):
        state = {
            "customer_id": "CUST-001",
            "customer_name": "",
            "customer_email": "",
            "cancellation_reason": "Too expensive",
            "customer_profile": {"tenure_months": 3},
            "playbook_policies": [],
            "proposed_offer": {},
            "audit_approved": False,
            "audit_feedback": "Discount too high for 3-month tenure",
            "iteration_count": 1,
            "final_email": "",
            "final_json": {},
        }
        prompt = build_strategist_prompt(state)
        assert "REJECTED" in prompt
        assert "Discount too high for 3-month tenure" in prompt


class TestStrategistNode:
    @patch("acra.agent.strategist.ChatOpenAI")
    def test_returns_offer_in_state(self, mock_chat):
        mock_offer = RetentionOffer(
            discount_percent=30,
            duration_months=6,
            offer_type="discount",
            justification="Within 12-24mo limit of 40%",
            email_draft="Dear customer, we value you...",
        )

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_offer
        mock_llm.with_structured_output.return_value = mock_structured
        mock_chat.return_value = mock_llm

        state = {
            "customer_id": "CUST-001",
            "customer_name": "Alice",
            "customer_email": "alice@example.com",
            "cancellation_reason": "Too expensive",
            "customer_profile": {"tenure_months": 14, "plan_name": "Professional", "lifetime_value_usd": 686},
            "playbook_policies": [
                {"policy_id": "POL-001", "content": "Max 40% discount for 12-24 month tenure."},
            ],
            "proposed_offer": {},
            "audit_approved": False,
            "audit_feedback": "",
            "iteration_count": 0,
            "final_email": "",
            "final_json": {},
        }

        node = create_strategist_node()
        result = node(state)

        assert "proposed_offer" in result
        assert result["proposed_offer"]["discount_percent"] == 30
        assert result["proposed_offer"]["offer_type"] == "discount"
        assert "email_draft" in result["proposed_offer"]
