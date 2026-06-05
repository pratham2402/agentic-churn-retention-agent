"""Tests for the Retention Strategist agent - tool binding and node creation."""

import pytest
from unittest.mock import MagicMock, patch

from acra.agent.strategist import create_strategist_node, STRATEGIST_TOOLS


class TestStrategistTools:
    def test_tools_are_defined(self):
        assert len(STRATEGIST_TOOLS) == 2
        tool_names = {t.name for t in STRATEGIST_TOOLS}
        assert "get_customer_profile" in tool_names
        assert "search_retention_policies" in tool_names


class TestStrategistNode:
    @patch("acra.agent.strategist.ChatOpenAI")
    def test_node_returns_tool_call_when_llm_requests_tool(self, mock_chat):
        from langchain_core.messages import AIMessage

        # Simulate the LLM requesting a tool call
        mock_llm = MagicMock()
        mock_llm_with_tools = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        mock_llm_with_tools.invoke.return_value = AIMessage(
            content="",
            tool_calls=[{
                "name": "get_customer_profile",
                "args": {"customer_id": "CUST-001"},
                "id": "call_1",
            }],
        )
        mock_chat.return_value = mock_llm

        state = {
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

        node = create_strategist_node()
        result = node(state)

        assert "messages" in result
        last_msg = result["messages"][-1]
        assert hasattr(last_msg, "tool_calls")
        assert len(last_msg.tool_calls) == 1

    @patch("acra.agent.strategist.ChatOpenAI")
    def test_node_returns_final_response_when_no_tool_calls(self, mock_chat):
        from langchain_core.messages import AIMessage

        mock_llm = MagicMock()
        mock_llm_with_tools = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        mock_llm_with_tools.invoke.return_value = AIMessage(
            content='{"discount_percent": 30, "duration_months": 6, '
                    '"offer_type": "discount", "justification": "Within limits", '
                    '"reasoning": "Test reasoning", "email_draft": "Dear Alice..."}',
        )
        mock_chat.return_value = mock_llm

        state = {
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

        node = create_strategist_node()
        result = node(state)

        assert "messages" in result
        last_msg = result["messages"][-1]
        assert hasattr(last_msg, "content")
        # Should not have tool calls
        assert not (hasattr(last_msg, "tool_calls") and last_msg.tool_calls)

    @patch("acra.agent.strategist.ChatOpenAI")
    def test_node_prepends_system_message_on_first_call(self, mock_chat):
        from langchain_core.messages import AIMessage, SystemMessage

        captured_messages = []

        mock_llm = MagicMock()
        mock_llm_with_tools = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm_with_tools

        def capture(messages):
            captured_messages.extend(messages)
            return AIMessage(content="final response")

        mock_llm_with_tools.invoke.side_effect = capture
        mock_chat.return_value = mock_llm

        state = {
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

        node = create_strategist_node()
        node(state)

        # First message should be a SystemMessage
        system_msgs = [m for m in captured_messages if isinstance(m, SystemMessage)]
        assert len(system_msgs) >= 1
