"""Tests for the LangChain @tool functions used by the Strategist agent."""

import pytest

from acra.agent.tools import (
    get_customer_profile,
    search_retention_policies,
    _get_retriever,
)


class TestGetCustomerProfile:
    def test_tool_is_invocable(self):
        """LangChain @tool decorator creates a BaseTool with .invoke()."""
        assert hasattr(get_customer_profile, "invoke")

    def test_tool_has_correct_name(self):
        assert get_customer_profile.name == "get_customer_profile"

    def test_tool_has_docstring(self):
        assert get_customer_profile.description is not None
        assert len(get_customer_profile.description) > 20

    def test_returns_json_string_for_missing_customer(self):
        """With no seeded data, returns empty JSON object."""
        result = get_customer_profile.invoke({"customer_id": "CUST-NONEXISTENT"})
        assert isinstance(result, str)
        assert "{}" in result

    def test_tool_accepts_customer_id_argument(self):
        """Verify the tool schema includes customer_id as a parameter."""
        schema = get_customer_profile.args_schema
        assert schema is not None
        fields = schema.model_fields if hasattr(schema, 'model_fields') else {}
        assert "customer_id" in fields


class TestSearchRetentionPolicies:
    def test_tool_is_invocable(self):
        """LangChain @tool decorator creates a BaseTool with .invoke()."""
        assert hasattr(search_retention_policies, "invoke")

    def test_tool_has_correct_name(self):
        assert search_retention_policies.name == "search_retention_policies"

    def test_tool_has_docstring(self):
        assert search_retention_policies.description is not None
        assert len(search_retention_policies.description) > 20

    def test_returns_string(self):
        """Tool should return a string result (or 'No matching policies found')."""
        result = search_retention_policies.invoke({"query": "discount limits"})
        assert isinstance(result, str)

    def test_tool_accepts_query_argument(self):
        """Verify the tool schema includes query as a parameter."""
        schema = search_retention_policies.args_schema
        assert schema is not None
        fields = schema.model_fields if hasattr(schema, 'model_fields') else {}
        assert "query" in fields


class TestRetrieverLazyInit:
    def test_get_retriever_returns_same_instance(self):
        """Lazy initialization should return the same instance on repeated calls."""
        # Force a fresh initialization
        import acra.agent.tools as tools_module
        tools_module._retriever = None

        r1 = _get_retriever()
        r2 = _get_retriever()
        assert r1 is r2
