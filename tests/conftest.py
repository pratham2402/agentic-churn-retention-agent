"""Pytest configuration — load env, provide shared fixtures."""

import os
import json
import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture
def sample_profile() -> dict:
    """A standard customer profile for testing."""
    return {
        "customer_id": "CUST-001",
        "name": "Alice Johnson",
        "email": "alice@example.com",
        "tenure_months": 14,
        "plan_name": "Professional",
        "monthly_cost_usd": 49.00,
        "lifetime_value_usd": 686.00,
        "support_tickets_last_90d": 2,
        "feature_usage_score": 0.72,
        "payment_history": "excellent",
    }


@pytest.fixture
def high_value_profile() -> dict:
    """A high-LTV enterprise customer profile."""
    return {
        "customer_id": "CUST-003",
        "name": "Carol Martinez",
        "email": "carol@example.com",
        "tenure_months": 36,
        "plan_name": "Enterprise",
        "monthly_cost_usd": 299.00,
        "lifetime_value_usd": 10764.00,
        "support_tickets_last_90d": 1,
        "feature_usage_score": 0.95,
        "payment_history": "excellent",
    }


@pytest.fixture
def new_customer_profile() -> dict:
    """A customer with very short tenure (< 6 months)."""
    return {
        "customer_id": "CUST-002",
        "name": "Bob Williams",
        "email": "bob@example.com",
        "tenure_months": 3,
        "plan_name": "Starter",
        "monthly_cost_usd": 19.00,
        "lifetime_value_usd": 57.00,
        "support_tickets_last_90d": 8,
        "feature_usage_score": 0.25,
        "payment_history": "good",
    }


@pytest.fixture
def valid_offer() -> dict:
    """A valid retention offer within policy limits for the sample customer."""
    return {
        "discount_percent": 30,
        "duration_months": 6,
        "offer_type": "discount",
        "justification": "Mid-tenure customer, discount within POL-001 limits",
        "reasoning": "Checked POL-001: 14mo tenure allows 40%, offering 30%. "
                     "POL-002: discounted rate $34.30 > $14.70 floor.",
        "email_draft": "Dear Alice, we value your 14 months with us...",
    }
