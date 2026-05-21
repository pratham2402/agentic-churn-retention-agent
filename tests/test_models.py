"""Tests for Pydantic data models."""

import pytest
from pydantic import ValidationError
from acra.models import RetentionOffer, AuditResult, CustomerProfile, RetentionPayload


class TestRetentionOffer:
    def test_valid_offer(self):
        offer = RetentionOffer(
            discount_percent=30,
            duration_months=6,
            offer_type="discount",
            justification="Test justification",
            email_draft="Dear customer...",
        )
        assert offer.discount_percent == 30
        assert offer.duration_months == 6

    def test_discount_bounds(self):
        with pytest.raises(ValidationError):
            RetentionOffer(
                discount_percent=150,
                duration_months=6,
                offer_type="discount",
                justification="Test",
                email_draft="Test",
            )

    def test_negative_discount(self):
        with pytest.raises(ValidationError):
            RetentionOffer(
                discount_percent=-10,
                duration_months=6,
                offer_type="discount",
                justification="Test",
                email_draft="Test",
            )


class TestAuditResult:
    def test_approved_result(self):
        result = AuditResult(approved=True, feedback="All good")
        assert result.approved is True
        assert result.policy_violations == []

    def test_rejected_result(self):
        result = AuditResult(
            approved=False,
            feedback="Violation found",
            policy_violations=["POL-001"],
        )
        assert result.approved is False
        assert "POL-001" in result.policy_violations


class TestCustomerProfile:
    def test_valid_profile(self):
        profile = CustomerProfile(
            customer_id="CUST-001",
            name="Alice",
            email="alice@example.com",
            tenure_months=14,
            plan_name="Professional",
            monthly_cost_usd=49.0,
            lifetime_value_usd=686.0,
            support_tickets_last_90d=2,
            feature_usage_score=0.72,
            payment_history="excellent",
        )
        assert profile.customer_id == "CUST-001"

    def test_feature_usage_bounds(self):
        with pytest.raises(ValidationError):
            CustomerProfile(
                customer_id="CUST-001",
                name="Alice",
                email="alice@example.com",
                tenure_months=14,
                plan_name="Professional",
                monthly_cost_usd=49.0,
                lifetime_value_usd=686.0,
                support_tickets_last_90d=2,
                feature_usage_score=1.5,
                payment_history="excellent",
            )


class TestRetentionPayload:
    def test_full_payload(self):
        offer = RetentionOffer(
            discount_percent=20,
            duration_months=3,
            offer_type="discount",
            justification="Within limits",
            email_draft="Dear Alice...",
        )
        payload = RetentionPayload(
            customer_id="CUST-001",
            customer_name="Alice",
            original_reason="Too expensive",
            final_offer=offer,
            audit_approved=True,
            email_to_send="Dear Alice...",
            db_update={"applied_discount": 20},
        )
        assert payload.audit_approved is True
        assert payload.final_offer.discount_percent == 20
