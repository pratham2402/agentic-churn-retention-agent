"""Tests for the programmatic Auditor policy checks.

These verify that every policy check function correctly enforces
the hard-coded rules with exact math. Zero LLM dependency - all
checks are deterministic Python functions.
"""

import pytest
from acra.models import RetentionOffer, AuditResult
from acra.agent.auditor import (
    check_pol_001_tenure_discount_limits,
    check_pol_002_ltv_protection,
    check_pol_003_free_month_rules,
    check_pol_004_plan_downgrade,
    check_pol_005_feature_gap,
    check_pol_006_high_value_protocol,
    check_pol_007_competitor_price_match,
    check_pol_hv_001_executive_reachout,
    run_all_policy_checks,
    parse_offer_from_messages,
    _extract_json_from_message,
    create_auditor_node,
)


# ── POL-001: Tenure-based discount limits ────────────────────────

class TestPOL001:
    def test_under_6_months_max_20_approved(self):
        passed, detail, _ = check_pol_001_tenure_discount_limits(3, 20)
        assert passed is True

    def test_under_6_months_25_rejected(self):
        passed, detail, _ = check_pol_001_tenure_discount_limits(3, 25)
        assert passed is False
        assert "POL-001" in detail
        assert "20%" in detail

    def test_6_to_12_months_max_30_approved(self):
        passed, detail, _ = check_pol_001_tenure_discount_limits(8, 30)
        assert passed is True

    def test_6_to_12_months_35_rejected(self):
        passed, detail, _ = check_pol_001_tenure_discount_limits(10, 35)
        assert passed is False
        assert "30%" in detail

    def test_12_to_24_months_max_40_approved(self):
        passed, detail, _ = check_pol_001_tenure_discount_limits(14, 40)
        assert passed is True

    def test_12_to_24_months_50_rejected(self):
        passed, detail, _ = check_pol_001_tenure_discount_limits(18, 50)
        assert passed is False
        assert "40%" in detail

    def test_24_plus_months_max_50_approved(self):
        passed, detail, _ = check_pol_001_tenure_discount_limits(36, 50)
        assert passed is True

    def test_absolute_cap_60_rejected_regardless_of_tenure(self):
        passed, detail, _ = check_pol_001_tenure_discount_limits(60, 60)
        assert passed is False
        assert ("absolute" in detail.lower() or "50%" in detail)

    def test_absolute_cap_55_rejected(self):
        passed, detail, _ = check_pol_001_tenure_discount_limits(100, 55)
        assert passed is False


# ── POL-002: LTV protection floor ─────────────────────────────────

class TestPOL002:
    def test_discounted_above_floor_approved(self):
        passed, detail, _ = check_pol_002_ltv_protection(49.00, 30)
        assert passed is True

    def test_discounted_below_30pct_rejected(self):
        passed, detail, _ = check_pol_002_ltv_protection(19.00, 90)
        assert passed is False
        assert "POL-002" in detail

    def test_low_monthly_absolute_5_dollar_floor(self):
        passed, detail, _ = check_pol_002_ltv_protection(10.00, 60)
        assert passed is False

    def test_high_monthly_moderate_discount_approved(self):
        passed, detail, _ = check_pol_002_ltv_protection(299.00, 40)
        assert passed is True


# ── POL-003: Free month rules ─────────────────────────────────────

class TestPOL003:
    def test_free_months_non_free_offer_skips(self):
        passed, detail, _ = check_pol_003_free_month_rules(6, "discount", 20, 6)
        assert passed is True

    def test_free_months_under_12_tenure_rejected(self):
        passed, detail, _ = check_pol_003_free_month_rules(8, "free_months", 0, 1)
        assert passed is False
        assert "POL-003" in detail
        assert "12" in detail

    def test_free_months_valid_12_plus_tenure_approved(self):
        passed, detail, _ = check_pol_003_free_month_rules(24, "free_months", 0, 2)
        assert passed is True

    def test_free_months_exceed_2(self):
        passed, detail, _ = check_pol_003_free_month_rules(24, "free_months", 0, 3)
        assert passed is False

    def test_free_months_combined_with_discount_rejected(self):
        passed, detail, _ = check_pol_003_free_month_rules(24, "free_months", 10, 1)
        assert passed is False
        assert "combine" in detail.lower() or "POL-003" in detail


# ── POL-004: Plan downgrade ───────────────────────────────────────

class TestPOL004:
    def test_cost_concern_downgrade_proposed_passes(self):
        passed, detail, _ = check_pol_004_plan_downgrade(
            "Professional", "plan_downgrade", "Too expensive", 0
        )
        assert passed is True

    def test_cost_concern_no_downgrade_starter_skips(self):
        passed, detail, _ = check_pol_004_plan_downgrade(
            "Starter", "discount", "Too expensive for me", 20
        )
        assert passed is True

    def test_not_cost_concern_skips(self):
        passed, detail, _ = check_pol_004_plan_downgrade(
            "Professional", "discount", "Missing feature X", 20
        )
        assert passed is True

    def test_enterprise_cost_concern_discount_warns(self):
        passed, detail, _ = check_pol_004_plan_downgrade(
            "Enterprise", "discount", "Budget cuts, need to reduce spending", 30
        )
        assert passed is False


# ── POL-005: Feature gap handling ─────────────────────────────────

class TestPOL005:
    def test_feature_gap_with_feature_unlock_passes(self):
        passed, detail, _ = check_pol_005_feature_gap(
            "Missing feature X that competitor has", "feature_unlock", 0
        )
        assert passed is True

    def test_feature_gap_with_discount_rejected(self):
        passed, detail, _ = check_pol_005_feature_gap(
            "Missing feature X that competitor has", "discount", 25
        )
        assert passed is False
        assert "POL-005" in detail

    def test_not_feature_gap_skips(self):
        passed, detail, _ = check_pol_005_feature_gap(
            "Too expensive", "discount", 20
        )
        assert passed is True

    def test_compound_offer_discount_plus_tier_trial_passes(self):
        """A discount with tier trial mentioned in justification should pass
        even for feature-gap cancellations - it addresses both concerns."""
        passed, detail, _ = check_pol_005_feature_gap(
            cancellation_reason="Missing feature X that competitor has",
            offer_type="discount",
            discount_percent=15,
            justification="Offering 15% discount plus a 30-day free trial of the next tier to address the feature gap.",
            reasoning="Customer cites both cost and features. Discount helps cost, tier trial helps features.",
            email_draft="I'd like to offer you a 15% discount and a free 30-day trial of our next tier.",
        )
        assert passed is True

    def test_pure_discount_without_trial_still_rejected(self):
        """A pure discount with no product access for a feature gap still fails."""
        passed, detail, _ = check_pol_005_feature_gap(
            cancellation_reason="Missing feature X that competitor has",
            offer_type="discount",
            discount_percent=15,
            justification="Within limits, discount is best option.",
            reasoning="POL-001 allows this.",
            email_draft="Here's a 15% discount.",
        )
        assert passed is False
        assert "POL-005" in detail


# ── POL-007: Competitor price match ───────────────────────────────

class TestPOL007:
    def test_competitor_reasonable_discount_passes(self):
        passed, detail, _ = check_pol_007_competitor_price_match(
            "competitor with better pricing", 30, 14, 49.00
        )
        assert passed is True

    def test_competitor_below_50pct_rejected(self):
        passed, detail, _ = check_pol_007_competitor_price_match(
            "moving to a competitor with better pricing", 55, 36, 299.00
        )
        assert passed is False
        assert "POL-007" in detail

    def test_not_competitor_skips(self):
        passed, detail, _ = check_pol_007_competitor_price_match(
            "Too expensive", 40, 14, 49.00
        )
        assert passed is True


# ── POL-HV-001: Executive reach-out ───────────────────────────────

class TestPOLHV001:
    def test_ltv_above_25k_triggers_notice(self):
        passed, detail, _ = check_pol_hv_001_executive_reachout(30000.00)
        assert passed is True
        assert "POL-HV-001" in detail

    def test_ltv_below_25k_no_notice(self):
        passed, detail, _ = check_pol_hv_001_executive_reachout(5000.00)
        assert passed is True
        assert detail == ""


# ── Aggregate audit ───────────────────────────────────────────────

class TestRunAllPolicyChecks:
    def test_valid_offer_passes_all_checks(self):
        offer = RetentionOffer(
            discount_percent=30,
            duration_months=6,
            offer_type="discount",
            justification="Within all limits",
            reasoning="Test",
            email_draft="Dear customer...",
        )
        profile = {
            "tenure_months": 14,
            "monthly_cost_usd": 49.00,
            "lifetime_value_usd": 686.00,
            "plan_name": "Professional",
        }
        result = run_all_policy_checks(offer, profile, "Too expensive")
        assert result.approved is True

    def test_excessive_discount_fails(self):
        offer = RetentionOffer(
            discount_percent=60,
            duration_months=6,
            offer_type="discount",
            justification="Generous offer",
            reasoning="Test",
            email_draft="Dear customer...",
        )
        profile = {
            "tenure_months": 14,
            "monthly_cost_usd": 49.00,
            "lifetime_value_usd": 686.00,
            "plan_name": "Professional",
        }
        result = run_all_policy_checks(offer, profile, "Too expensive")
        assert result.approved is False
        assert "POL-001" in result.policy_violations

    def test_feature_gap_with_discount_fails(self):
        offer = RetentionOffer(
            discount_percent=20,
            duration_months=3,
            offer_type="discount",
            justification="Here's a discount",
            reasoning="Test",
            email_draft="Dear customer...",
        )
        profile = {
            "tenure_months": 8,
            "monthly_cost_usd": 49.00,
            "lifetime_value_usd": 392.00,
            "plan_name": "Professional",
        }
        result = run_all_policy_checks(
            offer, profile, "Missing feature X that the competitor has"
        )
        assert result.approved is False
        assert "POL-005" in result.policy_violations


# ── JSON extraction ───────────────────────────────────────────────

class TestJSONExtraction:
    def test_extracts_bare_json(self):
        content = '{"discount_percent": 30, "duration_months": 6, "offer_type": "discount", "justification": "test", "reasoning": "test", "email_draft": "Dear..."}'
        parsed = _extract_json_from_message(content)
        assert parsed is not None
        assert parsed["discount_percent"] == 30

    def test_extracts_from_code_fence(self):
        content = (
            'Here is my offer:\n'
            '```json\n'
            '{"discount_percent": 25, "duration_months": 3, "offer_type": "discount", '
            '"justification": "ok", "reasoning": "thought", "email_draft": "Hi"}\n'
            '```\n'
            "That's it."
        )
        parsed = _extract_json_from_message(content)
        assert parsed is not None
        assert parsed["discount_percent"] == 25

    def test_returns_none_for_invalid(self):
        parsed = _extract_json_from_message("Just some random text")
        assert parsed is None


# ── Offer parsing from messages ───────────────────────────────────

class TestParseOfferFromMessages:
    def test_parses_valid_offer_from_ai_message(self, valid_offer):
        import json
        from langchain_core.messages import AIMessage

        messages = [AIMessage(content=json.dumps(valid_offer))]
        offer = parse_offer_from_messages(messages)
        assert offer is not None
        assert offer.discount_percent == 30
        assert offer.offer_type == "discount"

    def test_skips_tool_call_messages(self):
        import json
        from langchain_core.messages import AIMessage

        msg_with_tools = AIMessage(
            content="",
            tool_calls=[{
                "name": "get_customer_profile",
                "args": {"customer_id": "CUST-001"},
                "id": "call_1",
            }],
        )
        valid_json_msg = AIMessage(content=json.dumps({
            "discount_percent": 20,
            "duration_months": 3,
            "offer_type": "plan_downgrade",
            "justification": "Better fit",
            "reasoning": "Thought about it",
            "email_draft": "Hello",
        }))
        messages = [msg_with_tools, valid_json_msg]
        offer = parse_offer_from_messages(messages)
        assert offer is not None
        assert offer.discount_percent == 20


# ── Auditor node integration ──────────────────────────────────────

class TestAuditorNode:
    def test_approves_valid_offer(self, sample_profile, valid_offer):
        import json
        from langchain_core.messages import AIMessage, SystemMessage

        state = {
            "messages": [
                SystemMessage(content="Strategist system prompt"),
                AIMessage(content=json.dumps(valid_offer)),
            ],
            "customer_id": "CUST-001",
            "cancellation_reason": "Too expensive",
            "customer_profile": sample_profile,
            "proposed_offer": {},
            "audit_approved": False,
            "audit_feedback": "",
            "iteration_count": 0,
        }

        node = create_auditor_node()
        result = node(state)

        assert result["audit_approved"] is True
        assert "proposed_offer" in result

    def test_rejects_excessive_discount(self, sample_profile):
        import json
        from langchain_core.messages import AIMessage

        bad_offer = {
            "discount_percent": 70,
            "duration_months": 6,
            "offer_type": "discount",
            "justification": "Generous offer",
            "reasoning": "Test",
            "email_draft": "Dear customer...",
        }

        state = {
            "messages": [AIMessage(content=json.dumps(bad_offer))],
            "customer_id": "CUST-001",
            "cancellation_reason": "Too expensive",
            "customer_profile": sample_profile,
            "proposed_offer": {},
            "audit_approved": False,
            "audit_feedback": "",
            "iteration_count": 0,
        }

        node = create_auditor_node()
        result = node(state)

        assert result["audit_approved"] is False
        assert "POL-001" in result.get("audit_feedback", "")

    def test_handles_unparseable_response(self):
        from langchain_core.messages import AIMessage

        state = {
            "messages": [
                AIMessage(content="I cannot help with that request."),
            ],
            "customer_id": "CUST-001",
            "cancellation_reason": "Too expensive",
            "customer_profile": {},
            "proposed_offer": {},
            "audit_approved": False,
            "audit_feedback": "",
            "iteration_count": 0,
        }

        node = create_auditor_node()
        result = node(state)

        assert result["audit_approved"] is False
        assert "parse" in result.get("audit_feedback", "").lower()
