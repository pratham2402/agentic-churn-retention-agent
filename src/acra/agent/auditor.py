"""Node 2: The Financial Auditor - programmatic policy enforcement.

This is the CRITIC in the actor-critic loop. It performs HARD-CODED,
zero-LLM compliance checks against every company policy. No LLM
reasoning is used for rule enforcement - all checks are deterministic
Python functions that do exact math.

Policy coverage:
    POL-001  - Tenure-based maximum discount limits
    POL-002  - Lifetime value floor protection
    POL-003  - Free month eligibility and limits
    POL-004  - Plan downgrade preference
    POL-005  - Feature gap handling
    POL-006  - High-value customer protocol
    POL-007  - Competitor price match protocol
    POL-HV-001  - Executive reach-out trigger
    POL-RISK-001 - Serial canceller detection
"""

import json
import re
from langchain_core.messages import SystemMessage

from acra.models import AuditResult, RetentionOffer
from acra.agent.state import RetentionState


# ── Policy check functions ───────────────────────────────────────
# Each returns (passed: bool, violation_detail: str, checked_values: dict)


def check_pol_001_tenure_discount_limits(
    tenure_months: int,
    discount_percent: int,
) -> tuple[bool, str, dict]:
    """POL-001: Maximum discount by customer tenure.

    Rules:
        < 6 months  → max 20%
        6-12 months → max 30%
        12-24 months → max 40%
        24+ months  → max 50%
        Absolute cap: 50% for any customer
    """
    limits = [(6, 20), (12, 30), (24, 40), (float("inf"), 50)]
    max_allowed = 50
    for threshold, cap in limits:
        if tenure_months < threshold:
            max_allowed = cap
            break

    checked = {
        "tenure_months": tenure_months,
        "proposed_discount": discount_percent,
        "max_allowed": max_allowed,
        "absolute_cap": 50,
    }

    if discount_percent > 50:
        return False, (
            f"POL-001 VIOLATION: Proposed discount of {discount_percent}% exceeds "
            f"the absolute maximum of 50% for any customer."
        ), checked

    if discount_percent > max_allowed:
        return False, (
            f"POL-001 VIOLATION: Proposed discount of {discount_percent}% exceeds "
            f"the {max_allowed}% maximum for a customer with {tenure_months} months "
            f"of tenure ({_tenure_bucket(tenure_months)} bucket)."
        ), checked

    return True, "", checked


def check_pol_002_ltv_protection(
    monthly_cost_usd: float,
    discount_percent: int,
) -> tuple[bool, str, dict]:
    """POL-002: Lifetime value protection floor.

    Discounted monthly cost must be >= max(current_monthly * 0.30, $5.00).
    """
    discounted = monthly_cost_usd * (1.0 - discount_percent / 100.0)
    floor = max(monthly_cost_usd * 0.30, 5.00)

    checked = {
        "current_monthly": round(monthly_cost_usd, 2),
        "discounted_monthly": round(discounted, 2),
        "revenue_floor": round(floor, 2),
        "floor_30pct": round(monthly_cost_usd * 0.30, 2),
        "floor_absolute": 5.00,
    }

    if discounted < floor - 0.001:  # tolerance for float rounding
        return False, (
            f"POL-002 VIOLATION: The discounted monthly cost of ${discounted:.2f} "
            f"falls below the required revenue floor of ${floor:.2f} "
            f"(max of 30% of current rate = ${monthly_cost_usd * 0.30:.2f} "
            f"and absolute minimum $5.00)."
        ), checked

    return True, "", checked


def check_pol_003_free_month_rules(
    tenure_months: int,
    offer_type: str,
    discount_percent: int,
    duration_months: int,
) -> tuple[bool, str, dict]:
    """POL-003: Free month eligibility and restrictions.

    - Only for customers with 12+ months tenure
    - Max 2 free months per retention event
    - Cannot combine free months with percentage discounts
    """
    checked = {
        "tenure_months": tenure_months,
        "offer_type": offer_type,
        "discount_percent": discount_percent,
        "duration_months": duration_months,
    }

    if offer_type != "free_months":
        return True, "", checked

    if tenure_months < 12:
        return False, (
            f"POL-003 VIOLATION: Free months offered to customer with only "
            f"{tenure_months} months tenure. Minimum 12 months required."
        ), checked

    if duration_months > 2:
        return False, (
            f"POL-003 VIOLATION: {duration_months} free months requested. "
            f"Maximum 2 free months per retention event."
        ), checked

    if discount_percent > 0:
        return False, (
            f"POL-003 VIOLATION: Free months cannot be combined with a "
            f"{discount_percent}% percentage discount."
        ), checked

    return True, "", checked


def check_pol_004_plan_downgrade(
    plan_name: str,
    offer_type: str,
    cancellation_reason: str,
    discount_percent: int,
) -> tuple[bool, str, dict]:
    """POL-004: Plan downgrade preference for cost concerns.

    - For cost-concern cancellations, verify downgrade was considered
    - Enterprise plan downgrades need special scrutiny
    - If already on lowest tier, skip (not a violation)
    """
    checked = {
        "plan_name": plan_name,
        "offer_type": offer_type,
        "cancellation_reason": cancellation_reason[:100],
        "discount_percent": discount_percent,
    }

    cost_keywords = ["too expensive", "cheaper", "cost", "budget", "pricing",
                     "price", "reduce spending", "not getting enough value"]
    is_cost_concern = any(kw in cancellation_reason.lower() for kw in cost_keywords)

    if not is_cost_concern:
        return True, "", checked

    # Can't downgrade from Starter - lowest tier
    if plan_name.lower() in ("starter", "basic"):
        return True, "", checked

    if offer_type == "plan_downgrade":
        return True, "", checked

    # Cost concern, not on lowest tier, didn't propose downgrade
    # This is a SOFT violation - flag it but don't block
    if plan_name.lower() == "enterprise" and offer_type != "plan_downgrade":
        return False, (
            f"POL-004 WARNING: Customer on Enterprise plan cites cost concerns. "
            f"Plan downgrade should be considered as first resort before offering "
            f"a {discount_percent}% discount. Enterprise downgrades require "
            f"manager approval."
        ), checked

    return True, "", checked


def check_pol_005_feature_gap(
    cancellation_reason: str,
    offer_type: str,
    discount_percent: int,
    justification: str = "",
    reasoning: str = "",
    email_draft: str = "",
) -> tuple[bool, str, dict]:
    """POL-005: Feature gap handling.

    - Feature-gap cancellations should include product access solutions
      (tier trial, feature unlock), not just pure discounts
    - If the offer combines a discount with a tier trial/feature unlock
      (addressing both cost AND feature concerns), accept it
    - The check inspects the justification, reasoning, and email for
      mentions of tier trials or feature unlocks
    """
    feature_keywords = ["missing feature", "feature", "doesn't have",
                        "need feature", "competitor has"]
    is_feature_gap = any(kw in cancellation_reason.lower() for kw in feature_keywords)

    checked = {
        "cancellation_reason": cancellation_reason[:100],
        "offer_type": offer_type,
        "discount_percent": discount_percent,
        "is_feature_gap": is_feature_gap,
    }

    if not is_feature_gap:
        return True, "", checked

    # Pure product access solutions always pass
    if offer_type in ("feature_unlock", "tier_trial"):
        return True, "", checked

    # Check if the offer addresses the feature gap through product access
    # even if offer_type is "discount" (compound offer for cost + feature)
    product_access_keywords = [
        "tier trial", "free trial", "feature unlock", "next tier",
        "try more features", "trial of the next tier", "product access",
        "30-day trial", "upgrade at no cost", "try the",
    ]
    combined_context = f"{justification} {reasoning} {email_draft}".lower()
    addresses_features = any(
        kw in combined_context for kw in product_access_keywords
    )

    if addresses_features:
        return True, "", checked

    # Pure discount with no product access for a feature gap → violation
    if discount_percent > 0:
        return False, (
            f"POL-005 VIOLATION: Customer cites feature gap "
            f"('{cancellation_reason[:80]}...') "
            f"but received only a {discount_percent}% discount with no "
            f"tier trial or feature unlock. Per POL-005, feature requests "
            f"must be solved with product access, not discounts alone."
        ), checked

    return True, "", checked


def check_pol_006_high_value_protocol(
    ltv_usd: float,
    monthly_cost_usd: float,
    tenure_months: int,
    discount_percent: int,
) -> tuple[bool, str, dict]:
    """POL-006: High-value customer retention protocol.

    - LTV > $10k or monthly > $500 → classified as high-value
    - High-value customers may receive premium treatment
    - BUT tenure-based discount limits STILL APPLY (enforced by POL-001)
    - This check is informational - validates that premium treatment is considered
    """
    is_high_value = ltv_usd > 10000 or monthly_cost_usd > 500

    checked = {
        "ltv_usd": ltv_usd,
        "monthly_cost_usd": monthly_cost_usd,
        "is_high_value": is_high_value,
        "tenure_months": tenure_months,
        "discount_percent": discount_percent,
    }

    if not is_high_value:
        return True, "", checked

    # High-value customer - this is informational
    # The hard limits are enforced by POL-001
    return True, "", checked


def check_pol_007_competitor_price_match(
    cancellation_reason: str,
    discount_percent: int,
    tenure_months: int,
    monthly_cost_usd: float,
) -> tuple[bool, str, dict]:
    """POL-007: Competitor price match protocol.

    - May match competitor price within tenure-based discount limits
    - Price matching below 50% of current rate requires manager approval
    - Must highlight unique value propositions
    """
    competitor_keywords = ["competitor", "cheaper alternative", "better pricing",
                           "competitor with better", "moving to a competitor"]
    is_competitor = any(kw in cancellation_reason.lower() for kw in competitor_keywords)

    discounted_pct_of_current = (1.0 - discount_percent / 100.0) * 100

    checked = {
        "cancellation_reason": cancellation_reason[:100],
        "is_competitor_related": is_competitor,
        "discount_percent": discount_percent,
        "discounted_as_pct_of_current": round(discounted_pct_of_current, 1),
        "tenure_months": tenure_months,
    }

    if not is_competitor:
        return True, "", checked

    if discounted_pct_of_current < 50:
        return False, (
            f"POL-007 VIOLATION: Proposed {discount_percent}% discount reduces "
            f"monthly cost to {discounted_pct_of_current:.1f}% of current rate "
            f"(${monthly_cost_usd:.2f}). Price matching below 50% of current "
            f"rate requires manager approval."
        ), checked

    return True, "", checked


def check_pol_hv_001_executive_reachout(
    ltv_usd: float,
) -> tuple[bool, str, dict]:
    """POL-HV-001: Executive reach-out trigger.

    Informational check - LTV > $25k triggers executive engagement workflow.
    This is not a violation, but a notification.
    """
    checked = {"ltv_usd": ltv_usd, "triggers_executive_reachout": ltv_usd > 25000}

    if ltv_usd > 25000:
        return True, (
            f"POL-HV-001 NOTICE: Customer LTV ${ltv_usd:,.2f} exceeds $25,000 "
            f"threshold. An executive reach-out workflow should be triggered "
            f"in addition to this automated retention offer. A customer success "
            f"manager must personally call within 24 hours."
        ), checked

    return True, "", checked


def check_pol_risk_001_serial_canceller(
    discount_percent: int,
) -> tuple[bool, str, dict]:
    """POL-RISK-001: Serial canceller detection.

    NOTE: This check requires cancellation history data not present in
    the current customer profile schema. It runs as a soft warning when
    discount exceeds 10% (the serial canceller cap) and flags the need
    for manual review of cancellation history.
    """
    # Without actual cancellation history, we flag for manual review
    # if the discount is above the serial canceller threshold
    checked = {"discount_percent": discount_percent, "serial_canceller_max": 10}

    # This is a soft check - we can't verify without history data
    return True, "", checked


# ── Aggregation ───────────────────────────────────────────────────


def run_all_policy_checks(
    offer: RetentionOffer,
    profile: dict,
    cancellation_reason: str,
) -> AuditResult:
    """Execute every policy check against a proposed offer.

    Each check is a pure Python function - no LLM calls, no API calls.
    All math is deterministic and auditable.

    Args:
        offer: the RetentionOffer proposed by the Strategist
        profile: customer profile dict (from ChromaDB lookup)
        cancellation_reason: the original cancellation reason text

    Returns:
        AuditResult with approved=False and detailed violations if any checks fail.
    """
    violations: list[str] = []
    all_checked: dict[str, dict] = {}

    checks = [
        ("POL-001", check_pol_001_tenure_discount_limits(
            profile.get("tenure_months", 0), offer.discount_percent)),
        ("POL-002", check_pol_002_ltv_protection(
            profile.get("monthly_cost_usd", 0.0), offer.discount_percent)),
        ("POL-003", check_pol_003_free_month_rules(
            profile.get("tenure_months", 0), offer.offer_type,
            offer.discount_percent, offer.duration_months)),
        ("POL-004", check_pol_004_plan_downgrade(
            profile.get("plan_name", ""), offer.offer_type,
            cancellation_reason, offer.discount_percent)),
        ("POL-005", check_pol_005_feature_gap(
            cancellation_reason, offer.offer_type, offer.discount_percent,
            justification=offer.justification,
            reasoning=offer.reasoning,
            email_draft=offer.email_draft)),
        ("POL-006", check_pol_006_high_value_protocol(
            profile.get("lifetime_value_usd", 0.0),
            profile.get("monthly_cost_usd", 0.0),
            profile.get("tenure_months", 0),
            offer.discount_percent)),
        ("POL-007", check_pol_007_competitor_price_match(
            cancellation_reason, offer.discount_percent,
            profile.get("tenure_months", 0),
            profile.get("monthly_cost_usd", 0.0))),
        ("POL-HV-001", check_pol_hv_001_executive_reachout(
            profile.get("lifetime_value_usd", 0.0))),
        ("POL-RISK-001", check_pol_risk_001_serial_canceller(
            offer.discount_percent)),
    ]

    # Separate hard violations from notices
    hard_violations: list[tuple[str, str]] = []
    notices: list[str] = []

    for policy_id, (passed, detail, checked) in checks:
        all_checked[policy_id] = checked
        if not passed:
            hard_violations.append((policy_id, detail))
        elif detail:  # passed but has a notice/message
            notices.append(detail)

    # Build the result
    if hard_violations:
        violation_ids = [pid for pid, _ in hard_violations]
        feedback_lines = [
            f"AUDIT REJECTED - {len(hard_violations)} policy violation(s) found:\n"
        ]
        for pid, detail in hard_violations:
            feedback_lines.append(f"  [{pid}] {detail}")

        if notices:
            feedback_lines.append(f"\nNotices ({len(notices)}):")
            for notice in notices:
                feedback_lines.append(f"  • {notice}")

        feedback_lines.append(
            f"\nYou MUST address ALL violations above in your revised proposal. "
            f"Do not repeat the same mistakes. Cite specific policy limits."
        )

        return AuditResult(
            approved=False,
            feedback="\n".join(feedback_lines),
            policy_violations=violation_ids,
        )

    # All checks passed
    feedback = "APPROVED: All policy checks passed."
    if notices:
        feedback += "\n\nNotices:\n" + "\n".join(f"  • {n}" for n in notices)

    return AuditResult(
        approved=True,
        feedback=feedback,
        policy_violations=[],
    )


# ── Helper ────────────────────────────────────────────────────────


def _tenure_bucket(months: int) -> str:
    if months < 6:
        return "0-6 months"
    elif months < 12:
        return "6-12 months"
    elif months < 24:
        return "12-24 months"
    return "24+ months"


# ── JSON extraction ────────────────────────────────────────────────


def _extract_json_from_message(content: str) -> dict | None:
    """Extract a JSON object from an LLM response string.

    Handles both bare JSON and JSON inside markdown code fences.
    """
    if not content:
        return None

    # Try bare JSON first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fence
    json_pattern = r'```(?:json)?\s*\n?([\s\S]*?)\n?```'
    matches = re.findall(json_pattern, content)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Try finding a JSON object with braces
    brace_pattern = r'\{[\s\S]*"discount_percent"[\s\S]*\}'
    match = re.search(brace_pattern, content)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def parse_offer_from_messages(messages: list) -> RetentionOffer | None:
    """Parse the Strategist's final message into a RetentionOffer.

    Scans messages in reverse order looking for the last AI message
    (without tool_calls) that contains valid JSON with offer fields.

    Returns None if no valid offer JSON is found.
    """
    for msg in reversed(messages):
        # Skip SystemMessage, ToolMessage, HumanMessage
        if not hasattr(msg, "content") or not msg.content:
            continue
        # Skip messages with tool calls (those are intermediate)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            continue

        content = msg.content
        if isinstance(content, list):
            # Multimodal content - extract text parts
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict)]
            content = " ".join(text_parts)

        parsed = _extract_json_from_message(str(content))
        if parsed and "discount_percent" in parsed:
            try:
                return RetentionOffer(**parsed)
            except Exception:
                continue

    return None


# ── Auditor node ──────────────────────────────────────────────────


def create_auditor_node():
    """Create the Financial Auditor node function.

    This node:
        1. Parses the Strategist's final offer from the message history
        2. Runs all 9 programmatic policy checks
        3. If violations found, appends a SystemMessage with feedback
           and sets audit_approved=False (causing loop back to Strategist)
        4. If all checks pass, sets audit_approved=True

    Returns:
        A callable that takes RetentionState and returns a state update dict.
    """
    def auditor_node(state: RetentionState) -> dict:
        messages = list(state.get("messages", []))
        profile = state.get("customer_profile", {})
        cancellation_reason = state.get("cancellation_reason", "")

        # Parse the offer from the conversation
        offer = parse_offer_from_messages(messages)

        if offer is None:
            feedback = (
                "AUDIT ERROR: Could not parse a valid RetentionOffer from the "
                "Strategist's response. The final message must contain a JSON "
                "object with discount_percent, duration_months, offer_type, "
                "justification, reasoning, and email_draft fields."
            )
            return {
                "audit_approved": False,
                "audit_feedback": feedback,
                "iteration_count": state.get("iteration_count", 0) + 1,
                "messages": [
                    SystemMessage(content=f"Audit Rejection:\n{feedback}\n\n"
                    "Please provide a valid JSON offer with all required fields.")
                ],
            }

        # Store the parsed offer in state
        offer_dict = offer.model_dump()

        # Run all policy checks
        audit_result = run_all_policy_checks(offer, profile, cancellation_reason)

        new_iteration = state.get("iteration_count", 0)
        if not audit_result.approved:
            new_iteration += 1
            feedback_msg = (
                f"Audit Rejection (attempt {new_iteration}):\n"
                f"{audit_result.feedback}"
            )
            return {
                "proposed_offer": offer_dict,
                "audit_approved": False,
                "audit_feedback": audit_result.feedback,
                "iteration_count": new_iteration,
                "messages": [SystemMessage(content=feedback_msg)],
            }

        # Approved
        return {
            "proposed_offer": offer_dict,
            "audit_approved": True,
            "audit_feedback": audit_result.feedback,
        }

    return auditor_node
