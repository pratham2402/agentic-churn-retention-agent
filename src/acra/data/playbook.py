"""Company retention playbook - policies, discount rules, and guardrails."""

RETENTION_PLAYBOOK = [
    {
        "policy_id": "POL-001",
        "category": "discount_limit",
        "title": "Maximum Discount by Tenure",
        "content": (
            "Customers with less than 6 months tenure may receive at most 20% discount. "
            "Customers with 6-12 months tenure may receive at most 30% discount. "
            "Customers with 12-24 months tenure may receive at most 40% discount. "
            "Customers with over 24 months tenure may receive at most 50% discount. "
            "Under no circumstances may any customer receive more than 50% discount."
        ),
        "rules": {
            "max_discount_by_tenure": {
                "0-6": 20,
                "6-12": 30,
                "12-24": 40,
                "24+": 50,
            },
            "absolute_max_discount": 50,
        },
    },
    {
        "policy_id": "POL-002",
        "category": "ltv_protection",
        "title": "Lifetime Value Floor",
        "content": (
            "No retention offer may reduce the effective monthly revenue below 30% of "
            "the customer's current monthly cost. The discounted monthly cost must remain "
            "above $5.00 USD in all cases. For customers with LTV over $5,000, offers "
            "may be more aggressive but must still respect tenure-based limits."
        ),
        "rules": {
            "min_monthly_revenue_pct": 30,
            "absolute_min_monthly_usd": 5.00,
            "high_ltv_threshold_usd": 5000,
        },
    },
    {
        "policy_id": "POL-003",
        "category": "free_months",
        "title": "Free Month Policy",
        "content": (
            "Free months may only be offered to customers with over 12 months tenure. "
            "Maximum 2 free months per retention event. Free months count as 100% discount "
            "for the months applied and must be factored into the overall discount calculation. "
            "Free months cannot be combined with percentage discounts in the same offer."
        ),
        "rules": {
            "min_tenure_for_free_months": 12,
            "max_free_months": 2,
            "cannot_combine_with_discount": True,
        },
    },
    {
        "policy_id": "POL-004",
        "category": "plan_downgrade",
        "title": "Plan Downgrade Retention",
        "content": (
            "When a customer cites cost concerns, offer a plan downgrade as first resort "
            "before offering discounts. Downgrades preserve more revenue than discounts. "
            "If the customer is already on the lowest tier, skip downgrade and proceed to "
            "a modest discount within tenure limits. Enterprise plan customers should never "
            "be downgraded without manager approval."
        ),
        "rules": {
            "prefer_downgrade_over_discount": True,
            "enterprise_requires_approval": True,
            "lowest_tier_plans": ["Starter", "Basic"],
        },
    },
    {
        "policy_id": "POL-005",
        "category": "feature_request",
        "title": "Feature Gap Retention",
        "content": (
            "When a customer cites missing features, offer a free trial of the next tier "
            "for 30 days instead of a discount. If the feature is on the public roadmap "
            "within 90 days, mention the expected release date. Do not offer discounts "
            "for feature requests - solve with product access instead."
        ),
        "rules": {
            "offer_tier_trial_not_discount": True,
            "max_trial_days": 30,
            "roadmap_mention_window_days": 90,
        },
    },
    {
        "policy_id": "POL-006",
        "category": "high_value_protocol",
        "title": "High-Value Customer Retention",
        "content": (
            "Customers with LTV over $10,000 or monthly spend over $500 are classified "
            "as high-value. For these customers, the retention agent may propose premium "
            "retention offers including dedicated support, custom feature prioritization, "
            "or extended free trials. However, all offers must still pass the Financial "
            "Auditor's tenure-based discount limits. High-value customers may receive "
            "one additional free month beyond standard limits."
        ),
        "rules": {
            "ltv_threshold": 10000,
            "monthly_spend_threshold": 500,
            "extra_free_month": 1,
            "perks": ["dedicated_support", "feature_prioritization", "extended_trial"],
        },
    },
    {
        "policy_id": "POL-007",
        "category": "competitor_response",
        "title": "Competitor Price Match Protocol",
        "content": (
            "When a customer cites a competitor's lower price, the agent may match up to "
            "the competitor's price if it falls within tenure-based discount limits. "
            "Price matching below 50% of current rate requires manager approval. "
            "Always highlight unique value propositions alongside any price match."
        ),
        "rules": {
            "max_price_match_discount": 50,
            "below_50pct_requires_manager": True,
            "must_highlight_uvp": True,
        },
    },
]

HIGH_VALUE_CUSTOMER_POLICIES = [
    {
        "policy_id": "POL-HV-001",
        "category": "executive_engagement",
        "title": "Executive Reach-out for High-Value Accounts",
        "content": (
            "For customers with LTV exceeding $25,000, trigger an executive reach-out "
            "workflow in addition to the automated retention offer. A customer success "
            "manager must personally call within 24 hours."
        ),
    },
]

RISKY_CUSTOMER_FLAGS = [
    {
        "policy_id": "POL-RISK-001",
        "category": "fraud_risk",
        "title": "Serial Cancellers",
        "content": (
            "Customers who have cancelled and returned more than twice in 12 months "
            "should receive at most 10% discount. These customers exhibit gaming behavior."
        ),
    },
]
