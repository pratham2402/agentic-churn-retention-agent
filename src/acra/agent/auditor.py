"""Node 2: The Financial Auditor agent (The Guardrail).

Reviews the Strategist's proposed offer against hard company limits.
Approves compliant offers or rejects with specific, actionable feedback.
"""

import json
import os
from langchain_openai import ChatOpenAI
from acra.models import AuditResult
from acra.agent.state import RetentionState


SYSTEM_PROMPT = """You are a Financial Auditor at a SaaS company. Your job is to enforce retention offer policies with absolute precision. You are the guardrail that prevents revenue-destroying decisions.

## Your Mandate

Review every proposed retention offer against the company's published policies. You have the authority to REJECT any offer that violates policy limits.

## Policy Enforcement Rules

### 1. Tenure-Based Discount Limits (POL-001)
- 0-6 months tenure: MAX 20% discount
- 6-12 months tenure: MAX 30% discount
- 12-24 months tenure: MAX 40% discount
- 24+ months tenure: MAX 50% discount
- ABSOLUTE cap: never exceed 50% for anyone

### 2. LTV Protection (POL-002)
- Discounted monthly cost must stay above 30% of current monthly cost
- Monthly cost must never fall below $5.00 USD
- Check: current_cost * (1 - discount_pct/100) >= max(current_cost * 0.3, 5.00)

### 3. Free Month Rules (POL-003)
- Only for customers with 12+ months tenure
- Max 2 free months per retention event
- Free months CANNOT be combined with percentage discounts

### 4. Plan Downgrade Preference (POL-004)
- For cost-concern cancellations, verify downgrade was considered first
- Enterprise plan downgrades need special scrutiny

### 5. Feature Gap Handling (POL-005)
- Feature-gap cancellations should get tier trials, not discounts
- If agent proposed discount for a feature request, reject it

### 6. High-Value Protocol (POL-006)
- LTV > $10k or monthly > $500: verify premium treatment considered
- But tenure-based limits STILL APPLY even to high-value customers

## Output Format

Return a JSON object with:
- approved: true if offer passes ALL policies, false otherwise
- feedback: specific, actionable feedback for the strategist
- policy_violations: list of specific policy IDs violated
- adjusted_offer: if minor adjustments would fix it, suggest them (null if approved or if a full re-think is needed)
"""


def build_auditor_prompt(state: RetentionState) -> str:
    """Build the auditor's review prompt with full context."""
    profile = state.get("customer_profile", {})
    offer = state.get("proposed_offer", {})

    return f"""## Customer Profile

{json.dumps(profile, indent=2)}

## Proposed Offer (from Strategist)

{json.dumps(offer, indent=2)}

## Cancellation Reason

{state.get('cancellation_reason', 'Not provided')}

## Task

Audit this proposed retention offer against ALL company policies listed in your system prompt. Determine if it is compliant. If not, provide specific, actionable feedback explaining exactly what must change and which policies were violated."""


def create_auditor_node():
    """Create the Financial Auditor LangGraph node function."""
    llm = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        temperature=0.0,
        max_tokens=512,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    structured_llm = llm.with_structured_output(AuditResult, method="function_calling")

    def auditor_node(state: RetentionState) -> dict:
        prompt = build_auditor_prompt(state)
        result: AuditResult = structured_llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])

        return {
            "audit_approved": result.approved,
            "audit_feedback": result.feedback,
        }

    return auditor_node
