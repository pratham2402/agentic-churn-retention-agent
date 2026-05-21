"""Node 1: The Retention Strategist agent.

Reads the customer's cancellation reason and profile, queries the company
playbook via RAG, and proposes a personalized retention offer with an email draft.
"""

import json
import os
from langchain_openai import ChatOpenAI
from acra.models import RetentionOffer
from acra.agent.state import RetentionState


SYSTEM_PROMPT = """You are a senior Customer Retention Strategist at a SaaS company. Your job is to craft personalized, policy-compliant retention offers for customers who want to cancel their subscriptions.

## Your Process

1. Analyze the customer's profile: tenure, plan, LTV, feature usage, payment history
2. Review the company playbook policies provided to determine what offers are allowed
3. Craft the best possible offer WITHIN the policy limits
4. Write a warm, personalized retention email

## Important Guidelines

- Always respect tenure-based discount limits from the playbook
- Prefer plan downgrades over discounts for cost-concerned customers (per POL-004)
- For feature-gap cancellations, offer tier trials not discounts (per POL-005)
- High-value customers (LTV > $10k) deserve premium treatment (per POL-006)
- If you received previous audit feedback, ADDRESS all violations in your new proposal
- The email must be specific and reference the customer's actual usage and situation

## Output Format

Return a JSON object with these fields:
- discount_percent (int, 0-100)
- duration_months (int, 1-12)
- offer_type: "discount" | "free_months" | "plan_downgrade" | "feature_unlock"
- justification: brief explanation of why this offer fits the policies
- email_draft: the complete retention email to send
"""


def build_strategist_prompt(state: RetentionState) -> str:
    """Build the full prompt for the Strategist LLM call."""
    profile = state.get("customer_profile", {})
    policies = state.get("playbook_policies", [])
    previous_feedback = state.get("audit_feedback", "")

    policy_text = "\n\n".join([
        f"[{p.get('policy_id', 'N/A')}] {p.get('content', '')}"
        for p in policies
    ])

    feedback_block = ""
    if previous_feedback:
        feedback_block = f"""
## Previous Audit Rejection

Your last proposal was REJECTED. Here is the feedback:
{previous_feedback}

You MUST address all the issues above in your new proposal. Do not repeat the same mistakes.
"""

    return f"""## Customer Profile

{json.dumps(profile, indent=2)}

## Cancellation Reason

{state.get('cancellation_reason', 'Not provided')}

## Company Playbook Policies (from RAG)

{policy_text}
{feedback_block}
## Task

Propose a retention offer for this customer that maximizes retention probability while staying strictly within the company's published policies."""


def create_strategist_node():
    """Create the Retention Strategist LangGraph node function."""
    llm = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        temperature=0.3,
        max_tokens=1024,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    structured_llm = llm.with_structured_output(RetentionOffer, method="function_calling")

    def strategist_node(state: RetentionState) -> dict:
        prompt = build_strategist_prompt(state)
        result: RetentionOffer = structured_llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])

        return {
            "proposed_offer": result.model_dump(),
            "iteration_count": state.get("iteration_count", 0),
        }

    return strategist_node
