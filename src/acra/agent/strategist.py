"""Node 1: The Retention Strategist — a fully autonomous tool-calling agent.

This agent follows the ReAct (Reasoning + Acting) pattern:
    1. Receives the cancellation request and customer context
    2. Decides which tools to call and when (autonomous decisions)
    3. Calls get_customer_profile to retrieve account details
    4. Calls search_retention_policies to discover applicable policies
    5. Synthesizes findings into a personalized RetentionOffer

The agent outputs a JSON-formatted RetentionOffer in its final
message (when no more tool calls are needed). The graph's conditional
edge detects the absence of tool_calls and routes to the Auditor.
"""

import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from acra.agent.tools import get_customer_profile, search_retention_policies
from acra.agent.state import RetentionState

STRATEGIST_TOOLS = [get_customer_profile, search_retention_policies]

SYSTEM_PROMPT = """You are a senior Customer Retention Strategist at a SaaS company. You have access to tools that let you look up customer profiles and search company retention policies. You must use these tools to gather all necessary context before proposing an offer.

## Your Process

1. **Gather context**: Use get_customer_profile to load the customer's full account details. Use search_retention_policies to find applicable company policies based on the cancellation reason and customer profile.

2. **Analyze**: Consider the customer's tenure, plan, LTV, feature usage, payment history, and the cancellation reason. Cross-reference with all applicable policies you found.

3. **Propose**: Craft the best retention offer that maximizes retention probability while EXACTLY respecting all policy limits. You must justify your choices by citing specific policies.

## Important Guidelines

- Always search policies BEFORE proposing — never guess at limits
- Respect tenure-based discount limits EXACTLY as stated in policies
- Prefer plan downgrades over discounts for cost-concerned customers (POL-004)
- For feature-gap cancellations, offer tier trials, not discounts (POL-005)
- High-value customers (LTV > $10k) deserve premium treatment (POL-006)
- If you received previous audit feedback in the conversation, ADDRESS all violations in your new proposal
- The email must be specific and reference the customer's actual situation

## Final Output Format

When you have gathered all necessary context and are ready to propose, output ONLY a JSON object with exactly these fields (no other text):

```json
{
  "discount_percent": <int 0-100>,
  "duration_months": <int 1-12>,
  "offer_type": "<discount | free_months | plan_downgrade | feature_unlock>",
  "justification": "<brief explanation citing specific policies>",
  "reasoning": "<your full chain of thought: what you learned from tools, which policies apply, why you chose this specific offer>",
  "email_draft": "<the complete, warm, personalized retention email>"
}
```

Do NOT output JSON until you have called both tools and have all necessary context."""


def create_strategist_node():
    """Create the Strategist agent node function.

    The node calls the LLM with tools bound. If the LLM returns tool_calls,
    the graph's conditional edge routes to the tools node. If the LLM
    returns a final message (no tool_calls), it should contain a JSON
    RetentionOffer and the graph routes to the Auditor.

    The LLM client is created lazily on first invocation, so graph
    compilation does not require an API key.

    Returns:
        A callable that takes RetentionState and returns a state update dict.
    """
    _llm_with_tools = None

    def _get_llm():
        nonlocal _llm_with_tools
        if _llm_with_tools is None:
            llm = ChatOpenAI(
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                temperature=0.3,
                max_tokens=2048,
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            )
            _llm_with_tools = llm.bind_tools(STRATEGIST_TOOLS)
        return _llm_with_tools

    def strategist_node(state: RetentionState) -> dict:
        llm_with_tools = _get_llm()
        messages = list(state.get("messages", []))

        # Prepend system message if this is the first invocation
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        # Build the user task prompt
        customer_id = state.get("customer_id", "unknown")
        reason = state.get("cancellation_reason", "General dissatisfaction")
        iteration = state.get("iteration_count", 0)

        user_task = (
            f"Customer ID: {customer_id}\n"
            f"Cancellation reason: {reason}\n"
            f"Retry attempt: {iteration}\n\n"
            f"Use your tools to look up the customer profile and search for "
            f"relevant retention policies. Then propose the best retention offer."
        )

        messages.append(HumanMessage(content=user_task))

        response = llm_with_tools.invoke(messages)

        return {"messages": [response]}

    return strategist_node
