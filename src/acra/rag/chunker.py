"""LLM-based policy chunking for multi-vector RAG.

For each policy in the company retention playbook, generates:
    1. One concise, factual summary (for direct semantic matching)
    2. Three hypothetical questions that a user or agent might ask
       (for query-style matching - covers different phrasings)

These child documents are embedded separately in ChromaDB and linked
to the parent policy via metadata, enabling true multi-vector retrieval.
"""

import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

CHUNKER_SYSTEM_PROMPT = """You are a document processing assistant. Your task is to break down a company policy document into components that will be used for a multi-vector RAG (Retrieval-Augmented Generation) system.

For the given policy, produce exactly:

1. A **summary**: One concise paragraph (2-4 sentences) that accurately captures ALL the key rules, thresholds, and conditions in the policy. This must be factually complete - no details omitted.

2. Three **hypothetical questions**: Questions that a customer retention agent might ask that THIS specific policy would answer. Make the questions diverse - different phrasings, different angles. For example:
   - One question from the perspective of "can I do X?"
   - One question about specific thresholds or limits
   - One question about eligibility conditions

Output as a JSON object with keys "summary" and "questions" (list of 3 strings). No other text."""


def generate_policy_children(
    policy_id: str,
    policy_title: str,
    policy_content: str,
    llm: ChatOpenAI | None = None,
) -> list[Document]:
    """Generate child documents for a single policy using an LLM.

    Args:
        policy_id: unique policy identifier (e.g., "POL-001")
        policy_title: human-readable policy title
        policy_content: full text of the policy
        llm: ChatOpenAI instance (created if not provided)

    Returns:
        list of 4 Documents, each with metadata linking to the parent policy_id
    """
    if llm is None:
        llm = ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            temperature=0.1,
            max_tokens=512,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )

    user_prompt = f"""Policy ID: {policy_id}
Policy Title: {policy_title}

Policy Content:
{policy_content}"""

    response = llm.invoke([
        {"role": "system", "content": CHUNKER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])

    content = response.content.strip()
    # Handle potential markdown code fences in LLM output
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:]) if len(lines) > 1 else content
        if content.endswith("```"):
            content = content[:-3]

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: treat entire response as summary and generate basic questions
        parsed = {
            "summary": content[:500],
            "questions": [
                f"What are the rules for {policy_title.lower()}?",
                f"What limits does {policy_id} impose?",
                f"Who is eligible under {policy_id}?",
            ],
        }

    summary = parsed.get("summary", "").strip()
    questions = parsed.get("questions", [])[:3]
    while len(questions) < 3:
        questions.append(f"What does {policy_id} cover?")

    children = []
    if summary:
        children.append(Document(
            page_content=f"[{policy_id}] Summary: {summary}",
            metadata={
                "policy_id": policy_id,
                "chunk_type": "summary",
                "policy_title": policy_title,
            },
        ))

    for i, question in enumerate(questions):
        children.append(Document(
            page_content=f"[{policy_id}] {question}",
            metadata={
                "policy_id": policy_id,
                "chunk_type": "hypothetical_question",
                "question_index": i,
                "policy_title": policy_title,
            },
        ))

    return children
