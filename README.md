# ACRA - Agentic Customer Churn & Retention Agent

An intelligent agent that intercepts SaaS subscription cancellations and autonomously determines personalized retention offers using LangGraph, ChromaDB, and DeepSeek.

## Architecture

```
Customer Cancellation
        |
        v
+-------------------+
|  RAG Retrieval     |  <-- ChromaDB (profiles + playbook)
+-------------------+
        |
        v
+-------------------+
|  Node 1: Retention |  <-- Strategist Agent (DeepSeek)
|  Strategist        |      Proposes discount/offer
+-------------------+
        |
        v
+-------------------+
|  Node 2: Financial |  <-- Auditor Agent (DeepSeek)
|  Auditor           |      Validates against policies
+-------------------+
        |
   +----+----+
   |         |
  YES       NO (loop back, max 3 iterations)
   |
   v
+-------------------+
|  Finalize          |  <-- Generates email + JSON payload
+-------------------+
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | LangGraph (StateGraph with conditional edges) |
| RAG Architecture | Multi-Vector RAG with ChromaDB |
| LLM Orchestration | DeepSeek (deepseek-chat / deepseek-v4-pro) via LangChain |
| Structured Output | Pydantic models with function calling |
| Vector Database | ChromaDB (persistent) |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2, local) |

## Project Structure

```
acra/
├── src/acra/
│   ├── agent/
│   │   ├── graph.py          # LangGraph workflow (2-node cycle)
│   │   ├── state.py           # State schema (TypedDict)
│   │   ├── strategist.py      # Node 1: Retention Strategist
│   │   └── auditor.py         # Node 2: Financial Auditor
│   ├── rag/
│   │   ├── retriever.py       # Multi-vector RAG retrieval
│   │   ├── vector_store.py    # ChromaDB setup
│   │   └── embeddings.py      # Local sentence-transformer embeddings
│   ├── data/
│   │   ├── customer_profiles.py  # Sample customer data
│   │   ├── playbook.py        # Company retention policies
│   │   └── seed.py            # ChromaDB seeding script
│   ├── models/
│   │   └── __init__.py        # Pydantic schemas
│   └── main.py                # CLI entry point
└── tests/
    ├── test_graph.py
    ├── test_strategist.py
    ├── test_auditor.py
    └── test_models.py
```

## Setup

```bash
# Clone and enter project
cd acra

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install
pip install -e .

# Set your DeepSeek API key
cp .env.example .env
# Edit .env with your actual key

# Seed the vector database
make seed

# Run the demo
make run
```

## Usage

```bash
# Interactive mode
acra

# Single customer
acra --customer CUST-001 --reason "Too expensive"

# Run all demo scenarios
acra --demo

# Seed and run demo in one command
acra --seed --demo
```

## How It Works

### Node 1: Retention Strategist

Reads the customer's cancellation reason and profile from ChromaDB, searches the company playbook for applicable discount policies, and proposes a personalized retention offer (discount percentage, duration, type) along with a draft email.

### Node 2: Financial Auditor

Reviews the proposed offer against hard company limits: tenure-based discount caps, LTV protection floors, free month eligibility, and plan downgrade preferences. Approves compliant offers or rejects with specific feedback, forcing the Strategist to generate a new, policy-compliant alternative.

### The Loop

If the Auditor rejects an offer, the graph conditionally routes back to the Strategist with the rejection feedback. This loops up to 3 times, after which the best available offer is finalized regardless.

## Company Playbook (Governing Policies)

| Policy | Description |
|--------|-------------|
| POL-001 | Tenure-based maximum discount limits (20%-50%) |
| POL-002 | Lifetime value floor (30% of current rate minimum) |
| POL-003 | Free month eligibility and limits |
| POL-004 | Plan downgrade preference over discounts |
| POL-005 | Feature gap handling with tier trials |
| POL-006 | High-value customer premium protocols |
| POL-007 | Competitor price match protocol |

## Sample Customers

| ID | Name | Plan | Tenure | LTV |
|----|------|------|--------|-----|
| CUST-001 | Alice Johnson | Professional | 14mo | $686 |
| CUST-002 | Bob Williams | Starter | 3mo | $57 |
| CUST-003 | Carol Martinez | Enterprise | 36mo | $10,764 |
| CUST-004 | David Chen | Professional | 8mo | $392 |
| CUST-005 | Eva Thompson | Enterprise | 48mo | $14,352 |

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
