"""ACRA - Agentic Customer Churn & Retention Agent CLI.

Usage:
    acra                            Run interactive mode
    acra --customer CUST-001        Run retention flow for a specific customer
    acra --customer CUST-001 --reason "Too expensive"
    acra --demo                     Run all sample scenarios
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from acra.agent.graph import compile_graph
from acra.agent.state import RetentionState
from acra.data.customer_profiles import CUSTOMER_PROFILES

DEMO_SCENARIOS = [
    {
        "customer_id": "CUST-001",
        "reason": "Too expensive, I found a cheaper alternative",
        "description": "Mid-tenure professional plan user citing cost concerns",
    },
    {
        "customer_id": "CUST-002",
        "reason": "Missing feature X that the competitor has",
        "description": "New starter plan user citing feature gap",
    },
    {
        "customer_id": "CUST-003",
        "reason": "Budget cuts, need to reduce spending",
        "description": "Long-tenure enterprise high-LTV customer citing budget",
    },
    {
        "customer_id": "CUST-004",
        "reason": "Not getting enough value, too expensive for what I use",
        "description": "Mid-tenure professional user with low feature usage",
    },
    {
        "customer_id": "CUST-005",
        "reason": "Considering moving to a competitor with better pricing",
        "description": "Very long-tenure enterprise VIP citing competitor",
    },
]


def run_retention_flow(customer_id: str, cancellation_reason: str) -> dict:
    """Execute the full retention agent workflow for a single customer."""
    app = compile_graph()

    initial_state: RetentionState = {
        "customer_id": customer_id,
        "customer_name": "",
        "customer_email": "",
        "cancellation_reason": cancellation_reason,
        "customer_profile": {},
        "playbook_policies": [],
        "proposed_offer": {},
        "audit_approved": False,
        "audit_feedback": "",
        "iteration_count": 0,
        "final_email": "",
        "final_json": {},
    }

    result = app.invoke(initial_state)
    return result


def print_result(result: dict, scenario_desc: str = "") -> None:
    """Pretty-print the retention workflow result."""
    if scenario_desc:
        print(f"\n{'='*70}")
        print(f"  Scenario: {scenario_desc}")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")

    profile = result.get("customer_profile", {})
    offer = result.get("proposed_offer", {})

    print(f"\nCustomer: {profile.get('name', 'N/A')} ({profile.get('customer_id', 'N/A')})")
    print(f"Plan: {profile.get('plan_name', 'N/A')} | "
          f"Tenure: {profile.get('tenure_months', 0)} months | "
          f"LTV: ${profile.get('lifetime_value_usd', 0):.2f}")

    print(f"\nCancellation Reason: {result.get('cancellation_reason', 'N/A')}")

    print(f"\n--- Proposed Offer ---")
    print(f"Type:        {offer.get('offer_type', 'N/A')}")
    print(f"Discount:    {offer.get('discount_percent', 0)}%")
    print(f"Duration:    {offer.get('duration_months', 0)} months")
    print(f"Justification: {offer.get('justification', 'N/A')}")

    print(f"\n--- Audit Result ---")
    status = "APPROVED" if result.get("audit_approved") else "REJECTED"
    print(f"Status:   {status}")
    print(f"Feedback: {result.get('audit_feedback', 'N/A')}")
    print(f"Iterations: {result.get('iteration_count', 0) + 1}")

    print(f"\n--- Email Draft ---")
    print(result.get("final_email", offer.get("email_draft", "N/A")))

    print(f"\n--- DB Payload (JSON) ---")
    print(json.dumps(result.get("final_json", {}), indent=2))
    print(f"\n{'='*70}\n")


def interactive_mode() -> None:
    """Run the ACRA agent in interactive CLI mode."""
    print("\n" + "=" * 60)
    print("  ACRA - Agentic Customer Churn & Retention Agent")
    print("=" * 60)
    print("\nAvailable customer IDs:")
    for p in CUSTOMER_PROFILES:
        print(f"  {p['customer_id']} - {p['name']} ({p['plan_name']}, {p['tenure_months']}mo)")

    customer_id = input("\nEnter customer ID: ").strip()
    if not customer_id:
        print("No customer ID provided. Exiting.")
        return

    valid_ids = {p["customer_id"] for p in CUSTOMER_PROFILES}
    if customer_id not in valid_ids:
        print(f"Unknown customer ID: {customer_id}")
        print(f"Valid IDs: {', '.join(valid_ids)}")
        return

    reason = input("Cancellation reason: ").strip()
    if not reason:
        reason = "General dissatisfaction"

    print("\nRunning retention flow...")
    result = run_retention_flow(customer_id, reason)
    print_result(result)


def demo_mode() -> None:
    """Run all demo scenarios to showcase the agent's capabilities."""
    print("\n" + "=" * 60)
    print("  ACRA Demo - Running All Sample Scenarios")
    print("=" * 60)

    for i, scenario in enumerate(DEMO_SCENARIOS, 1):
        print(f"\n[{i}/{len(DEMO_SCENARIOS)}] Processing...")
        result = run_retention_flow(
            customer_id=scenario["customer_id"],
            cancellation_reason=scenario["reason"],
        )
        print_result(result, scenario["description"])

    print("Demo complete. All scenarios processed.")


def main():
    parser = argparse.ArgumentParser(
        description="ACRA - Agentic Customer Churn & Retention Agent",
    )
    parser.add_argument(
        "--customer", "-c",
        type=str,
        help="Customer ID to run retention flow for",
    )
    parser.add_argument(
        "--reason", "-r",
        type=str,
        default="General dissatisfaction",
        help="Cancellation reason",
    )
    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Run all demo scenarios",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Seed the ChromaDB database before running",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for JSON results (demo mode only)",
    )

    args = parser.parse_args()

    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Error: DEEPSEEK_API_KEY environment variable is not set.")
        print("Copy .env.example to .env and add your DeepSeek API key.")
        sys.exit(1)

    if args.seed:
        from acra.data.seed import main as seed_main
        seed_main()

    if args.demo:
        demo_mode()
    elif args.customer:
        result = run_retention_flow(args.customer, args.reason)
        print_result(result)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result.get("final_json", {}), f, indent=2)
            print(f"JSON output written to {args.output}")
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
