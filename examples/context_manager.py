"""
context_manager.py — llmspy.session() context manager demo.

Perfect for one-off cost measurement without decorating every function.

    python examples/context_manager.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import llmspy

print("=== Context manager demo ===\n")

# Measure cost of a single block
with llmspy.session("research_pipeline") as session:
    # Simulate what would happen inside your actual agent:
    # (real calls would go here — we show how to use the API)
    print("  Running research pipeline inside session...")

    # Manually inject a fake record to demonstrate the API
    # (In real usage, just make normal SDK calls inside the `with` block)
    from llmspy.tracker import CallRecord
    session._tracker.record(
        CallRecord(
            function_name="research_pipeline",
            call_stack=["research_pipeline"],
            model="gpt-4o",
            input_tokens=8000,
            output_tokens=500,
            cost_usd=0.025,
            duration_ms=2100.0,
            provider="openai",
        )
    )
    session._tracker.record(
        CallRecord(
            function_name="research_pipeline",
            call_stack=["research_pipeline"],
            model="claude-haiku-4-5",
            input_tokens=2000,
            output_tokens=300,
            cost_usd=0.00280,
            duration_ms=800.0,
            provider="anthropic",
        )
    )

# After the `with` block — inspect results
print(f"\n  Session cost: {session.cost_str}")
print(f"  Tokens used: {session.tokens:,}")
print(f"  API calls:   {session.calls}")

print("\n  Full summary:")
summary = session.summary()
for model, cost in summary["by_model"].items():
    print(f"    {model}: ${cost:.4f}")
