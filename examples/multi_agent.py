"""
multi_agent.py — Multi-function cost profiling demo.

Simulates a three-stage pipeline (research → summarize → report) and
shows how llmspy attributes cost to each stage, with optimization hints.

    python examples/multi_agent.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import llmspy
from llmspy.tracker import CallRecord, get_global_tracker

# ── Simulated pipeline functions ───────────────────────────────────────────────

@llmspy.profile
def research_topic(topic: str) -> str:
    """Heavy research step — uses expensive model with large context."""
    _inject_fake_call("research_topic", model="gpt-4o", input_t=12000, output_t=800, cost=0.038)
    return f"[research on {topic}]"


@llmspy.profile
def summarize_findings(findings: str) -> str:
    """Summarization step — cheaper model."""
    _inject_fake_call("summarize_findings", model="gpt-4o-mini", input_t=4000, output_t=300, cost=0.0008)
    return "[summary]"


@llmspy.profile
def generate_report(summary: str) -> str:
    """Report generation — mid-tier model."""
    _inject_fake_call("generate_report", model="gpt-4o", input_t=3000, output_t=600, cost=0.0135)
    return "[report]"


def _inject_fake_call(fn: str, model: str, input_t: int, output_t: int, cost: float):
    """Injects a synthetic CallRecord (replaces real SDK call for the demo)."""
    get_global_tracker().record(
        CallRecord(
            function_name=fn,
            call_stack=[fn],
            model=model,
            input_tokens=input_t,
            output_tokens=output_t,
            cost_usd=cost,
            duration_ms=float(input_t // 10),
            provider="openai",
        )
    )


# ── Run the pipeline ───────────────────────────────────────────────────────────

print("Running multi-stage agent pipeline...\n")

findings = research_topic("LLM cost optimization techniques")
summary = summarize_findings(findings)
report = generate_report(summary)

print("Pipeline complete.\n")

# ── Print the flame graph ──────────────────────────────────────────────────────
llmspy.report()

# ── Optional: generate HTML report ────────────────────────────────────────────
from pathlib import Path

html_path = Path("/tmp/llmspy_demo_report.html")
llmspy.report(format="html", output=str(html_path))
print(f"\nHTML report: {html_path}")
