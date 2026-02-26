"""
optimizer.py — Rule-based prompt optimization hints.

Analyzes tracked calls and suggests concrete ways to reduce cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from llmspy import pricing

if TYPE_CHECKING:
    from llmspy.tracker import CallRecord, Tracker

# Approximate monthly multiplier: assume 5 calls/min * 60 min * 8 hours * 22 days
# This is intentionally conservative; users can override.
CALLS_PER_MINUTE_DEFAULT = 5
MINUTES_PER_MONTH = 60 * 8 * 22  # ~10,560 minutes/month


@dataclass
class Hint:
    function_name: str
    current_model: str
    suggestion: str
    monthly_savings_usd: float | None
    severity: str  # "high" | "medium" | "low"

    def __str__(self) -> str:
        savings = (
            f" (~${self.monthly_savings_usd:.0f}/month)"
            if self.monthly_savings_usd
            else ""
        )
        icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(self.severity, "⚡")
        return f"  {icon} {self.function_name} [{self.current_model}]: {self.suggestion}{savings}"


def generate_hints(tracker: Tracker, calls_per_minute: float = CALLS_PER_MINUTE_DEFAULT) -> list[Hint]:
    """Analyse tracked records and return actionable optimization hints."""
    records = tracker.records()
    if not records:
        return []

    hints: list[Hint] = []

    # Group by function + model
    groups: dict[tuple[str, str], list[CallRecord]] = {}
    for r in records:
        key = (r.function_name, r.model)
        groups.setdefault(key, []).append(r)

    for (fn, model), recs in groups.items():
        avg_input = sum(r.input_tokens for r in recs) / len(recs)
        avg_output = sum(r.output_tokens for r in recs) / len(recs)
        total_cost = sum(r.cost_usd for r in recs)
        avg_cost = total_cost / len(recs)

        monthly_calls = calls_per_minute * MINUTES_PER_MONTH
        monthly_cost = avg_cost * monthly_calls

        # Hint 1: cheaper model available
        alt_model = pricing.get_cheaper_alternative(model)
        if alt_model:
            alt_cost = pricing.calculate(alt_model, int(avg_input), int(avg_output))
            alt_monthly = alt_cost * monthly_calls
            savings = monthly_cost - alt_monthly
            severity = "high" if savings > 100 else "medium" if savings > 20 else "low"
            hints.append(
                Hint(
                    function_name=fn,
                    current_model=model,
                    suggestion=(
                        f"Switch to {alt_model} — "
                        f"{savings / monthly_cost * 100:.0f}% cheaper per call"
                    ),
                    monthly_savings_usd=savings,
                    severity=severity,
                )
            )

        # Hint 2: large input prompts
        if avg_input > 4000:
            severity = "high" if avg_input > 10000 else "medium"
            hints.append(
                Hint(
                    function_name=fn,
                    current_model=model,
                    suggestion=(
                        f"Average input is {avg_input:,.0f} tokens. "
                        f"Trim context, use summarization, or limit retrieval chunks."
                    ),
                    monthly_savings_usd=None,
                    severity=severity,
                )
            )

        # Hint 3: high output tokens (model is being verbose)
        if avg_output > 2000:
            hints.append(
                Hint(
                    function_name=fn,
                    current_model=model,
                    suggestion=(
                        f"Average output is {avg_output:,.0f} tokens. "
                        f"Add 'Be concise' to your system prompt or lower max_tokens."
                    ),
                    monthly_savings_usd=None,
                    severity="low",
                )
            )

        # Hint 4: expensive model for a cheap task (output < 200 tokens)
        if avg_output < 200:
            alt_model = pricing.get_cheaper_alternative(model)
            if alt_model:
                hints.append(
                    Hint(
                        function_name=fn,
                        current_model=model,
                        suggestion=(
                            f"Short output ({avg_output:.0f} tokens avg) — "
                            f"likely fine with {alt_model} for classification/extraction."
                        ),
                        monthly_savings_usd=None,
                        severity="low",
                    )
                )

    # Sort: high severity first, then by monthly savings
    return sorted(
        hints,
        key=lambda h: (
            {"high": 0, "medium": 1, "low": 2}[h.severity],
            -(h.monthly_savings_usd or 0),
        ),
    )


def render_hints(hints: list[Hint]) -> str:
    if not hints:
        return ""
    lines = ["", "Optimization hints:"]
    for h in hints:
        lines.append(str(h))
    lines.append("")
    return "\n".join(lines)
