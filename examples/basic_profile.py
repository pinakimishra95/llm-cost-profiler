"""
basic_profile.py — @llmspy.profile decorator demo.

Run this with real API keys to see actual cost data:
    OPENAI_API_KEY=sk-... python examples/basic_profile.py

Or run without keys — llmspy will still show the report structure
(calls with 0 cost because the SDK isn't making real requests).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import llmspy

# ── Step 1: Decorate any function ─────────────────────────────────────────────

@llmspy.profile
def summarize_document(text: str) -> str:
    """Summarizes a document using OpenAI (or a stub if no key set)."""
    try:
        import openai
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize the following text in 2 sentences."},
                {"role": "user", "content": text},
            ],
            max_tokens=100,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[stub] Summary of: {text[:50]}..."


@llmspy.profile
def extract_entities(text: str) -> list[str]:
    """Extracts named entities using OpenAI (or a stub if no key set)."""
    try:
        import openai
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "List all named entities. One per line."},
                {"role": "user", "content": text},
            ],
            max_tokens=50,
        )
        return response.choices[0].message.content.splitlines()
    except Exception as e:
        return ["[stub] Entity1", "[stub] Entity2"]


# ── Step 2: Call your functions normally ──────────────────────────────────────

sample_text = """
Anthropic released Claude 3.5 Sonnet in June 2024, which outperformed GPT-4o on
several benchmarks including coding, reasoning, and vision tasks.
San Francisco-based Anthropic raised $750M in a Series C led by Google.
"""

print("Running summarize_document...")
summary = summarize_document(sample_text)
print(f"  → {summary}\n")

print("Running extract_entities...")
entities = extract_entities(sample_text)
print(f"  → {entities}\n")

# ── Step 3: Print the cost report ─────────────────────────────────────────────

llmspy.report()

# ── Bonus: access stats programmatically ──────────────────────────────────────
data = llmspy.stats()
print(f"\nProgrammatic access:")
print(f"  Total cost: ${data['total_cost_usd']:.4f}")
print(f"  Total tokens: {data['total_tokens']:,}")
print(f"  Calls by function: {data['by_function']}")
