# llmspy 🔥

**You're spending $800/month on LLMs. Which function is burning it? Find out in one line.**

[![PyPI version](https://badge.fury.io/py/llmspy.svg)](https://badge.fury.io/py/llmspy)
[![Tests](https://github.com/pinakimishra95/llm-cost-profiler/actions/workflows/tests.yml/badge.svg)](https://github.com/pinakimishra95/llm-cost-profiler/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Zero dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)]()

---

## The Problem

You get an invoice. It's $800 this month. No idea which function caused it.

```python
# You have no idea which of these is expensive:
def run_pipeline(query):
    docs = fetch_and_summarize(query)    # ← this one costs $600?
    entities = extract_entities(docs)   # ← or this one?
    return generate_report(entities)    # ← or this one?
```

Langfuse and Helicone require you to reroute all your traffic through their proxy. Sign up. Configure. Break your local dev setup.

**llmspy takes 1 line. No proxy. No signup. No traffic rerouting.**

---

## The Solution

```python
import llmspy

@llmspy.profile
def run_pipeline(query):
    docs = fetch_and_summarize(query)
    entities = extract_entities(docs)
    return generate_report(entities)

run_pipeline("Analyze Q3 earnings reports")
llmspy.report()
```

**Output:**

```
llmspy cost report — total: $0.0523  (18,734 tokens, 3 calls)
──────────────────────────────────────────────────────────────────────
  fetch_and_summarize                 $0.038  ████████████     73%
    └─ gpt-4o                         $0.038  ████████████     73%  [12,000 tokens]
  generate_report                     $0.011  ████             21%
    └─ gpt-4o                         $0.011  ████             21%  [3,600 tokens]
  extract_entities                    $0.003  █                6%
    └─ gpt-4o-mini                    $0.003  █                6%  [3,134 tokens]

Optimization hints:
  🔴 fetch_and_summarize [gpt-4o]: Switch to gpt-4o-mini — 82% cheaper per call (~$540/month)
  🟡 fetch_and_summarize [gpt-4o]: Average input is 12,000 tokens. Trim context or limit retrieval.
```

---

## Install

```bash
pip install llmspy
```

No dependencies. Works with whatever SDK you already have (`openai`, `anthropic`, or both).

---

## Quick Start

### Decorator

```python
import llmspy

@llmspy.profile
def my_agent(query: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": query}]
    )
    return response.choices[0].message.content

my_agent("What's the capital of France?")
llmspy.report()
```

### Context Manager

```python
with llmspy.session("research_task") as s:
    response = anthropic_client.messages.create(
        model="claude-haiku-4-5",
        messages=[{"role": "user", "content": "Summarize this document..."}]
    )

print(f"Cost: {s.cost_str}")    # "$0.0012"
print(f"Tokens: {s.tokens}")   # 3,240
```

### HTML Flame Graph

```python
llmspy.report(format="html")   # writes llmspy_report.html and opens in browser
```

### Persistent Tracking

```python
# Track costs across sessions — saves to ~/.llmspy/usage.db
llmspy.init(persist=True)

@llmspy.profile
def my_agent(query):
    ...
```

### Programmatic Access

```python
data = llmspy.stats()
# {
#   "total_cost_usd": 0.042,
#   "total_tokens": 15000,
#   "total_calls": 3,
#   "by_function": {"fetch_and_summarize": 0.038, ...},
#   "by_model": {"gpt-4o": 0.04, ...},
#   "calls": [...]
# }
```

---

## How It Works

llmspy monkey-patches your SDK clients **in-process** — no proxy, no HTTP interception, no configuration.

```
Your Code
    │
    ├─ @llmspy.profile ──→ sets "active function" name
    │
    └─ openai_client.chat.completions.create(...)
            │
            └─ llmspy interceptor (monkey-patch)
                    ├─ calls original SDK method
                    ├─ reads response.usage
                    ├─ looks up price in built-in pricing table
                    ├─ records: function, model, tokens, cost, duration
                    └─ returns response unchanged

llmspy.report() → renders flame graph from recorded data
```

This is the same technique used by `py-spy`, `line_profiler`, and other Python profilers. Zero runtime overhead on your hot path — just bookkeeping after the API call returns.

---

## Supported Providers

| Provider | SDK | Intercepted method |
|---|---|---|
| OpenAI | `openai>=1.0` | `chat.completions.create` |
| Anthropic | `anthropic>=0.30` | `messages.create` |
| Google | `google-generativeai>=0.7` | `generate_content` |

---

## Pricing Table

Built-in pricing for 30+ models — no API call needed.

| Model | Input ($/1M) | Output ($/1M) |
|---|---|---|
| claude-opus-4-6 | $15.00 | $75.00 |
| claude-sonnet-4-6 | $3.00 | $15.00 |
| claude-haiku-4-5 | $0.80 | $4.00 |
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| gemini-1.5-pro | $1.25 | $5.00 |
| gemini-1.5-flash | $0.075 | $0.30 |

[Full list](llmspy/pricing.py)

---

## Comparison

| | Langfuse | Helicone | LiteLLM | **llmspy** |
|---|---|---|---|---|
| Requires proxy | Yes | Yes | Yes | **No** |
| Requires signup | Yes | Yes | No | **No** |
| Flame graph output | No | No | No | **Yes** |
| cProfile-style decorator | No | No | No | **Yes** |
| Optimization hints | No | No | No | **Yes** |
| Local-first | No | No | Partial | **Yes** |
| Zero config | No | No | No | **Yes** |
| pip install | Yes | No | Yes | **Yes** |
| Zero dependencies | No | No | No | **Yes** |

---

## API Reference

| Function | Description |
|---|---|
| `@llmspy.profile` | Decorator: profile all LLM calls in the function |
| `llmspy.session(name)` | Context manager: profile calls in a block |
| `llmspy.report()` | Print text cost report |
| `llmspy.report(format="html")` | Write + open HTML flame graph |
| `llmspy.stats()` | Return cost dict for programmatic access |
| `llmspy.reset()` | Clear all recorded data |
| `llmspy.init(persist=True)` | Enable SQLite persistence across sessions |

---

## Roadmap

- [ ] Streaming response support
- [ ] LangChain/LangGraph integration
- [ ] Token budget alerts (`@llmspy.profile(budget_usd=0.10)`)
- [ ] CI cost reporting (GitHub Actions annotation)
- [ ] Cost comparison across git commits
- [ ] CLI: `llmspy history` / `llmspy report`

---

## Contributing

```bash
git clone https://github.com/pinakimishra95/llm-cost-profiler
cd llm-cost-profiler
pip install -e ".[dev]"
pytest tests/
```

---

## License

MIT. See [LICENSE](LICENSE).

---

**Star this repo** if you're tired of mystery LLM invoices. ⭐
