"""
pricing.py — Built-in model pricing table (no API call needed).

Prices in USD per 1M tokens. Updated Feb 2026.
"""

from __future__ import annotations

# fmt: off
# (input_price_per_1M, output_price_per_1M)
PRICING: dict[str, tuple[float, float]] = {
    # Anthropic — Claude 4.x / Haiku 4.5
    "claude-opus-4-6":                  (15.00,  75.00),
    "claude-opus-4-5":                  (15.00,  75.00),
    "claude-sonnet-4-6":                ( 3.00,  15.00),
    "claude-sonnet-4-5":                ( 3.00,  15.00),
    "claude-haiku-4-5":                 ( 0.80,   4.00),
    "claude-haiku-4-5-20251001":        ( 0.80,   4.00),
    # Anthropic — Claude 3.x (legacy)
    "claude-3-5-sonnet-20241022":       ( 3.00,  15.00),
    "claude-3-5-haiku-20241022":        ( 0.80,   4.00),
    "claude-3-opus-20240229":           (15.00,  75.00),
    "claude-3-sonnet-20240229":         ( 3.00,  15.00),
    "claude-3-haiku-20240307":          ( 0.25,   1.25),
    # OpenAI — GPT-4o family
    "gpt-4o":                           ( 2.50,  10.00),
    "gpt-4o-2024-11-20":                ( 2.50,  10.00),
    "gpt-4o-mini":                      ( 0.15,   0.60),
    "gpt-4o-mini-2024-07-18":           ( 0.15,   0.60),
    # OpenAI — o-series reasoning
    "o1":                               (15.00,  60.00),
    "o1-mini":                          ( 3.00,  12.00),
    "o3-mini":                          ( 1.10,   4.40),
    # OpenAI — GPT-4 legacy
    "gpt-4-turbo":                      (10.00,  30.00),
    "gpt-4":                            (30.00,  60.00),
    "gpt-3.5-turbo":                    ( 0.50,   1.50),
    # Google — Gemini 1.5
    "gemini-1.5-pro":                   ( 1.25,   5.00),
    "gemini-1.5-flash":                 ( 0.075,  0.30),
    "gemini-1.5-flash-8b":              ( 0.0375, 0.15),
    # Google — Gemini 2.0
    "gemini-2.0-flash-exp":             ( 0.075,  0.30),
    "gemini-2.0-flash":                 ( 0.10,   0.40),
    # Meta — Llama (via API providers, approximate)
    "llama-3.1-70b-instruct":           ( 0.88,   0.88),
    "llama-3.1-8b-instruct":            ( 0.20,   0.20),
    # Mistral (via API)
    "mistral-large-latest":             ( 2.00,   6.00),
    "mistral-small-latest":             ( 0.20,   0.60),
}
# fmt: on

# Canonical "cheaper alternative" map for optimizer hints
CHEAPER_ALTERNATIVES: dict[str, str] = {
    "claude-opus-4-6":           "claude-sonnet-4-6",
    "claude-opus-4-5":           "claude-sonnet-4-5",
    "claude-sonnet-4-6":         "claude-haiku-4-5",
    "claude-sonnet-4-5":         "claude-haiku-4-5",
    "claude-3-5-sonnet-20241022":"claude-3-5-haiku-20241022",
    "claude-3-opus-20240229":    "claude-3-sonnet-20240229",
    "gpt-4o":                    "gpt-4o-mini",
    "gpt-4-turbo":               "gpt-4o",
    "gpt-4":                     "gpt-4o",
    "o1":                        "o3-mini",
    "gemini-1.5-pro":            "gemini-1.5-flash",
}


def calculate(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return cost in USD for the given model and token counts."""
    model_lower = model.lower()
    pricing = _lookup(model_lower)
    if pricing is None:
        return 0.0
    input_price, output_price = pricing
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def get_cheaper_alternative(model: str) -> str | None:
    """Return a cheaper model name if one is known, else None."""
    return CHEAPER_ALTERNATIVES.get(model.lower())


def get_price_per_million(model: str) -> tuple[float, float] | None:
    """Return (input_$/1M, output_$/1M) or None if unknown."""
    return _lookup(model.lower())


def _lookup(model_lower: str) -> tuple[float, float] | None:
    # Exact match first
    if model_lower in PRICING:
        return PRICING[model_lower]
    # Prefix match — handles versioned names like "gpt-4o-2024-05-13"
    for key, price in PRICING.items():
        if model_lower.startswith(key) or key.startswith(model_lower):
            return price
    return None


def list_models() -> list[str]:
    """Return sorted list of all known model names."""
    return sorted(PRICING.keys())
