"""Tests for pricing.py — no external dependencies."""


from tokenspy.pricing import (
    CHEAPER_ALTERNATIVES,
    calculate,
    get_cheaper_alternative,
    get_price_per_million,
    list_models,
)


class TestCalculate:
    def test_gpt4o_cost(self):
        # gpt-4o: $2.50/1M input, $10.00/1M output
        cost = calculate("gpt-4o", input_tokens=1_000_000, output_tokens=0)
        assert abs(cost - 2.50) < 0.001

    def test_gpt4o_output_cost(self):
        cost = calculate("gpt-4o", input_tokens=0, output_tokens=1_000_000)
        assert abs(cost - 10.00) < 0.001

    def test_claude_haiku_cheap(self):
        # haiku: $0.80/1M input
        cost = calculate("claude-haiku-4-5", input_tokens=1_000, output_tokens=200)
        expected = (1000 * 0.80 + 200 * 4.00) / 1_000_000
        assert abs(cost - expected) < 1e-8

    def test_unknown_model_returns_zero(self):
        cost = calculate("not-a-real-model-xyz", input_tokens=1000, output_tokens=500)
        assert cost == 0.0

    def test_zero_tokens(self):
        assert calculate("gpt-4o", 0, 0) == 0.0

    def test_versioned_model_prefix_match(self):
        # "gpt-4o-2024-11-20" should match via prefix
        cost_exact = calculate("gpt-4o", 1000, 500)
        cost_versioned = calculate("gpt-4o-2024-11-20", 1000, 500)
        assert abs(cost_exact - cost_versioned) < 1e-8

    def test_claude_opus_cost(self):
        # $15/1M input, $75/1M output
        cost = calculate("claude-opus-4-6", input_tokens=100_000, output_tokens=10_000)
        expected = (100_000 * 15.0 + 10_000 * 75.0) / 1_000_000
        assert abs(cost - expected) < 1e-6

    def test_case_insensitive(self):
        lower = calculate("gpt-4o", 1000, 500)
        upper = calculate("GPT-4O", 1000, 500)
        # Both should return same (both go through model_lower)
        assert lower == upper


class TestCheaperAlternative:
    def test_gpt4o_alternative(self):
        alt = get_cheaper_alternative("gpt-4o")
        assert alt == "gpt-4o-mini"

    def test_claude_opus_alternative(self):
        alt = get_cheaper_alternative("claude-opus-4-6")
        assert alt == "claude-sonnet-4-6"

    def test_unknown_model_returns_none(self):
        assert get_cheaper_alternative("some-random-model") is None

    def test_all_alternatives_are_in_pricing(self):
        """Every alt model in CHEAPER_ALTERNATIVES must exist in PRICING."""
        from tokenspy.pricing import PRICING

        for original, alt in CHEAPER_ALTERNATIVES.items():
            assert alt in PRICING, f"alt model {alt!r} for {original!r} not in PRICING"


class TestGetPricePerMillion:
    def test_returns_tuple(self):
        result = get_price_per_million("gpt-4o")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_unknown_returns_none(self):
        assert get_price_per_million("fantasy-model") is None


class TestListModels:
    def test_returns_list(self):
        models = list_models()
        assert isinstance(models, list)
        assert len(models) > 5

    def test_sorted(self):
        models = list_models()
        assert models == sorted(models)

    def test_known_models_present(self):
        models = list_models()
        assert "gpt-4o" in models
        assert "claude-opus-4-6" in models
        assert "gemini-1.5-pro" in models
