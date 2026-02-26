"""Tests for flamegraph.py and optimizer.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tokenspy.tracker import CallRecord, Tracker


def _make_tracker(*records: dict) -> Tracker:
    t = Tracker()
    for kwargs in records:
        defaults = dict(
            function_name="fn",
            call_stack=["fn"],
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=200,
            cost_usd=0.007,
            duration_ms=350.0,
            provider="openai",
        )
        defaults.update(kwargs)
        t.record(CallRecord(**defaults))
    return t


class TestRenderText:
    def test_empty_tracker(self):
        from tokenspy.flamegraph import render_text

        t = Tracker()
        text = render_text(t)
        assert "no LLM calls recorded" in text

    def test_single_record(self):
        from tokenspy.flamegraph import render_text

        t = _make_tracker({"function_name": "summarize", "cost_usd": 0.05})
        text = render_text(t)
        assert "summarize" in text
        assert "$0.05" in text

    def test_multiple_functions(self):
        from tokenspy.flamegraph import render_text

        t = _make_tracker(
            {"function_name": "fn_a", "cost_usd": 0.05},
            {"function_name": "fn_b", "cost_usd": 0.02},
        )
        text = render_text(t)
        assert "fn_a" in text
        assert "fn_b" in text

    def test_shows_total(self):
        from tokenspy.flamegraph import render_text

        t = _make_tracker({"cost_usd": 0.042})
        text = render_text(t)
        assert "0.042" in text or "0.04" in text

    def test_percentages_shown(self):
        from tokenspy.flamegraph import render_text

        t = _make_tracker({"function_name": "only_fn", "cost_usd": 0.01})
        text = render_text(t)
        assert "100%" in text


class TestRenderHTML:
    def test_returns_html_string(self):
        from tokenspy.flamegraph import render_html

        t = _make_tracker({"function_name": "agent_run", "cost_usd": 0.03})
        html = render_html(t)
        assert "<html" in html
        assert "agent_run" in html

    def test_writes_file(self):
        from tokenspy.flamegraph import render_html

        t = _make_tracker({"function_name": "run_fn", "cost_usd": 0.01})
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            render_html(t, output_path=out)
            assert out.exists()
            content = out.read_text()
            assert "run_fn" in content

    def test_empty_tracker_returns_placeholder(self):
        from tokenspy.flamegraph import render_html

        t = Tracker()
        html = render_html(t)
        assert "No LLM calls recorded" in html

    def test_svg_present(self):
        from tokenspy.flamegraph import render_html

        t = _make_tracker({"cost_usd": 0.01})
        html = render_html(t)
        assert "<svg" in html


class TestOptimizer:
    def test_no_hints_for_empty_tracker(self):
        from tokenspy.optimizer import generate_hints

        t = Tracker()
        hints = generate_hints(t)
        assert hints == []

    def test_cheaper_model_hint(self):
        from tokenspy.optimizer import generate_hints

        # gpt-4o → gpt-4o-mini alternative
        t = _make_tracker({"model": "gpt-4o", "cost_usd": 0.01})
        hints = generate_hints(t)
        model_hints = [h for h in hints if "gpt-4o-mini" in h.suggestion]
        assert len(model_hints) >= 1

    def test_large_input_hint(self):
        from tokenspy.optimizer import generate_hints

        t = _make_tracker({"input_tokens": 12000, "cost_usd": 0.05})
        hints = generate_hints(t)
        token_hints = [h for h in hints if "tokens" in h.suggestion.lower()]
        assert len(token_hints) >= 1

    def test_hints_sorted_by_severity(self):
        from tokenspy.optimizer import generate_hints

        t = _make_tracker(
            {"model": "gpt-4o", "input_tokens": 15000, "cost_usd": 0.10},
        )
        hints = generate_hints(t)
        severities = [h.severity for h in hints]
        # "high" should come before "low"
        seen_low = False
        for s in severities:
            if s == "low":
                seen_low = True
            if seen_low and s == "high":
                pytest.fail("high severity hint appeared after low")

    def test_render_hints_returns_string(self):
        from tokenspy.optimizer import generate_hints, render_hints

        t = _make_tracker({"model": "gpt-4o", "cost_usd": 0.01})
        hints = generate_hints(t)
        rendered = render_hints(hints)
        assert isinstance(rendered, str)

    def test_render_hints_empty(self):
        from tokenspy.optimizer import render_hints

        assert render_hints([]) == ""
