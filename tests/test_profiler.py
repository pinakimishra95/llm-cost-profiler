"""
Tests for the @profile decorator, session() context manager, and interceptor.

Uses the openai/anthropic stubs registered in conftest.py.
No real API keys needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import tokenspy
from tokenspy.tracker import CallRecord, Tracker, get_global_tracker, set_global_tracker

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_openai_response(model: str = "gpt-4o", prompt: int = 1000, completion: int = 200):
    """Build a mock OpenAI-style response object."""
    usage = MagicMock()
    usage.prompt_tokens = prompt
    usage.completion_tokens = completion
    resp = MagicMock()
    resp.usage = usage
    return resp


def _make_anthropic_response(model: str = "claude-haiku-4-5", input: int = 500, output: int = 100):
    """Build a mock Anthropic-style response object."""
    usage = MagicMock()
    usage.input_tokens = input
    usage.output_tokens = output
    resp = MagicMock()
    resp.usage = usage
    return resp


# ── Decorator tests ────────────────────────────────────────────────────────────

class TestProfileDecorator:
    def test_decorator_records_openai_call(self):
        from openai.resources.chat.completions import Completions

        tracker = Tracker()
        set_global_tracker(tracker)

        # Patch the Completions.create to return a mock response
        mock_response = _make_openai_response(model="gpt-4o", prompt=2000, completion=500)
        original = Completions.create

        def fake_create(self, *args, **kwargs):
            kwargs.setdefault("model", "gpt-4o")
            return mock_response

        Completions.create = fake_create

        try:
            @tokenspy.profile
            def my_fn():
                client = MagicMock()
                client.chat.completions = Completions()
                return Completions().create(model="gpt-4o", messages=[])

            my_fn()
        finally:
            Completions.create = original
            from tokenspy.providers import openai as _op2
            _op2._patched = False
            _op2._original_create = None

        records = tracker.records()
        assert len(records) == 1
        assert records[0].model == "gpt-4o"
        assert records[0].input_tokens == 2000
        assert records[0].output_tokens == 500
        assert records[0].cost_usd > 0

    def test_decorator_preserves_return_value(self):
        @tokenspy.profile
        def add(a, b):
            return a + b

        result = add(3, 4)
        assert result == 7

    def test_decorator_preserves_exceptions(self):
        @tokenspy.profile
        def broken():
            raise ValueError("oops")

        with pytest.raises(ValueError, match="oops"):
            broken()

    def test_decorator_sets_function_name(self):
        from tokenspy import interceptor

        names_seen = []

        @tokenspy.profile
        def named_function():
            names_seen.append(interceptor.get_current_function())

        named_function()
        assert "named_function" in names_seen[0]

    def test_decorator_restores_function_name_after_call(self):
        from tokenspy import interceptor

        interceptor.set_current_function("outer")

        @tokenspy.profile
        def inner_fn():
            pass

        inner_fn()
        # After the call, function name should be restored to "outer"
        assert interceptor.get_current_function() == "outer"


# ── Session / context manager tests ────────────────────────────────────────────

class TestSession:
    def test_session_cost_starts_at_zero(self):
        with tokenspy.session() as s:
            pass
        assert s.cost == 0.0

    def test_session_tracks_records(self):
        from tokenspy.tracker import CallRecord

        # Manually inject a record into the session tracker
        session_instance = tokenspy.profiler.Session(name="test")
        rec = CallRecord(
            function_name="test",
            call_stack=["test"],
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=200,
            cost_usd=0.007,
            duration_ms=100.0,
        )
        session_instance._tracker.record(rec)

        assert abs(session_instance.cost - 0.007) < 1e-9
        assert session_instance.cost_str == "$0.0070"
        assert session_instance.tokens == 1200

    def test_session_context_manager_yields_session(self):
        with tokenspy.session("my_session") as s:
            assert isinstance(s, tokenspy.Session)
            assert s.name == "my_session"


# ── Global API ─────────────────────────────────────────────────────────────────

class TestGlobalAPI:
    def test_stats_empty(self):
        tokenspy.reset()
        s = tokenspy.stats()
        assert s["total_cost_usd"] == 0.0
        assert s["total_calls"] == 0

    def test_reset_clears_tracker(self):
        tracker = get_global_tracker()
        rec = CallRecord(
            function_name="fn",
            call_stack=["fn"],
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
            duration_ms=50.0,
        )
        tracker.record(rec)
        assert tracker.total_calls() == 1

        tokenspy.reset()
        assert tracker.total_calls() == 0

    def test_report_text_no_crash_on_empty(self, capsys):
        tokenspy.reset()
        tokenspy.report()
        captured = capsys.readouterr()
        assert "no LLM calls recorded" in captured.out

    def test_report_text_shows_data(self, capsys):
        tracker = get_global_tracker()
        tracker.record(
            CallRecord(
                function_name="summarize",
                call_stack=["summarize"],
                model="gpt-4o",
                input_tokens=5000,
                output_tokens=500,
                cost_usd=0.0175,
                duration_ms=1200.0,
            )
        )
        tokenspy.report()
        captured = capsys.readouterr()
        assert "summarize" in captured.out
        assert "$" in captured.out


# ── Init ───────────────────────────────────────────────────────────────────────

class TestInit:
    def test_init_no_persist(self):
        # Should not raise
        tokenspy.init(persist=False)

    def test_init_with_persist(self, tmp_path):
        tokenspy.init(persist=True, persist_dir=str(tmp_path))
        # Check that the tracker has a persist path set
        tracker = get_global_tracker()
        assert tracker._persist_path is not None
        assert tracker._persist_path.exists()
