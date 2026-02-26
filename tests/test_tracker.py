"""Tests for tracker.py — no external dependencies."""

import tempfile
from pathlib import Path

from tokenspy.tracker import CallRecord, Tracker


def _make_record(**kwargs) -> CallRecord:
    defaults = dict(
        function_name="test_fn",
        call_stack=["test_fn"],
        model="gpt-4o",
        input_tokens=1000,
        output_tokens=200,
        cost_usd=0.004,
        duration_ms=350.0,
        provider="openai",
    )
    defaults.update(kwargs)
    return CallRecord(**defaults)


class TestTracker:
    def test_record_and_retrieve(self):
        t = Tracker()
        rec = _make_record()
        t.record(rec)
        assert len(t.records()) == 1
        assert t.records()[0].function_name == "test_fn"

    def test_multiple_records(self):
        t = Tracker()
        t.record(_make_record(cost_usd=0.01))
        t.record(_make_record(cost_usd=0.02))
        t.record(_make_record(cost_usd=0.03))
        assert len(t.records()) == 3

    def test_total_cost(self):
        t = Tracker()
        t.record(_make_record(cost_usd=0.01))
        t.record(_make_record(cost_usd=0.02))
        assert abs(t.total_cost() - 0.03) < 1e-9

    def test_total_tokens(self):
        t = Tracker()
        t.record(_make_record(input_tokens=1000, output_tokens=200))
        t.record(_make_record(input_tokens=500, output_tokens=100))
        assert t.total_tokens() == 1800

    def test_total_calls(self):
        t = Tracker()
        assert t.total_calls() == 0
        t.record(_make_record())
        assert t.total_calls() == 1

    def test_reset_clears_records(self):
        t = Tracker()
        t.record(_make_record())
        t.reset()
        assert len(t.records()) == 0
        assert t.total_cost() == 0.0

    def test_cost_by_function(self):
        t = Tracker()
        t.record(_make_record(function_name="fn_a", cost_usd=0.05))
        t.record(_make_record(function_name="fn_b", cost_usd=0.02))
        t.record(_make_record(function_name="fn_a", cost_usd=0.03))
        by_fn = t.cost_by_function()
        assert abs(by_fn["fn_a"] - 0.08) < 1e-9
        assert abs(by_fn["fn_b"] - 0.02) < 1e-9

    def test_cost_by_model(self):
        t = Tracker()
        t.record(_make_record(model="gpt-4o", cost_usd=0.05))
        t.record(_make_record(model="gpt-4o-mini", cost_usd=0.01))
        by_model = t.cost_by_model()
        assert "gpt-4o" in by_model
        assert "gpt-4o-mini" in by_model

    def test_summary_structure(self):
        t = Tracker()
        t.record(_make_record(cost_usd=0.01))
        summary = t.summary()
        assert "total_cost_usd" in summary
        assert "total_tokens" in summary
        assert "total_calls" in summary
        assert "calls" in summary
        assert "by_function" in summary
        assert "by_model" in summary
        assert len(summary["calls"]) == 1

    def test_thread_safety(self):
        import threading

        t = Tracker()
        errors = []

        def add_records():
            try:
                for _ in range(100):
                    t.record(_make_record())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_records) for _ in range(5)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert len(errors) == 0
        assert t.total_calls() == 500


class TestTrackerSQLitePersistence:
    def test_persist_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            t = Tracker(persist_path=db_path)
            rec = _make_record(function_name="persisted_fn", cost_usd=0.042)
            t.record(rec)

            # Load from a fresh tracker reading the same db
            t2 = Tracker(persist_path=db_path)
            loaded = t2.load_from_db()
            assert len(loaded) == 1
            assert loaded[0].function_name == "persisted_fn"
            assert abs(loaded[0].cost_usd - 0.042) < 1e-9

    def test_persist_creates_parent_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "nested" / "deep" / "usage.db"
            t = Tracker(persist_path=db_path)
            t.record(_make_record())
            assert db_path.exists()

    def test_load_from_nonexistent_db_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Point to a file that doesn't exist yet (directory exists, file does not)
            db_path = Path(tmpdir) / "missing.db"
            t = Tracker()
            t._persist_path = db_path  # set path without calling _init_db
            assert t.load_from_db() == []
