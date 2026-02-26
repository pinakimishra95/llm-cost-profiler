"""
profiler.py — @profile decorator and session() context manager.

Usage:
    import tokenspy

    @tokenspy.profile
    def run_agent():
        ...

    tokenspy.report()

    # Context manager
    with tokenspy.session() as s:
        ...
    print(s.cost)
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from tokenspy import interceptor as _interceptor
from tokenspy.tracker import CallRecord, Tracker, get_global_tracker, set_global_tracker

# ── Decorator ──────────────────────────────────────────────────────────────────

def profile(func: Callable) -> Callable:
    """Decorator that intercepts all LLM calls made inside the function.

    Can be used with or without arguments::

        @tokenspy.profile
        def my_fn(): ...

        @tokenspy.profile()
        def my_fn(): ...
    """
    # Support @tokenspy.profile() with parens (returns decorator)
    if func is None:
        return profile

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Ensure interceptors are active
        _interceptor.activate(get_global_tracker())
        prev = _interceptor.get_current_function()
        _interceptor.set_current_function(func.__qualname__)
        try:
            return func(*args, **kwargs)
        finally:
            _interceptor.set_current_function(prev)

    return wrapper


# ── Context manager / session ──────────────────────────────────────────────────

class Session:
    """A profiling session. Tracks calls made within a ``with`` block."""

    def __init__(self, name: str = "session") -> None:
        self.name = name
        self._tracker = Tracker()
        self._records: list[CallRecord] = []

    def __enter__(self) -> Session:
        _interceptor.activate(self._tracker)
        _interceptor.set_current_function(self.name)
        return self

    def __exit__(self, *_: Any) -> None:
        self._records = self._tracker.records()
        _interceptor.deactivate()

    @property
    def cost(self) -> float:
        """Total cost in USD."""
        return self._tracker.total_cost() or sum(r.cost_usd for r in self._records)

    @property
    def cost_str(self) -> str:
        return f"${self.cost:.4f}"

    @property
    def tokens(self) -> int:
        return self._tracker.total_tokens() or sum(r.total_tokens for r in self._records)

    @property
    def calls(self) -> int:
        return len(self._records)

    def summary(self) -> dict:
        return self._tracker.summary()


@contextmanager
def session(name: str = "session") -> Generator[Session, None, None]:
    """Context manager that profiles all LLM calls within the block."""
    s = Session(name=name)
    with s:
        yield s


# ── Global init ────────────────────────────────────────────────────────────────

def init(persist: bool = False, persist_dir: str | None = None) -> None:
    """Configure tokenspy global state.

    Args:
        persist: If True, all calls are persisted to a local SQLite database.
        persist_dir: Directory for the SQLite file. Defaults to ~/.tokenspy/.
    """
    if persist:
        db_dir = Path(persist_dir) if persist_dir else Path.home() / ".tokenspy"
        db_path = db_dir / "usage.db"
        tracker = Tracker(persist_path=db_path)
    else:
        tracker = Tracker()

    set_global_tracker(tracker)
    _interceptor.activate(tracker)
