"""
providers/openai.py — Intercepts OpenAI SDK calls.

Patches openai.resources.chat.completions.Completions.create (sync)
and the async variant. Falls back gracefully if openai is not installed.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tokenspy.tracker import Tracker

_original_create: Any = None
_original_acreate: Any = None
_patched = False


def patch(tracker: Tracker, current_function: list[str]) -> bool:
    """Monkey-patch the OpenAI SDK. Returns True if successful."""
    global _original_create, _original_acreate, _patched

    try:
        from openai.resources.chat.completions import Completions
    except ImportError:
        return False

    if _patched:
        return True


    _original_create = Completions.create

    def _patched_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        response = _original_create(self, *args, **kwargs)
        duration_ms = (time.perf_counter() - start) * 1000

        _record(tracker, current_function, response, kwargs, duration_ms, "openai")
        return response

    Completions.create = _patched_create  # type: ignore[method-assign]

    # Async variant (best-effort)
    try:
        from openai.resources.chat.completions import AsyncCompletions

        _original_acreate = AsyncCompletions.create

        async def _patched_acreate(self: Any, *args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            response = await _original_acreate(self, *args, **kwargs)
            duration_ms = (time.perf_counter() - start) * 1000
            _record(tracker, current_function, response, kwargs, duration_ms, "openai")
            return response

        AsyncCompletions.create = _patched_acreate  # type: ignore[method-assign]
    except Exception:
        pass

    _patched = True
    return True


def unpatch() -> None:
    global _original_create, _original_acreate, _patched

    if not _patched:
        return

    try:
        from openai.resources.chat.completions import Completions

        if _original_create is not None:
            Completions.create = _original_create  # type: ignore[method-assign]

        from openai.resources.chat.completions import AsyncCompletions

        if _original_acreate is not None:
            AsyncCompletions.create = _original_acreate  # type: ignore[method-assign]
    except Exception:
        pass

    _patched = False
    _original_create = None
    _original_acreate = None


def _record(
    tracker: Tracker,
    current_function: list[str],
    response: Any,
    kwargs: dict,
    duration_ms: float,
    provider: str,
) -> None:
    try:
        from tokenspy import pricing
        from tokenspy.tracker import CallRecord

        model = kwargs.get("model", "unknown")
        usage = getattr(response, "usage", None)
        if usage is None:
            return

        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        cost = pricing.calculate(model, input_tokens, output_tokens)

        fn_name = current_function[0] if current_function else "<unknown>"
        tracker.record(
            CallRecord(
                function_name=fn_name,
                call_stack=list(current_function),
                model=model,
                provider=provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                duration_ms=duration_ms,
            )
        )
    except Exception:
        pass  # never crash user code
