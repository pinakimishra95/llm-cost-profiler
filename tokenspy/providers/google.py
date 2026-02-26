"""
providers/google.py — Intercepts Google Generative AI SDK calls.

Patches google.generativeai.GenerativeModel.generate_content (sync).
Falls back gracefully if the package is not installed.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tokenspy.tracker import Tracker

_original_generate: Any = None
_patched = False


def patch(tracker: Tracker, current_function: list[str]) -> bool:
    """Monkey-patch the Google Generative AI SDK. Returns True if successful."""
    global _original_generate, _patched

    try:
        from google.generativeai import GenerativeModel
    except ImportError:
        return False

    if _patched:
        return True

    _original_generate = GenerativeModel.generate_content

    def _patched_generate(self: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        response = _original_generate(self, *args, **kwargs)
        duration_ms = (time.perf_counter() - start) * 1000
        _record(tracker, current_function, response, self, duration_ms, "google")
        return response

    GenerativeModel.generate_content = _patched_generate  # type: ignore[method-assign]
    _patched = True
    return True


def unpatch() -> None:
    global _original_generate, _patched

    if not _patched:
        return

    try:
        from google.generativeai import GenerativeModel

        if _original_generate is not None:
            GenerativeModel.generate_content = _original_generate  # type: ignore[method-assign]
    except Exception:
        pass

    _patched = False
    _original_generate = None


def _record(
    tracker: Tracker,
    current_function: list[str],
    response: Any,
    model_instance: Any,
    duration_ms: float,
    provider: str,
) -> None:
    try:
        from tokenspy import pricing
        from tokenspy.tracker import CallRecord

        # Get model name from the instance
        model_name = getattr(model_instance, "model_name", "unknown")
        # Strip "models/" prefix if present
        if "/" in model_name:
            model_name = model_name.split("/")[-1]

        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return

        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0
        cost = pricing.calculate(model_name, input_tokens, output_tokens)

        fn_name = current_function[0] if current_function else "<unknown>"
        tracker.record(
            CallRecord(
                function_name=fn_name,
                call_stack=list(current_function),
                model=model_name,
                provider=provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                duration_ms=duration_ms,
            )
        )
    except Exception:
        pass
