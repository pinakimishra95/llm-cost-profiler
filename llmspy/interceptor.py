"""
interceptor.py — Coordinates all provider monkey-patches.

The current_function list acts as a mutable shared reference so that
the profiler.py decorator can set the active function name which
the provider interceptors read when recording a call.
"""

from __future__ import annotations

from llmspy.tracker import Tracker, get_global_tracker

# Shared mutable list — providers read [0] to get the active function name.
# Using a list so it can be mutated in-place without rebinding.
_current_function: list[str] = ["<unknown>"]

# Whether any patches are currently active
_active = False


def activate(tracker: Tracker | None = None) -> None:
    """Activate all available provider interceptors."""
    global _active

    if tracker is None:
        tracker = get_global_tracker()

    from llmspy.providers import anthropic as _anthropic_provider
    from llmspy.providers import google as _google_provider
    from llmspy.providers import openai as _openai_provider

    _openai_provider.patch(tracker, _current_function)
    _anthropic_provider.patch(tracker, _current_function)
    _google_provider.patch(tracker, _current_function)

    _active = True


def deactivate() -> None:
    """Remove all provider patches."""
    global _active

    from llmspy.providers import anthropic as _anthropic_provider
    from llmspy.providers import google as _google_provider
    from llmspy.providers import openai as _openai_provider

    _openai_provider.unpatch()
    _anthropic_provider.unpatch()
    _google_provider.unpatch()

    _active = False


def set_current_function(name: str) -> None:
    """Set the name of the currently profiled function."""
    _current_function[0] = name


def get_current_function() -> str:
    return _current_function[0]


def is_active() -> bool:
    return _active
