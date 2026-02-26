"""
Shared fixtures and mocks for llmspy tests.

Stubs openai and anthropic in sys.modules so no real packages are needed.
"""

import sys
import types

import pytest


def _make_openai_stub() -> types.ModuleType:
    """Build a minimal openai module tree that the interceptor can patch."""
    openai = types.ModuleType("openai")

    # resources.chat.completions.Completions
    completions_mod = types.ModuleType("openai.resources.chat.completions")

    class Completions:
        def create(self, *args, **kwargs):  # noqa: D401
            raise RuntimeError("real openai not available in tests")

    class AsyncCompletions:
        async def create(self, *args, **kwargs):
            raise RuntimeError("real openai not available in tests")

    completions_mod.Completions = Completions
    completions_mod.AsyncCompletions = AsyncCompletions
    openai.resources = types.ModuleType("openai.resources")
    openai.resources.chat = types.ModuleType("openai.resources.chat")
    openai.resources.chat.completions = completions_mod

    sys.modules["openai"] = openai
    sys.modules["openai.resources"] = openai.resources
    sys.modules["openai.resources.chat"] = openai.resources.chat
    sys.modules["openai.resources.chat.completions"] = completions_mod

    return openai


def _make_anthropic_stub() -> types.ModuleType:
    """Build a minimal anthropic module tree that the interceptor can patch."""
    anthropic = types.ModuleType("anthropic")

    messages_mod = types.ModuleType("anthropic.resources.messages")

    class Messages:
        def create(self, *args, **kwargs):
            raise RuntimeError("real anthropic not available in tests")

    class AsyncMessages:
        async def create(self, *args, **kwargs):
            raise RuntimeError("real anthropic not available in tests")

    messages_mod.Messages = Messages
    messages_mod.AsyncMessages = AsyncMessages
    anthropic.resources = types.ModuleType("anthropic.resources")
    anthropic.resources.messages = messages_mod

    sys.modules["anthropic"] = anthropic
    sys.modules["anthropic.resources"] = anthropic.resources
    sys.modules["anthropic.resources.messages"] = messages_mod

    return anthropic


# Stub both providers once at import time so provider modules can import them.
_openai_stub = _make_openai_stub()
_anthropic_stub = _make_anthropic_stub()


@pytest.fixture(autouse=True)
def reset_global_tracker():
    """Reset the global tracker and deactivate all interceptors between tests."""
    from tokenspy import interceptor
    from tokenspy.tracker import Tracker, set_global_tracker

    set_global_tracker(Tracker())
    interceptor._active = False
    # Reset provider patch state
    from tokenspy.providers import anthropic as _ap
    from tokenspy.providers import openai as _op

    _op._patched = False
    _op._original_create = None
    _ap._patched = False
    _ap._original_create = None

    yield

    # Cleanup after test
    set_global_tracker(Tracker())
    interceptor._active = False
    _op._patched = False
    _op._original_create = None
    _ap._patched = False
    _ap._original_create = None
