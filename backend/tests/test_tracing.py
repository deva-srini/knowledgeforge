"""Unit tests for the LangSmith tracing module.

Uses monkeypatch to control environment variables and mocks to avoid
real LangSmith API calls.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import KnowledgeForgeConfig, ObservabilityConfig
import app.observability.tracing as tracing_module
from app.observability.tracing import is_tracing_enabled, setup_tracing, traced


@pytest.fixture(autouse=True)
def reset_tracing_state(monkeypatch):
    """Reset the module-level tracing flag before each test."""
    tracing_module._tracing_enabled = False
    # Clean up any env vars we might set
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)


class TestSetupTracing:
    """Tests for setup_tracing."""

    def test_setup_tracing_disabled(self, monkeypatch):
        """Returns False and leaves env vars unchanged when disabled in config."""
        config = KnowledgeForgeConfig(
            observability=ObservabilityConfig(langsmith_enabled=False)
        )

        result = setup_tracing(config)

        assert result is False
        assert is_tracing_enabled() is False
        assert os.environ.get("LANGCHAIN_TRACING_V2") is None

    def test_setup_tracing_no_api_key(self, monkeypatch):
        """Returns False with warning when enabled but LANGCHAIN_API_KEY missing."""
        config = KnowledgeForgeConfig(
            observability=ObservabilityConfig(langsmith_enabled=True)
        )
        # Ensure no API key
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

        result = setup_tracing(config)

        assert result is False
        assert is_tracing_enabled() is False

    def test_setup_tracing_enabled(self, monkeypatch):
        """Sets env vars and returns True when enabled with API key present."""
        config = KnowledgeForgeConfig(
            observability=ObservabilityConfig(
                langsmith_enabled=True,
                langsmith_project="test-project",
            )
        )
        monkeypatch.setenv("LANGCHAIN_API_KEY", "test-key-123")

        result = setup_tracing(config)

        assert result is True
        assert is_tracing_enabled() is True
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
        assert os.environ.get("LANGCHAIN_PROJECT") == "test-project"


class TestTracedDecorator:
    """Tests for the traced decorator."""

    def test_traced_decorator_disabled(self):
        """Function runs normally without wrapping when tracing is disabled."""
        tracing_module._tracing_enabled = False

        @traced(run_type="chain", name="test_fn")
        def my_func(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y

        result = my_func(2, 3)
        assert result == 5

    def test_traced_decorator_enabled(self, monkeypatch):
        """Function is wrapped with langsmith.traceable when tracing is enabled."""
        tracing_module._tracing_enabled = True

        mock_traceable = MagicMock()
        # traceable() returns a decorator, which returns the wrapped fn
        mock_inner_decorator = MagicMock(side_effect=lambda fn: fn)
        mock_traceable.return_value = mock_inner_decorator

        with patch.dict("sys.modules", {"langsmith": MagicMock(traceable=mock_traceable)}):
            @traced(run_type="tool", name="my_tool")
            def my_func() -> str:
                """Return greeting."""
                return "hello"

            result = my_func()

        assert result == "hello"
        mock_traceable.assert_called_once_with(run_type="tool", name="my_tool")


class TestIsTracingEnabled:
    """Tests for is_tracing_enabled."""

    def test_is_tracing_enabled_reflects_state(self):
        """is_tracing_enabled returns the current module-level flag."""
        assert is_tracing_enabled() is False

        tracing_module._tracing_enabled = True
        assert is_tracing_enabled() is True

        tracing_module._tracing_enabled = False
        assert is_tracing_enabled() is False
