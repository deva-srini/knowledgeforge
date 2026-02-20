"""LangSmith observability integration for KnowledgeForge.

Provides a ``setup_tracing`` initialiser that configures LangSmith environment
variables, an ``is_tracing_enabled`` query function, and a ``traced`` decorator
that wraps functions with ``langsmith.traceable`` when tracing is active.

When tracing is disabled the decorator is a zero-overhead no-op — LangSmith is
never imported.
"""

import logging
import os
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

from app.core.config import KnowledgeForgeConfig

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_tracing_enabled: bool = False


def setup_tracing(config: KnowledgeForgeConfig) -> bool:
    """Configure LangSmith tracing from the application config.

    Sets ``LANGCHAIN_TRACING_V2`` and ``LANGCHAIN_PROJECT`` environment
    variables when tracing is enabled **and** a ``LANGCHAIN_API_KEY`` is
    present. Returns False (no-op) when disabled or when the API key is
    missing.

    Args:
        config: KnowledgeForge configuration object.

    Returns:
        True if tracing was successfully enabled, False otherwise.
    """
    global _tracing_enabled

    if not config.observability.langsmith_enabled:
        _tracing_enabled = False
        logger.debug("LangSmith tracing is disabled in config")
        return False

    api_key = os.environ.get("LANGCHAIN_API_KEY", "")
    if not api_key:
        _tracing_enabled = False
        logger.warning(
            "LangSmith tracing enabled in config but LANGCHAIN_API_KEY "
            "is not set — tracing will be disabled"
        )
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = config.observability.langsmith_project

    _tracing_enabled = True
    logger.info(
        "LangSmith tracing enabled for project '%s'",
        config.observability.langsmith_project,
    )
    return True


def is_tracing_enabled() -> bool:
    """Check whether LangSmith tracing is currently active.

    Returns:
        True if tracing was successfully set up, False otherwise.
    """
    return _tracing_enabled


def traced(
    run_type: str = "chain",
    name: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Callable[[F], F]:
    """Decorator that wraps a function with ``langsmith.traceable``.

    When tracing is disabled the original function is returned unchanged
    with zero overhead. When enabled, ``langsmith.traceable`` is lazily
    imported and applied.

    Args:
        run_type: The LangSmith run type (e.g. "chain", "tool").
        name: Optional human-readable name for the trace span.
        metadata: Optional metadata dict attached to the trace.

    Returns:
        A decorator that either wraps with traceable or is a no-op.
    """

    def decorator(fn: F) -> F:
        """Apply tracing wrapper if enabled."""

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Execute the function, optionally under a LangSmith trace."""
            if not _tracing_enabled:
                return fn(*args, **kwargs)

            from langsmith import traceable

            trace_kwargs: dict[str, Any] = {"run_type": run_type}
            if name is not None:
                trace_kwargs["name"] = name
            if metadata is not None:
                trace_kwargs["metadata"] = metadata

            traced_fn = traceable(**trace_kwargs)(fn)
            return traced_fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
