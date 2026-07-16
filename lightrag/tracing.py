from __future__ import annotations

import contextlib
import logging
import os
from dotenv import load_dotenv
from typing import Any

logger = logging.getLogger("lightrag")

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)

# Cached result of the last tracing-enabled check.  ``None`` means the cache
# is cold and must be recomputed on next access.  Call
# ``_invalidate_tracing_cache()`` from tests or after mutating env vars to
# force a recheck.
_tracing_enabled: bool | None = None


def _invalidate_tracing_cache() -> None:
    """Reset the cached tracing-enabled flag.

    Call this in test fixtures (or after changing env vars at runtime) to
    ensure the next ``is_tracing_enabled()`` call re-evaluates the current
    environment.
    """
    global _tracing_enabled
    _tracing_enabled = None


def _compute_tracing_enabled() -> bool:
    """Evaluate whether Langfuse tracing should be active right now.

    Checks, in order:
    1. The ``langfuse`` package must be importable.
    2. ``LANGFUSE_TRACING_ENABLED=false`` — explicit opt-out even when keys exist.
    3. ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` must both be set.
    """
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    langfuse_disabled = os.environ.get("LANGFUSE_TRACING_ENABLED", "true").lower() == "false"
    try:
        import langfuse  # noqa: F401
    except ImportError:
        if public_key or secret_key:
            logger.warning(
                "Langfuse keys are set but the langfuse package is not installed; tracing disabled. "
                "To enable tracing, install the langfuse package" 
            )
        return False
    if langfuse_disabled:
        logger.debug("Langfuse tracing explicitly disabled via LANGFUSE_TRACING_ENABLED")
        return False

    if not public_key or not secret_key:
        logger.warning(
            "Langfuse public or secret key not set; tracing disabled. "
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables to enable tracing."
        )
        return False
    logger.info("Langfuse tracing enabled")
    return True


def is_tracing_enabled() -> bool:
    """Return whether Langfuse tracing is currently enabled.

    The result is cached after the first evaluation.  Call
    ``_invalidate_tracing_cache()`` to force a recheck (e.g. in tests or
    after changing env vars at runtime).
    """
    global _tracing_enabled
    if _tracing_enabled is None:
        _tracing_enabled = _compute_tracing_enabled()
    return _tracing_enabled
    

def _get_max_output_chars() -> int:
    """Return the configured maximum characters for LLM outputs in Langfuse spans."""
    try:
        return int(os.environ.get("LANGFUSE_MAX_OUTPUT_CHARS", "2000"))
    except (ValueError, TypeError):
        return 2000
    
def lf_observe(**kwargs):
    """Conditionally apply Langfuse's ``@observe`` decorator.

    Tracing is resolved once at decoration/import time via
    :func:`is_tracing_enabled`.

    If tracing is enabled, this imports and applies
    :func:`langfuse.observe` with the provided keyword arguments.

    If tracing is disabled, the original function is returned unchanged,
    avoiding any Langfuse import, wrapping, or runtime overhead.

    This allows callers to use ``@lf_observe(...)`` unconditionally across
    the codebase while centralizing tracing configuration in one place.

    Notes
    -----
    * Tracing configuration is fixed at import time. Changing environment
      variables after module import will not affect already-decorated
      functions.
    """
    def decorator(func):
        if is_tracing_enabled():
            from langfuse import observe
            return observe(**kwargs)(func)
        return func

    return decorator


def lf_get_client() -> Any | None:
    """Return the active Langfuse client, or ``None`` if tracing is disabled."""
    if not is_tracing_enabled():
        return None
    from langfuse import get_client

    return get_client()


def lf_get_trace_url(trace_id: str) -> str | None:
    """Return a clickable Langfuse UI URL for *trace_id*.

    Delegates to the SDK's own ``Langfuse.get_trace_url()``, which resolves
    and caches the project id via this instance's own credentials — no
    project id needs to be configured/duplicated separately.

    The SDK builds the URL from ``LANGFUSE_HOST``/``LANGFUSE_BASE_URL``, which
    is also the address this process uses to *send* traces — in a Docker
    setup that's often an internal service hostname the browser can't reach.
    When ``LANGFUSE_PUBLIC_HOST`` is set, its scheme+host replace the SDK's
    for the returned URL so links are actually clickable from outside the
    container network; the path/query (project id, trace id) are untouched.

    Returns ``None`` when tracing is disabled or the SDK can't resolve a URL
    (e.g. project id lookup failed) — callers should treat this as
    best-effort.
    """
    client = lf_get_client()
    if client is None:
        return None
    try:
        url = client.get_trace_url(trace_id=trace_id)
    except Exception as exc:
        logger.debug("Failed to build Langfuse trace URL: %s", exc)
        return None
    if not url:
        return None

    public_host = os.environ.get("LANGFUSE_PUBLIC_HOST")
    if not public_host:
        return url

    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(url)
    public_parts = urlsplit(public_host.rstrip("/"))
    return urlunsplit(
        (
            public_parts.scheme or parts.scheme,
            public_parts.netloc or parts.netloc,
            parts.path,
            parts.query,
            parts.fragment,
        )
    )


def lf_update_current_span(**kwargs: Any) -> None:
    """Update the active Langfuse span's attributes (input, output, metadata, etc.).

    Centralises the ``client = lf_get_client(); if client: client.update_current_span(...)``
    guard that was copy-pasted across lightrag.py and pipeline.py.  Silent
    no-op when tracing is disabled or no span is active.
    """
    client = lf_get_client()
    if client is None:
        return
    try:
        client.update_current_span(**kwargs)
    except Exception as exc:
        logger.debug("Failed to update Langfuse span: %s", exc)


@contextlib.asynccontextmanager
async def lf_start_as_current_observation(**kwargs):
    """Async context manager that wraps a block in a named Langfuse span observation.

    Use this as a lightweight grouping envelope around LLM or embedding calls
    that are already auto-instrumented by ``langfuse.openai``.  The actual
    generation observation (model, tokens, latency, output) is created
    automatically by the OpenAI wrapper as a child of this span — no manual
    ``update_output`` or ``wrap_streaming_iterator`` calls are needed.

    When tracing is disabled or Langfuse is not installed, this is a transparent
    no-op so callers need no ``if client:`` guard.

    Args:
        name: Human-readable span name shown in the Langfuse UI.
        metadata: Additional key/value metadata attached to the span.

    Usage::

        async with start_as_current_observation(name="query-response", metadata={"mode": "hybrid"}):
            response = await use_model_func(query, stream=True)
    """
    
    client = lf_get_client()
    if client is None:
        yield
        return
    with client.start_as_current_observation(**kwargs):
        yield


@contextlib.asynccontextmanager
async def lf_propagate_attributes(
    user_id: str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, str] | None = None,
    trace_name: str | None = None,
):
    """Set trace-level attributes for all observations created in this context.

    Wraps ``langfuse.propagate_attributes`` with graceful degradation when
    tracing is disabled.

    When the Langfuse ``__enter__`` fails (e.g. network / auth error during
    context set-up) the body is still executed via a no-op yield so callers
    are not broken.  If user code inside the ``with`` block raises, that
    exception propagates normally — we do not double-yield.
    """
    if not is_tracing_enabled():
        yield
        return

    entered = False
    try:
        from langfuse import propagate_attributes

        with propagate_attributes(
            user_id=user_id,
            session_id=session_id,
            tags=tags,
            metadata=metadata,
            trace_name=trace_name,
        ):
            entered = True
            yield
    except Exception as exc:
        logger.warning("Langfuse propagate_attributes error: %s", exc)
        if not entered:
            # __enter__ failed before yielding control — yield the no-op path
            # so the calling ``with`` block can still execute.
            yield
        # If entered=True, user code raised; re-raise is implicit here.


def lf_get_current_trace_context() -> dict[str, str]:
    """Return the current Langfuse trace_id and observation_id for explicit context passing.

    When passed as kwargs to ``openai.chat.completions.create()``, the
    ``langfuse.openai`` auto-instrumentation will use these to attach the
    generation to the correct trace/span — bypassing the implicit OTel
    context resolution that can be unreliable across async boundaries.

    Returns an empty dict when tracing is disabled or no span is active,
    so callers can safely unpack with ``kwargs.update()``.
    """
    client = lf_get_client()
    if client is None:
        return {}
    try:
        trace_id = client.get_current_trace_id()
        observation_id = client.get_current_observation_id()
        if trace_id is None:
            return {}
        ctx: dict[str, str] = {"trace_id": trace_id}
        if observation_id is not None:
            ctx["parent_observation_id"] = observation_id
        return ctx
    except Exception as exc:
        logger.debug("Failed to get Langfuse trace context: %s", exc)
        return {}


def lf_score_current_span(scores: list[dict[str, Any]]) -> None:
    """Attach scores to the currently active Langfuse span.

    Each entry in *scores* is a dict with keys that map directly to
    ``client.score_current_span()``:

        - ``name``      (str, required)
        - ``value``     (float | bool | str, required)
        - ``data_type`` (str, optional — inferred if omitted)
        - ``comment``   (str, optional)

    Silent no-op when tracing is disabled or no span is active.
    """
    client = lf_get_client()
    if client is None:
        return
    for score in scores:
        try:
            client.score_current_span(**score)
        except Exception as exc:
            logger.debug("Failed to score Langfuse span (%s): %s", score.get("name"), exc)


def lf_flush() -> None:
    """Flush pending Langfuse events."""
    if not is_tracing_enabled():
        return
    try:
        client = lf_get_client()
        if client is not None:
            client.flush()
            logger.debug("Langfuse traces flushed")
    except Exception as exc:
        logger.warning("Failed to flush Langfuse traces: %s", exc)


def lf_shutdown() -> None:
    """Gracefully shut down the Langfuse client (flushes + waits for background threads)."""
    if not is_tracing_enabled():
        return
    try:
        client = lf_get_client()
        if client is not None:
            client.shutdown()
            logger.debug("Langfuse client shut down")
    except Exception as exc:
        logger.warning("Failed to shut down Langfuse client: %s", exc)
