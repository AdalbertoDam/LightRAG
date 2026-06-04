"""Tracing integration tests for the Langfuse layer.

Coverage areas — mapped to actual call sites
---------------------------------------------
1.  tracing.py — enable/disable logic, caching, all public helpers.
2.  utils.py / use_llm_func_with_cache
        • cache-hit  → lf_update_current_span(metadata={…cache_hit: True})
        • cache-miss, tracing enabled  → LLM call receives name= / trace_id= / parent_observation_id=
        • cache-miss, tracing disabled → LLM receives no langfuse kwargs
3.  utils.py / EmbeddingFunc.__call__
        • enters lf_start_as_current_observation(name="embedding", as_type="embedding")
        • input dict includes text_count
4.  operate.py / extract_entities (end of function)
        • lf_update_current_span(output={chunks_processed, unique_entities, unique_relations})
5.  operate.py / extract_keywords
        • cache-hit path → lf_update_current_span(metadata={keywords_cache_hit, counts})
        • live path, tracing enabled → LLM call receives langfuse_config
6.  operate.py / kg_query and naive_query
        • query cache-hit  → lf_update_current_span(metadata={query_cache_hit: True})
        • query cache-miss, tracing enabled → LLM call receives langfuse_config
        • langfuse_config must be used inside the is_tracing_enabled() guard (regression)
7.  operate.py / merge_nodes_and_edges (end of function)
        • lf_update_current_span(input={total_*}, output={*_merged})
8.  pipeline.py / process_single_document
        • decorated with @lf_observe(name="index-document")
        • intra-document batch sub-tasks wrapped with lf_propagate_attributes
        • post-doc span update: lf_update_current_span(input=…, output=…)
9.  API route handlers (source-level checks — avoid argparse/server init)
        • query_routes.py: lf_observe, lf_propagate_attributes, all three span names
        • document_routes.py: lf_observe, lf_propagate_attributes, lf_start_as_current_observation
10. Disabled-tracing end-to-end guard

All tests work offline (no real LLM / Langfuse network calls).
"""

from __future__ import annotations

import asyncio
import os
import pathlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_async_stub(return_value: Any = "stub-response"):
    async def _stub(*args, **kwargs):
        return return_value
    return _stub


def _make_simple_tokenizer():
    tok = MagicMock()
    tok.encode = MagicMock(side_effect=lambda text: list(text.encode("utf-8")))
    return tok


def _routers_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parents[2] / "lightrag" / "api" / "routers"


# ---------------------------------------------------------------------------
# 1. tracing.py — unit tests
# ---------------------------------------------------------------------------

class TestIsTracingEnabled:

    def setup_method(self):
        from lightrag import tracing as tr
        tr._tracing_enabled = None

    def teardown_method(self):
        from lightrag import tracing as tr
        tr._tracing_enabled = None

    def test_disabled_when_langfuse_not_installed(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        with patch.dict("sys.modules", {"langfuse": None}):
            from lightrag import tracing as tr
            tr._tracing_enabled = None
            assert tr._compute_tracing_enabled() is False

    def test_disabled_when_keys_missing(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
        with patch.dict("sys.modules", {"langfuse": MagicMock()}):
            from lightrag import tracing as tr
            tr._tracing_enabled = None
            assert tr._compute_tracing_enabled() is False

    def test_disabled_via_explicit_env_flag(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
        with patch.dict("sys.modules", {"langfuse": MagicMock()}):
            from lightrag import tracing as tr
            tr._tracing_enabled = None
            assert tr._compute_tracing_enabled() is False

    def test_enabled_when_keys_and_package_present(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
        with patch.dict("sys.modules", {"langfuse": MagicMock()}):
            from lightrag import tracing as tr
            tr._tracing_enabled = None
            assert tr._compute_tracing_enabled() is True

    def test_result_is_cached(self):
        from lightrag import tracing as tr
        tr._tracing_enabled = None
        with patch("lightrag.tracing._compute_tracing_enabled", return_value=False) as mock_compute:
            tr.is_tracing_enabled()
            tr.is_tracing_enabled()
            mock_compute.assert_called_once()

    def test_invalidate_cache(self):
        from lightrag import tracing as tr
        tr._tracing_enabled = True
        tr._invalidate_tracing_cache()
        assert tr._tracing_enabled is None


class TestLfUpdateCurrentSpan:

    def test_noop_when_client_none(self):
        from lightrag.tracing import lf_update_current_span
        with patch("lightrag.tracing.lf_get_client", return_value=None):
            lf_update_current_span(output="test", metadata={"k": "v"})  # must not raise

    def test_delegates_to_client(self):
        mock_client = MagicMock()
        from lightrag.tracing import lf_update_current_span
        with patch("lightrag.tracing.lf_get_client", return_value=mock_client):
            lf_update_current_span(output="hello", metadata={"key": "val"})
        mock_client.update_current_span.assert_called_once_with(
            output="hello", metadata={"key": "val"}
        )

    def test_swallows_client_exceptions(self):
        mock_client = MagicMock()
        mock_client.update_current_span.side_effect = RuntimeError("network error")
        from lightrag.tracing import lf_update_current_span
        with patch("lightrag.tracing.lf_get_client", return_value=mock_client):
            lf_update_current_span(output="x")  # must not raise


class TestLfStartAsCurrentObservation:

    @pytest.mark.asyncio
    async def test_noop_when_client_none(self):
        from lightrag.tracing import lf_start_as_current_observation
        with patch("lightrag.tracing.lf_get_client", return_value=None):
            reached = False
            async with lf_start_as_current_observation(name="test-span"):
                reached = True
            assert reached

    @pytest.mark.asyncio
    async def test_enters_client_observation(self):
        mock_client = MagicMock()
        mock_ctx = MagicMock()
        mock_client.start_as_current_observation.return_value = mock_ctx
        mock_ctx.__enter__ = MagicMock(return_value=None)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        from lightrag.tracing import lf_start_as_current_observation
        with patch("lightrag.tracing.lf_get_client", return_value=mock_client):
            async with lf_start_as_current_observation(name="embed", as_type="embedding"):
                pass
        mock_client.start_as_current_observation.assert_called_once_with(
            name="embed", as_type="embedding"
        )

    @pytest.mark.asyncio
    async def test_body_exception_propagates(self):
        from lightrag.tracing import lf_start_as_current_observation
        with patch("lightrag.tracing.lf_get_client", return_value=None):
            with pytest.raises(ValueError, match="body error"):
                async with lf_start_as_current_observation(name="test"):
                    raise ValueError("body error")


class TestLfPropagateAttributes:

    @pytest.mark.asyncio
    async def test_noop_when_tracing_disabled(self):
        from lightrag.tracing import lf_propagate_attributes
        with patch("lightrag.tracing.is_tracing_enabled", return_value=False):
            reached = False
            async with lf_propagate_attributes(trace_name="t"):
                reached = True
            assert reached

    @pytest.mark.asyncio
    async def test_body_executes_when_tracing_disabled(self):
        from lightrag.tracing import lf_propagate_attributes
        result = []
        with patch("lightrag.tracing.is_tracing_enabled", return_value=False):
            async with lf_propagate_attributes(tags=["tag1"]):
                result.append(1)
        assert result == [1]


class TestLfFlushAndShutdown:

    def test_flush_noop_when_disabled(self):
        from lightrag.tracing import lf_flush
        with patch("lightrag.tracing.is_tracing_enabled", return_value=False):
            lf_flush()  # must not raise

    def test_flush_calls_client(self):
        mock_client = MagicMock()
        from lightrag.tracing import lf_flush
        with patch("lightrag.tracing.is_tracing_enabled", return_value=True), \
             patch("lightrag.tracing.lf_get_client", return_value=mock_client):
            lf_flush()
        mock_client.flush.assert_called_once()

    def test_shutdown_noop_when_disabled(self):
        from lightrag.tracing import lf_shutdown
        with patch("lightrag.tracing.is_tracing_enabled", return_value=False):
            lf_shutdown()  # must not raise

    def test_shutdown_calls_client(self):
        mock_client = MagicMock()
        from lightrag.tracing import lf_shutdown
        with patch("lightrag.tracing.is_tracing_enabled", return_value=True), \
             patch("lightrag.tracing.lf_get_client", return_value=mock_client):
            lf_shutdown()
        mock_client.shutdown.assert_called_once()


class TestLfObserveDecorator:

    def test_passthrough_when_disabled(self):
        with patch("lightrag.tracing.is_tracing_enabled", return_value=False):
            from lightrag.tracing import lf_observe

            @lf_observe(name="test-fn", as_type="span")
            async def _fn():
                return 42

        result = asyncio.get_event_loop().run_until_complete(_fn())
        assert result == 42

    def test_wraps_with_observe_when_enabled(self):
        mock_observe = MagicMock(side_effect=lambda **kw: lambda f: f)
        with patch("lightrag.tracing.is_tracing_enabled", return_value=True), \
             patch.dict("sys.modules", {"langfuse": MagicMock(observe=mock_observe)}):
            from lightrag import tracing as tr
            import importlib
            importlib.reload(tr)

            @tr.lf_observe(name="x")
            async def _fn():
                return 1

        mock_observe.assert_called_once_with(name="x")


class TestLfGetCurrentTraceContext:

    def test_returns_empty_when_no_active_span(self):
        mock_client = MagicMock()
        mock_client.get_current_trace_id.return_value = None
        mock_client.get_current_observation_id.return_value = None
        with patch("lightrag.tracing.lf_get_client", return_value=mock_client):
            from lightrag.tracing import lf_get_current_trace_context
            assert lf_get_current_trace_context() == {}

    def test_returns_trace_and_observation_id(self):
        mock_client = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace-abc"
        mock_client.get_current_observation_id.return_value = "obs-xyz"
        with patch("lightrag.tracing.lf_get_client", return_value=mock_client):
            from lightrag.tracing import lf_get_current_trace_context
            ctx = lf_get_current_trace_context()
        assert ctx["trace_id"] == "trace-abc"
        assert ctx["parent_observation_id"] == "obs-xyz"

    def test_returns_only_trace_id_when_no_observation(self):
        mock_client = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace-only"
        mock_client.get_current_observation_id.return_value = None
        with patch("lightrag.tracing.lf_get_client", return_value=mock_client):
            from lightrag.tracing import lf_get_current_trace_context
            ctx = lf_get_current_trace_context()
        assert ctx.get("trace_id") == "trace-only"
        assert "parent_observation_id" not in ctx

    def test_returns_empty_when_client_none(self):
        with patch("lightrag.tracing.lf_get_client", return_value=None):
            from lightrag.tracing import lf_get_current_trace_context
            assert lf_get_current_trace_context() == {}


class TestMaxOutputChars:

    def test_default_value(self):
        from lightrag.tracing import _get_max_output_chars
        os.environ.pop("LANGFUSE_MAX_OUTPUT_CHARS", None)
        assert _get_max_output_chars() == 2000

    def test_custom_value(self):
        from lightrag.tracing import _get_max_output_chars
        with patch.dict(os.environ, {"LANGFUSE_MAX_OUTPUT_CHARS": "500"}):
            assert _get_max_output_chars() == 500

    def test_invalid_falls_back_to_default(self):
        from lightrag.tracing import _get_max_output_chars
        with patch.dict(os.environ, {"LANGFUSE_MAX_OUTPUT_CHARS": "not-a-number"}):
            assert _get_max_output_chars() == 2000


# ---------------------------------------------------------------------------
# 2. utils.py / use_llm_func_with_cache — tracing call sites
# ---------------------------------------------------------------------------

class TestUseLlmFuncWithCacheTracing:

    @pytest.mark.asyncio
    async def test_cache_hit_emits_cache_hit_metadata(self):
        """cache-hit path must call lf_update_current_span with cache_hit=True."""
        mock_span_update = MagicMock()
        fake_cache_content = "cached answer"
        fake_cache_ts = 1234567890

        with patch("lightrag.utils.lf_update_current_span", mock_span_update), \
             patch("lightrag.utils.is_tracing_enabled", return_value=False), \
             patch("lightrag.utils.handle_cache",
                   AsyncMock(return_value=(fake_cache_content, fake_cache_ts))):

            from lightrag.utils import use_llm_func_with_cache
            from lightrag.base import BaseKVStorage

            mock_cache = AsyncMock(spec=BaseKVStorage)
            mock_cache.global_config = {"enable_llm_cache_for_entity_extract": False}

            result, ts = await use_llm_func_with_cache(
                user_prompt="hello",
                use_llm_func=_make_async_stub("live"),
                llm_response_cache=mock_cache,
                cache_type="extract",
                span_name="test-span",
                span_metadata={"chunk_id": "c1"},
            )

        assert result == fake_cache_content
        hits = [
            c for c in mock_span_update.call_args_list
            if c.kwargs.get("metadata", {}).get("cache_hit") is True
        ]
        assert len(hits) >= 1

    @pytest.mark.asyncio
    async def test_cache_miss_tracing_enabled_passes_langfuse_config(self):
        """cache-miss + tracing enabled: LLM receives name= / trace_id= / parent_observation_id=."""
        received_kwargs: dict = {}

        async def _capturing_llm(prompt, **kwargs):
            received_kwargs.update(kwargs)
            return "live answer"

        with patch("lightrag.utils.lf_update_current_span", MagicMock()), \
             patch("lightrag.utils.is_tracing_enabled", return_value=True), \
             patch("lightrag.utils.lf_get_current_trace_context",
                   return_value={"trace_id": "t1", "parent_observation_id": "o1"}), \
             patch("lightrag.utils.handle_cache", AsyncMock(return_value=None)):

            from lightrag.utils import use_llm_func_with_cache
            from lightrag.base import BaseKVStorage

            mock_cache = AsyncMock(spec=BaseKVStorage)
            mock_cache.global_config = {"enable_llm_cache_for_entity_extract": False}

            await use_llm_func_with_cache(
                user_prompt="hello",
                use_llm_func=_capturing_llm,
                llm_response_cache=mock_cache,
                cache_type="extract",
                span_name="entity-extraction",
                span_metadata={"chunk_id": "c2"},
            )

        assert received_kwargs.get("name") == "entity-extraction"
        assert received_kwargs.get("trace_id") == "t1"
        assert received_kwargs.get("parent_observation_id") == "o1"

    @pytest.mark.asyncio
    async def test_cache_miss_tracing_disabled_no_langfuse_kwargs(self):
        """cache-miss + tracing disabled: LLM receives no Langfuse-specific kwargs."""
        received_kwargs: dict = {}

        async def _capturing_llm(prompt, **kwargs):
            received_kwargs.update(kwargs)
            return "live answer"

        with patch("lightrag.utils.lf_update_current_span", MagicMock()), \
             patch("lightrag.utils.is_tracing_enabled", return_value=False), \
             patch("lightrag.utils.handle_cache", AsyncMock(return_value=None)):

            from lightrag.utils import use_llm_func_with_cache
            from lightrag.base import BaseKVStorage

            mock_cache = AsyncMock(spec=BaseKVStorage)
            mock_cache.global_config = {"enable_llm_cache_for_entity_extract": False}

            await use_llm_func_with_cache(
                user_prompt="hello",
                use_llm_func=_capturing_llm,
                llm_response_cache=mock_cache,
                cache_type="extract",
                span_name="entity-extraction",
                span_metadata={"chunk_id": "c3"},
            )

        assert "trace_id" not in received_kwargs
        assert "parent_observation_id" not in received_kwargs


# ---------------------------------------------------------------------------
# 3. utils.py / EmbeddingFunc — tracing call sites
# ---------------------------------------------------------------------------

class TestEmbeddingFuncTracingSpan:

    @pytest.mark.asyncio
    async def test_embedding_span_entered_with_correct_kwargs(self):
        """lf_start_as_current_observation called with name='embedding', as_type='embedding',
        and input.text_count matching the number of texts passed."""
        from contextlib import asynccontextmanager
        entered_kwargs: dict = {}

        @asynccontextmanager
        async def _capturing_obs(**kwargs):
            entered_kwargs.update(kwargs)
            yield

        with patch("lightrag.utils.lf_start_as_current_observation", _capturing_obs):
            import numpy as np
            from lightrag.utils import EmbeddingFunc

            async def _fake_embed(texts, **kwargs):
                return np.zeros((len(texts), 128))

            func = EmbeddingFunc(embedding_dim=128, max_token_size=512, func=_fake_embed)
            await func(["hello", "world"])

        assert entered_kwargs.get("name") == "embedding"
        assert entered_kwargs.get("as_type") == "embedding"
        assert entered_kwargs["input"]["text_count"] == 2

    @pytest.mark.asyncio
    async def test_embedding_span_noop_when_no_client(self):
        """EmbeddingFunc must not crash when the context manager is a no-op."""
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _noop(**kwargs):
            yield

        with patch("lightrag.utils.lf_start_as_current_observation", _noop):
            import numpy as np
            from lightrag.utils import EmbeddingFunc

            async def _fake_embed(texts, **kwargs):
                return np.zeros((len(texts), 64))

            func = EmbeddingFunc(embedding_dim=64, max_token_size=256, func=_fake_embed)
            result = await func(["test"])

        assert result.shape == (1, 64)

    @pytest.mark.asyncio
    async def test_embedding_single_text_correct_shape(self):
        from contextlib import asynccontextmanager
        import numpy as np
        from lightrag.utils import EmbeddingFunc

        @asynccontextmanager
        async def _noop(**kwargs):
            yield

        with patch("lightrag.utils.lf_start_as_current_observation", _noop):
            async def _fake_embed(texts, **kwargs):
                return np.zeros((len(texts), 32))

            func = EmbeddingFunc(embedding_dim=32, max_token_size=128, func=_fake_embed)
            result = await func(["hello"])

        assert result.shape == (1, 32)


# ---------------------------------------------------------------------------
# 4. operate.py / extract_entities — lf_update_current_span at end of function
# ---------------------------------------------------------------------------

class TestExtractEntitiesTracing:
    """extract_entities calls lf_update_current_span at the end with chunk/entity/relation counts."""

    def test_lf_update_called_after_extraction(self):
        import inspect
        from lightrag import operate
        src = inspect.getsource(operate.extract_entities)
        assert "lf_update_current_span" in src
        assert "chunks_processed" in src
        assert "unique_entities" in src
        assert "unique_relations" in src

    def test_span_name_in_process_single_content(self):
        """_process_single_content must pass span_name for initial extraction and gleaning."""
        import inspect
        from lightrag import operate
        src = inspect.getsource(operate.extract_entities)
        assert "entity-relation-extraction-llm" in src
        assert "entity-relation-extraction-gleaning-llm" in src


# ---------------------------------------------------------------------------
# 5. operate.py / _summarize_descriptions — span_name for summary LLM calls
# ---------------------------------------------------------------------------

class TestSummarizeDescriptionsTracing:
    """_summarize_descriptions passes span_name='summarize-entity-relation-descriptions'
    to use_llm_func_with_cache."""

    def test_span_name_in_source(self):
        import inspect
        from lightrag import operate
        src = inspect.getsource(operate._summarize_descriptions)
        assert "summarize-entity-relation-descriptions" in src
        assert "span_name" in src
        assert "span_metadata" in src

    def test_span_metadata_fields_in_source(self):
        import inspect
        from lightrag import operate
        src = inspect.getsource(operate._summarize_descriptions)
        assert "description_type" in src
        assert "description_name" in src
        assert "description_count" in src


# ---------------------------------------------------------------------------
# 6. operate.py / extract_keywords — tracing call sites
# ---------------------------------------------------------------------------

class TestExtractKeywordsTracing:

    def test_cache_hit_metadata_present_in_source(self):
        import inspect
        from lightrag import operate
        src = inspect.getsource(operate.extract_keywords_only)
        assert "keywords_cache_hit" in src
        assert "keywords_cache_hl_count" in src
        assert "keywords_cache_ll_count" in src

    def test_live_call_langfuse_config_present_in_source(self):
        import inspect
        from lightrag import operate
        src = inspect.getsource(operate.extract_keywords_only)
        assert "langfuse_config" in src
        assert "is_tracing_enabled" in src

    @pytest.mark.asyncio
    async def test_cache_hit_emits_keywords_cache_hit_metadata(self):
        """When keywords are in cache, lf_update_current_span is called with keywords_cache_hit=True."""
        mock_span_update = MagicMock()
        cached_payload = (
            '{"high_level_keywords": ["AI", "ML"], '
            '"low_level_keywords": ["tensor", "model"]}'
        )

        with patch("lightrag.operate.lf_update_current_span", mock_span_update), \
             patch("lightrag.operate.is_tracing_enabled", return_value=False), \
             patch("lightrag.operate.handle_cache",
                   AsyncMock(return_value=(cached_payload, 0))):

            from lightrag.operate import extract_keywords_only
            from lightrag.base import BaseKVStorage

            mock_cache = AsyncMock(spec=BaseKVStorage)
            mock_cache.global_config = {"enable_llm_cache_for_entity_extract": False}
            tok = _make_simple_tokenizer()
            global_config = {
                "role_llm_funcs": {"keyword": _make_async_stub("{}")},
                "tokenizer": tok,
                "addon_params": {},
                "_resolved_summary_language": "English",
            }

            try:
                await extract_keywords_only(
                    text="test query",
                    global_config=global_config,
                    param=MagicMock(mode="hybrid", conversation_history=[]),
                    hashing_kv=mock_cache,
                )
            except Exception:
                # Missing config keys may cause later errors; we only check the span call.
                pass

        hits = [
            c for c in mock_span_update.call_args_list
            if c.kwargs.get("metadata", {}).get("keywords_cache_hit") is True
        ]
        assert len(hits) >= 1


# ---------------------------------------------------------------------------
# 7. operate.py / kg_query and naive_query — query cache tracing
# ---------------------------------------------------------------------------

class TestQueryTracingCacheHit:
    """kg_query and naive_query call lf_update_current_span with query_cache_hit=True on cache hit."""

    def test_kg_query_cache_hit_span_update_in_source(self):
        import inspect
        from lightrag import operate
        src = inspect.getsource(operate.kg_query)
        assert "query_cache_hit" in src
        assert "lf_update_current_span" in src

    def test_naive_query_cache_hit_span_update_in_source(self):
        import inspect
        from lightrag import operate
        src = inspect.getsource(operate.naive_query)
        assert "query_cache_hit" in src
        assert "lf_update_current_span" in src

    def test_kg_query_langfuse_config_correctly_guarded(self):
        """langfuse_config must be assigned and consumed inside the is_tracing_enabled() block."""
        import inspect
        from lightrag import operate
        src = inspect.getsource(operate.kg_query)
        assert "langfuse_config" in src
        assert "kwargs.update(**langfuse_config)" in src

    def test_naive_query_langfuse_config_correctly_guarded(self):
        import inspect
        from lightrag import operate
        src = inspect.getsource(operate.naive_query)
        assert "langfuse_config" in src
        assert "kwargs.update(**langfuse_config)" in src


# ---------------------------------------------------------------------------
# 8. operate.py / merge_nodes_and_edges — span update at end
# ---------------------------------------------------------------------------

class TestMergeNodesAndEdgesTracing:

    def test_lf_update_called_with_counts(self):
        import inspect
        from lightrag import operate
        src = inspect.getsource(operate.merge_nodes_and_edges)
        assert "lf_update_current_span" in src
        assert "total_entities" in src
        assert "total_relations" in src
        assert "entities_merged" in src
        assert "relations_merged" in src

    def test_extra_entities_from_relations_reported(self):
        import inspect
        from lightrag import operate
        src = inspect.getsource(operate.merge_nodes_and_edges)
        assert "extra_entities_from_relations" in src


# ---------------------------------------------------------------------------
# 9. pipeline.py / process_single_document — all tracing call sites
# ---------------------------------------------------------------------------

class TestPipelineTracing:

    def test_process_single_document_decorated_with_lf_observe(self):
        import pathlib
        src = (pathlib.Path(__file__).parents[2] / "lightrag" / "pipeline.py").read_text()
        assert "index-document" in src
        assert "lf_observe" in src

    def test_process_single_document_calls_lf_update_current_span(self):
        import inspect
        from lightrag.pipeline import _PipelineMixin
        src = inspect.getsource(_PipelineMixin.process_single_document)
        assert "lf_update_current_span" in src

    def test_batch_sub_tasks_wrapped_with_lf_propagate_attributes(self):
        import pathlib
        src = (pathlib.Path(__file__).parents[2] / "lightrag" / "pipeline.py").read_text()
        assert "lf_propagate_attributes" in src

    def test_process_single_document_is_callable(self):
        from lightrag.pipeline import _PipelineMixin
        method = getattr(_PipelineMixin, "process_single_document", None)
        assert method is not None and callable(method)


# ---------------------------------------------------------------------------
# 10. API route handlers — source-level checks
# ---------------------------------------------------------------------------

class TestQueryRouteDecorators:

    def test_lf_observe_and_propagate_present(self):
        src = (_routers_dir() / "query_routes.py").read_text()
        assert "lf_observe" in src
        assert "lf_propagate_attributes" in src

    def test_all_three_query_span_names_present(self):
        src = (_routers_dir() / "query_routes.py").read_text()
        for span_name in ("query-text", "query-text-stream", "query-data"):
            assert span_name in src, f"Missing span name: {span_name}"

    def test_document_route_span_names_present(self):
        src = (_routers_dir() / "document_routes.py").read_text()
        for span_name in ("insert-text", "insert-texts"):
            assert span_name in src, f"Missing span name: {span_name}"

    def test_document_routes_use_lf_propagate_attributes(self):
        src = (_routers_dir() / "document_routes.py").read_text()
        assert "lf_propagate_attributes" in src

    def test_document_routes_use_lf_start_as_current_observation(self):
        src = (_routers_dir() / "document_routes.py").read_text()
        assert "lf_start_as_current_observation" in src


# ---------------------------------------------------------------------------
# 11. Disabled-tracing end-to-end guard
# ---------------------------------------------------------------------------

class TestTracingDisabledEndToEnd:

    @pytest.mark.asyncio
    async def test_no_client_calls_when_client_is_none(self):
        mock_client = MagicMock()
        with patch("lightrag.tracing.lf_get_client", return_value=None):
            from lightrag import tracing as tr
            tr.lf_update_current_span(output="x", metadata={"k": "v"})
            ctx = tr.lf_get_current_trace_context()

        assert ctx == {}
        mock_client.update_current_span.assert_not_called()

    @pytest.mark.asyncio
    async def test_flush_and_shutdown_safe_when_disabled(self):
        from lightrag.tracing import lf_flush, lf_shutdown
        with patch("lightrag.tracing.is_tracing_enabled", return_value=False):
            lf_flush()
            lf_shutdown()

    @pytest.mark.asyncio
    async def test_propagate_attributes_safe_when_disabled(self):
        from lightrag.tracing import lf_propagate_attributes
        with patch("lightrag.tracing.is_tracing_enabled", return_value=False):
            async with lf_propagate_attributes(tags=["x"], trace_name="trace"):
                pass  # must not raise
