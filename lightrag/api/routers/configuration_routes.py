from collections.abc import Callable
from copy import deepcopy
from typing import Any, Literal

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, Field

from lightrag.api.routers.document_routes import (
    _CHUNKING_PARAMS_MODEL,
    _STRATEGY_TO_PROCESS_OPTION,
)
from lightrag.api.routers.query_routes import MAX_KEYWORD_CHARS, QueryRequest
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.chunker import chunking_by_token_size
from lightrag.lightrag import LightRAG
from lightrag.parser.routing import chunk_strategy_key, resolve_chunk_options


QUERY_OPTIONS = (
    "mode",
    "top_k",
    "chunk_top_k",
    "max_entity_tokens",
    "max_relation_tokens",
    "max_total_tokens",
    "enable_rerank",
    "response_type",
    "user_prompt",
    "hl_keywords",
    "ll_keywords",
    "disable_user_prompt_prefix",
)


class OptionGroup(BaseModel):
    defaults: dict[str, Any]
    config_schema: dict[str, Any] = Field(alias="schema")
    available: bool = True
    constraints: list[str] = Field(default_factory=list)


class TextIngestionConfiguration(BaseModel):
    default_strategy: Literal["fixed_token"] = "fixed_token"
    routes: list[str] = ["/documents/text", "/documents/texts"]
    strategies: dict[str, OptionGroup]


class ConfigurationDiscovery(BaseModel):
    schema_version: Literal[1] = 1
    server: dict[str, str]
    capabilities: dict[str, bool]
    query: OptionGroup
    text_ingestion: TextIngestionConfiguration
    upload: dict[str, str] = {"configuration_source": "filename_hints_and_parser_rules"}


def build_configuration(
    rag: LightRAG, server: dict[str, str]
) -> ConfigurationDiscovery:
    query_defaults = QueryRequest(query="configuration defaults").to_query_params(False)
    properties = QueryRequest.model_json_schema()["properties"]
    query_properties = {name: deepcopy(properties[name]) for name in QUERY_OPTIONS}
    for name in ("hl_keywords", "ll_keywords"):
        query_properties[name]["items"]["maxLength"] = MAX_KEYWORD_CHARS
    strategies = {}
    addon_params = deepcopy(dict(rag.addon_params))
    for strategy, model in _CHUNKING_PARAMS_MODEL.items():
        process_options = _STRATEGY_TO_PROCESS_OPTION[strategy]
        snapshot = resolve_chunk_options(addon_params, process_options=process_options)
        selected = snapshot[chunk_strategy_key(process_options)]
        fallbacks = {
            "chunk_token_size": snapshot.get("chunk_token_size", rag.chunk_token_size),
            "chunk_overlap_token_size": rag.chunk_overlap_token_size,
            "split_by_character": None,
            "split_by_character_only": False,
            "drop_references": False,
            "breakpoint_threshold_type": "percentile",
            "breakpoint_threshold_amount": None,
            "buffer_size": 1,
        }
        defaults = {
            name: selected.get(name, fallbacks.get(name, field.default))
            for name, field in model.model_fields.items()
        }
        definition = model.model_json_schema()
        definition["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        definition["required"] = list(model.model_fields)
        constraints = (
            ["overlap_less_than_size"] if "chunk_overlap_token_size" in defaults else []
        )
        if strategy == "semantic_vector":
            constraints = ["semantic_threshold_range"]
        strategies[strategy] = OptionGroup(
            defaults=defaults,
            schema=definition,
            available=strategy != "custom"
            or rag.chunking_func is not chunking_by_token_size,
            constraints=constraints,
        )
    return ConfigurationDiscovery(
        server=server,
        capabilities={
            "reranker_configured": rag.rerank_model_func is not None,
            "prompt_prefix_configured": bool(rag.user_prompt_prefix),
        },
        query=OptionGroup(
            defaults={name: getattr(query_defaults, name) for name in QUERY_OPTIONS},
            schema={
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": query_properties,
                "required": list(QUERY_OPTIONS),
                "additionalProperties": False,
            },
        ),
        text_ingestion=TextIngestionConfiguration(strategies=strategies),
    )


def create_configuration_routes(
    rag: LightRAG,
    server_configuration: Callable[[], dict[str, str]],
    api_key: str | None = None,
) -> APIRouter:
    router = APIRouter()
    auth = get_combined_auth_dependency(api_key, respect_whitelist=False)

    @router.get(
        "/configuration",
        response_model=ConfigurationDiscovery,
        dependencies=[Depends(auth)],
    )
    async def configuration(response: Response) -> ConfigurationDiscovery:
        response.headers["Cache-Control"] = "no-store"
        return build_configuration(rag, server_configuration())

    return router
