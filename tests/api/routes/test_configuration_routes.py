from copy import deepcopy
from types import SimpleNamespace
import importlib
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightrag.chunker import chunking_by_token_size

_original_argv = sys.argv[:]
try:
    sys.argv = [sys.argv[0]]
    _configuration = importlib.import_module(
        "lightrag.api.routers.configuration_routes"
    )
finally:
    sys.argv = _original_argv
build_configuration = _configuration.build_configuration
create_configuration_routes = _configuration.create_configuration_routes
QueryRequest = _configuration.QueryRequest


pytestmark = pytest.mark.offline


def backend():
    return SimpleNamespace(
        addon_params={
            "chunker": {
                "chunk_token_size": 1200,
                "fixed_token": {"chunk_overlap_token_size": 100},
                "recursive_character": {"separators": ["\n\n", ""]},
                "semantic_vector": {"sentence_split_regex": "operator-only"},
            },
            "secret": "never disclose",
        },
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        chunking_func=chunking_by_token_size,
        rerank_model_func=None,
        user_prompt_prefix="private instructions",
    )


def test_defaults_use_request_resolution_without_mutating_backend(monkeypatch):
    rag = backend()
    before = deepcopy(rag.addon_params)
    monkeypatch.setenv("TOP_K", "987")
    monkeypatch.setenv("CHUNK_P_SIZE", "2000")
    result = build_configuration(rag, {"llm_model": "model"})
    actual = QueryRequest(query="configuration defaults").to_query_params(False)
    assert result.query.defaults["top_k"] == actual.top_k
    assert rag.addon_params == before
    assert (
        result.text_ingestion.strategies["fixed_token"].defaults["chunk_token_size"]
        == 1200
    )
    assert (
        result.text_ingestion.strategies["paragraph_semantic"].defaults[
            "chunk_token_size"
        ]
        == 2000
    )
    assert not result.text_ingestion.strategies["custom"].available
    assert (
        result.query.config_schema["properties"]["hl_keywords"]["items"]["maxLength"]
        > 0
    )
    serialized = result.model_dump_json(by_alias=True)
    for secret in (
        "never disclose",
        "private instructions",
        "operator-only",
        "sentence_split_regex",
    ):
        assert secret not in serialized


def test_auth_ignores_whitelist_and_returns_no_store(monkeypatch):
    from lightrag.api import utils_api

    monkeypatch.setattr(utils_api, "auth_configured", False)
    monkeypatch.setattr(utils_api, "whitelist_patterns", [("/", True)])
    app = FastAPI()
    app.include_router(
        create_configuration_routes(
            backend(), lambda: {"llm_model": "model"}, "backend-key"
        )
    )
    with TestClient(app) as client:
        assert client.get("/configuration").status_code == 403
        assert (
            client.get("/configuration", headers={"X-API-Key": "wrong"}).status_code
            == 403
        )
        response = client.get("/configuration", headers={"X-API-Key": "backend-key"})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["schema_version"] == 1
