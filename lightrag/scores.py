"""
This module defines functions to compute and submit scores to Langfuse from traces triggered by LightRAG API calls.
These scores are used for monitoring, debugging retrieval behavior, and evaluating graph-quality signals.
"""

from lightrag.tracing import lf_score_current_span
from collections import defaultdict


def _submit_retrieval_scores(response: dict) -> None:
    """
    Compute and submit retrieval-quality scores to Langfuse.

    Metrics are grouped into:
    - Retrieval Volume
    - Truncation Pressure
    - Graph Connectivity
    - Graph Topology
    - Semantic Strength
    - Retrieval Concentration
    """

    # =========================================================
    # 🔷 Extract response payload
    # =========================================================
    data = response.get("data") or {}
    entities = data.get("entities") or []
    relationships = data.get("relationships") or []
    chunks = data.get("chunks") or []
    chunks_references = data.get("references") or {}

    meta = response.get("metadata") or {}
    info = meta.get("processing_info") or {}

    success = response.get("status") == "success"
    has_results = bool(entities or relationships or chunks)

    # =========================================================
    # 🔷 Truncation / pipeline stats
    # =========================================================
    total_ent = info.get("total_entities_found", 0)
    kept_ent = info.get("entities_after_truncation", 0)

    total_rel = info.get("total_relations_found", 0)
    kept_rel = info.get("relations_after_truncation", 0)

    total_chk = info.get("merged_chunks_count", 0)
    kept_chk = info.get("final_chunks_count", 0)

    ent_retention = round(kept_ent / total_ent, 3) if total_ent else -1
    rel_retention = round(kept_rel / total_rel, 3) if total_rel else -1
    chk_retention = round(kept_chk / total_chk, 3) if total_chk else -1

    # =========================================================
    # 🔷 Build retrieval-induced graph
    # =========================================================

    # Nodes that actually participate in relations (true graph structure)
    graph_nodes = {
        n
        for r in relationships
        for n in (r.get("src_id"), r.get("tgt_id"))
        if n
    }

    # Nodes returned by LightRAG entity retrieval (raw output)
    retrieved_nodes = { 
        e.get("entity_name") for e in entities
    }

    # =========================================================
    # 🔷 Graph Connectivity Metrics (core diagnostic signals)
    # =========================================================

    # Entities that are not supported by any relationship
    isolated_nodes = retrieved_nodes - graph_nodes

    isolated_ratio = (
        round(len(isolated_nodes) / len(retrieved_nodes), 3)
        if retrieved_nodes else 0.0
    )
    # → Measures how many retrieved entities are "floating" without graph support: 0 means all entities are grounded in relations, 1 means none are

    graph_coverage = (
        round(len(retrieved_nodes & graph_nodes) / len(retrieved_nodes), 3)
        if retrieved_nodes else 0.0
    )
    # → Measures how much of retrieved entity set is grounded in relations: 0 means no entities are supported by relations, 1 means all are

    # =========================================================
    # 🔷 Graph Topology Metrics
    # =========================================================

    # Undirected-style average degree
    avg_degree = (
        round((2 * kept_rel) / len(graph_nodes), 3)
        if graph_nodes else 0.0
    )
    # → Overall connectivity strength of induced subgraph

    density = (
        round(kept_rel / (len(graph_nodes) * (len(graph_nodes) - 1)), 3)
        if len(graph_nodes) > 1 else 0.0
    )
    # → How close graph is to being fully connected

    # Directed out-degree distribution
    out_degree = defaultdict(int)

    for r in relationships:
        src = r.get("src_id")
        if src:
            out_degree[src] += 1

    avg_out_degree = (
        round(sum(out_degree.values()) / len(graph_nodes), 3)
        if graph_nodes else 0.0
    )
    # → Average outgoing edges per node in retrieval subgraph

    # =========================================================
    # 🔷 Semantic Strength
    # =========================================================

    weights = [
        r.get("weight", 0)
        for r in relationships
        if isinstance(r.get("weight"), (int, float))
    ]

    avg_relationship_weight = (
        round(sum(weights) / len(weights), 3)
        if weights else 0.0
    )
    # → Confidence/strength of retrieved relationships

    # =========================================================
    # 🔷 Retrieval Concentration (source diversity)
    # =========================================================

    source_files = set()

    for e in entities:
        for p in (e.get("file_path") or "").split("<SEP>"):
            if p.strip():
                source_files.add(p.strip())

    for r in relationships:
        for p in (r.get("file_path") or "").split("<SEP>"):
            if p.strip():
                source_files.add(p.strip())

    source_document_count = len(source_files)

    avg_entities_per_document = (
        round(kept_ent / source_document_count, 3)
        if source_document_count else 0.0
    )
    # → Measures retrieval concentration across documents

    context_document_count = len(chunks_references)

    # =========================================================
    # 🔷 Langfuse scoring
    # =========================================================

    lf_score_current_span([
        # -----------------------------------------------------
        # General status
        # -----------------------------------------------------
        {
            "name": "retrieval_success",
            "value": success,
            "data_type": "BOOLEAN",
            "comment": "Whether retrieval pipeline completed successfully",
        },
        {
            "name": "has_any_results",
            "value": has_results,
            "data_type": "BOOLEAN",
            "comment": "Whether any entities, relations, or chunks were returned",
        },

        # -----------------------------------------------------
        # Volume
        # -----------------------------------------------------
        {
            "name": "entity_count",
            "value": float(kept_ent),
            "data_type": "NUMERIC",
            "comment": "Number of entities after truncation (query-level retrieval size)",
        },
        {
            "name": "relationship_count",
            "value": float(kept_rel),
            "data_type": "NUMERIC",
            "comment": "Number of relationships after truncation",
        },
        {
            "name": "chunk_count",
            "value": float(kept_chk),
            "data_type": "NUMERIC",
            "comment": "Number of text chunks used in final context",
        },

        # -----------------------------------------------------
        # Truncation pressure
        # -----------------------------------------------------
        {
            "name": "entity_retention_ratio",
            "value": ent_retention,
            "data_type": "NUMERIC",
            "comment": "Fraction of entities preserved after truncation",
        },
        {
            "name": "relationship_retention_ratio",
            "value": rel_retention,
            "data_type": "NUMERIC",
            "comment": "Fraction of relationships preserved after truncation",
        },
        {
            "name": "chunk_retention_ratio",
            "value": chk_retention,
            "data_type": "NUMERIC",
            "comment": "Fraction of chunks preserved after truncation",
        },

        # -----------------------------------------------------
        # Graph connectivity (most important diagnostics)
        # -----------------------------------------------------
        {
            "name": "isolated_entity_ratio",
            "value": isolated_ratio,
            "data_type": "NUMERIC",
            "comment": "Fraction of retrieved entities not connected to any relationship",
        },
        {
            "name": "graph_coverage",
            "value": graph_coverage,
            "data_type": "NUMERIC",
            "comment": "Fraction of retrieved entities supported by at least one relation",
        },

        # -----------------------------------------------------
        # Graph topology
        # -----------------------------------------------------
        {
            "name": "avg_degree",
            "value": avg_degree,
            "data_type": "NUMERIC",
            "comment": "Average undirected degree of retrieval-induced graph",
        },
        {
            "name": "avg_out_degree",
            "value": avg_out_degree,
            "data_type": "NUMERIC",
            "comment": "Average outgoing edges per node in retrieved graph",
        },
        {
            "name": "density",
            "value": density,
            "data_type": "NUMERIC",
            "comment": "Graph density of retrieved subgraph (0 to 1, higher means more connected up to fully connected)",
        },

        # -----------------------------------------------------
        # Semantic strength
        # -----------------------------------------------------
        {
            "name": "avg_relationship_weight",
            "value": avg_relationship_weight,
            "data_type": "NUMERIC",
            "comment": "Average confidence/weight of retrieved relationships",
        },

        # -----------------------------------------------------
        # Retrieval concentration
        # -----------------------------------------------------
        {
            "name": "source_document_count",
            "value": float(source_document_count),
            "data_type": "NUMERIC",
            "comment": "Number of unique source documents contributing to retrieved entities/relationships",
        },
        {
            "name": "avg_entities_per_document",
            "value": avg_entities_per_document,
            "data_type": "NUMERIC",
            "comment": "Average number of entities per source document",
        },
        {
            "name": "context_document_count",
            "value": float(context_document_count),
            "data_type": "NUMERIC",
            "comment": "Number of unique documents referenced by final chunks",
        },
    ])