"""Small observability helpers for the document ingestion pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _doc_display_name(status_doc: Any, doc_id: str) -> str:
    """Human-readable document name for Langfuse traces, matching the WebUI."""
    file_path = getattr(status_doc, "file_path", "") or ""
    if file_path:
        return Path(file_path).name
    return doc_id


def _track_id_prefix(track_id: str | None) -> str:
    """Extract the route/call-type prefix from a pipeline ``track_id``."""
    if not track_id:
        return "unknown"
    return track_id.split("_", 1)[0]