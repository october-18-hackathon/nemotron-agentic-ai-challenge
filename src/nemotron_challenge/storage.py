"""A small replaceable store for artifact provenance."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Protocol

from .contracts import ArtifactRecord


class ArtifactStore(Protocol):
    def put(self, record: ArtifactRecord) -> None: ...
    def get(self, record_id: str) -> ArtifactRecord | None: ...


class LanceArtifactStore:
    """Persist artifact records. Teams define their retrieval tables and indexes."""

    def __init__(self, path: str | Path):
        import lancedb
        import pyarrow as pa

        self.db = lancedb.connect(str(path))
        self.table = self.db.create_table(
            "artifact_provenance",
            schema=pa.schema([("record_id", pa.string()), ("record_json", pa.string())]),
            exist_ok=True,
        )

    def put(self, record: ArtifactRecord) -> None:
        self.table.merge_insert("record_id").when_matched_update_all().when_not_matched_insert_all().execute([
            {"record_id": record.record_id, "record_json": record.model_dump_json()}
        ])

    def get(self, record_id: str) -> ArtifactRecord | None:
        if not record_id or len(record_id) > 200:
            raise ValueError("record_id must contain 1 to 200 characters")
        escaped = record_id.replace("'", "''")
        rows = self.table.search().where(f"record_id = '{escaped}'").select(["record_json"]).limit(2).to_list(
            timeout=timedelta(seconds=10))
        if len(rows) > 1:
            raise ValueError("provenance record ID is not unique")
        return ArtifactRecord.model_validate_json(rows[0]["record_json"]) if rows else None
