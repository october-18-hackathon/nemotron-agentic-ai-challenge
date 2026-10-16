"""Data contracts keep source evidence with team artifacts."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ScanReference(Contract):
    scan_key: str = Field(min_length=1)
    dataset: str
    dataset_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    split: str
    row_index: int = Field(ge=0)
    source_id: str
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SourceReference(Contract):
    dataset: str
    dataset_revision: str
    source_id: str
    source_url: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license: str


class ArtifactRecord(Contract):
    record_id: str = Field(min_length=1, max_length=200)
    stage: Literal["extract", "match", "compare", "agent", "site", "evaluate"]
    sources: list[SourceReference] = Field(min_length=1)
    artifact_path: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WikiProvenance(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    dataset: str
    dataset_revision: str
    dataset_url: str
    source_key: str
    source_row_idx: int = Field(ge=0)
    source_shard: str | None
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    extraction: str
    revision_url: str


class WikiArticle(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, allow_inf_nan=False)
    page_id: str = Field(pattern=r"^[0-9]{1,20}$")
    revision_id: str = Field(pattern=r"^[0-9]+$")
    title: str
    url: str
    provenance: WikiProvenance
    license: dict[str, str]
    text: str
    offset: int = Field(ge=0)
    returned_chars: int = Field(ge=0)
    total_chars: int = Field(ge=0)
    next_offset: int | None

    @model_validator(mode="after")
    def valid_page(self) -> WikiArticle:
        if self.returned_chars != len(self.text) or self.offset + self.returned_chars > max(self.offset, self.total_chars):
            raise ValueError("article page length differs from its metadata")
        expected_next = self.offset + self.returned_chars if self.offset + self.returned_chars < self.total_chars else None
        if self.next_offset != expected_next:
            raise ValueError("article page continuation differs from its metadata")
        return self


class WikiHit(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, allow_inf_nan=False)
    page_id: str = Field(pattern=r"^[0-9]{1,20}$")
    revision_id: str = Field(pattern=r"^[0-9]+$")
    title: str
    url: str
    provenance: WikiProvenance
    license: dict[str, str]
    rank: int = Field(ge=1)
    bm25_score: float
    excerpt: str
    excerpt_offset: int = Field(ge=0)
    total_chars: int = Field(ge=0)


class WikiSearch(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    query: str
    retrieval: Literal["lexical_bm25"]
    corpus_revision: str
    count: int = Field(ge=0)
    hits: list[WikiHit]
