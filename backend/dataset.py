from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .contracts import Category


class ManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1, max_length=120)
    source_id: str = Field(min_length=1, max_length=120)
    source_sha256: str
    split: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    categories: list[Category] = Field(default_factory=list, max_length=3)
    provenance: str = Field(min_length=1, max_length=500)
    license: str = Field(min_length=1, max_length=200)
    notes: str = Field(default="", max_length=500)

    @field_validator("source_sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("source_sha256 must be 64 lowercase hexadecimal characters")
        return value

    @field_validator("split")
    @classmethod
    def valid_split(cls, value: str) -> str:
        if value not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        return value

    @field_validator("end_ms")
    @classmethod
    def ordered_interval(cls, value: int, info: Any) -> int:
        if value <= info.data.get("start_ms", -1):
            raise ValueError("end_ms must be after start_ms")
        return value


def validate_manifest(entries: list[ManifestEntry]) -> dict[str, Any]:
    if not entries:
        raise ValueError("manifest must contain at least one sample")
    sample_counts = Counter(entry.sample_id for entry in entries)
    duplicates = sorted(sample_id for sample_id, count in sample_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate sample_id values: {', '.join(duplicates)}")

    source_splits: dict[str, set[str]] = defaultdict(set)
    source_hashes: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        source_splits[entry.source_id].add(entry.split)
        source_hashes[entry.source_id].add(entry.source_sha256)
    leaked = sorted(source for source, splits in source_splits.items() if len(splits) > 1)
    if leaked:
        raise ValueError(f"source leakage across splits: {', '.join(leaked)}")
    inconsistent = sorted(source for source, hashes in source_hashes.items() if len(hashes) > 1)
    if inconsistent:
        raise ValueError(f"inconsistent source hash: {', '.join(inconsistent)}")

    split_counts = Counter(entry.split for entry in entries)
    category_counts = Counter(str(category) for entry in entries for category in entry.categories)
    negative_count = sum(not entry.categories for entry in entries)
    warnings = []
    if negative_count == 0:
        warnings.append("manifest has no explicitly negative samples")
    if split_counts["test"] == 0:
        warnings.append("manifest has no frozen test samples")
    return {
        "samples": len(entries),
        "sources": len(source_splits),
        "splits": dict(sorted(split_counts.items())),
        "categories": dict(sorted(category_counts.items())),
        "negative_samples": negative_count,
        "warnings": warnings,
    }
