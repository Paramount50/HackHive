from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Category(StrEnum):
    PERSON_DOWN = "person_down"
    CROWD_GATHERING = "crowd_gathering"
    SUSPECTED_TAILGATING = "suspected_tailgating"


class DecisionState(StrEnum):
    CANDIDATE = "candidate"
    ACCEPTED = "accepted"
    NEEDS_REVIEW = "needs_review"
    DISMISSED = "dismissed"


class ReviewStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class MediaStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    MISSING = "missing"
    EXPIRED = "expired"
    FAILED = "failed"


class FixtureAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Category | str
    summary: str = Field(min_length=1, max_length=500)
    observations: list[dict[str, Any]] = Field(default_factory=list, max_length=8)
    limitations: list[str] = Field(default_factory=list, max_length=8)
    frame_indices: list[int] = Field(default_factory=list, max_length=16)
    model_id: str = "fixture://deterministic"
    model_revision: str = "fixture-v1"
    prompt_version: str = "fixture-v1"

    @field_validator("outcome")
    @classmethod
    def validate_outcome(cls, value: Category | str) -> Category | str:
        if value not in {*Category, "none", "unknown"}:
            raise ValueError("outcome must be a supported category, none, or unknown")
        return value


class QueryIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: str = "list"
    categories: list[Category] = Field(default_factory=list, max_length=3)
    camera_ids: list[str] = Field(default_factory=list, max_length=20)
    decision_states: list[DecisionState] = Field(
        default_factory=lambda: [DecisionState.ACCEPTED, DecisionState.NEEDS_REVIEW]
    )
    start_utc_ms: int | None = None
    end_utc_ms: int | None = None
    limit: int = Field(default=20, ge=1, le=100)
    group_by: str | None = None

    @field_validator("operation")
    @classmethod
    def supported_operation(cls, value: str) -> str:
        if value not in {"list", "count", "group"}:
            raise ValueError("operation must be list, count, or group")
        return value

    @field_validator("group_by")
    @classmethod
    def supported_group(cls, value: str | None) -> str | None:
        if value is not None and value not in {"camera", "category"}:
            raise ValueError("group_by must be camera or category")
        return value

    @field_validator("end_utc_ms")
    @classmethod
    def ordered_range(cls, value: int | None, info: Any) -> int | None:
        start = info.data.get("start_utc_ms")
        if value is not None and start is not None and value <= start:
            raise ValueError("end_utc_ms must be after start_utc_ms")
        return value


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str
    actor_label: str = Field(min_length=1, max_length=100)
    note: str = Field(default="", max_length=1000)
    request_key: str = Field(min_length=1, max_length=200)

    @field_validator("action")
    @classmethod
    def supported_action(cls, value: str) -> str:
        if value not in {"confirm", "reject"}:
            raise ValueError("action must be confirm or reject")
        return value


def utc_now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)
