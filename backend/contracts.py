from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Category(StrEnum):
    PERSON_DOWN = "person_down"
    CROWD_GATHERING = "crowd_gathering"
    SUSPECTED_TAILGATING = "suspected_tailgating"
    AGGRESSION_FIGHT = "aggression_fight"
    FIRE_SMOKE = "fire_smoke"
    INTRUSION_UNAUTHORIZED_ACCESS = "intrusion_unauthorized_access"


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
    categories: list[Category] = Field(default_factory=list, max_length=6)
    camera_ids: list[str] = Field(default_factory=list, max_length=20)
    decision_states: list[DecisionState] = Field(
        default_factory=lambda: [DecisionState.ACCEPTED, DecisionState.NEEDS_REVIEW]
    )
    review_statuses: list[ReviewStatus] = Field(default_factory=list, max_length=3)
    has_ready_media: bool | None = None
    source_name: str | None = Field(default=None, min_length=1, max_length=240)
    start_pts_ms: int | None = Field(default=None, ge=0)
    end_pts_ms: int | None = Field(default=None, gt=0)
    start_utc_ms: int | None = None
    end_utc_ms: int | None = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
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

    @field_validator("end_pts_ms")
    @classmethod
    def ordered_source_range(cls, value: int | None, info: Any) -> int | None:
        start = info.data.get("start_pts_ms")
        if value is not None and start is not None and value <= start:
            raise ValueError("end_pts_ms must be after start_pts_ms")
        return value

    @model_validator(mode="after")
    def source_offset_requires_recording(self) -> "QueryIntent":
        has_offset = self.start_pts_ms is not None or self.end_pts_ms is not None
        if has_offset and (self.start_pts_ms is None or self.end_pts_ms is None or not self.source_name):
            raise ValueError("video-offset filters require source_name and both start_pts_ms/end_pts_ms")
        return self


class NaturalQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=500)
    timezone: str = "Asia/Kolkata"
    now_utc_ms: int | None = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)

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


class ReplayStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_name: str = Field(min_length=1, max_length=240)
    sample_fps: float = Field(default=5.0, ge=0.5, le=10.0)
    pace: bool = True
    decode_strategy: Literal["sequential", "grab"] = "sequential"
    max_frames: int | None = Field(default=None, ge=1, le=10_000)
    gate_line_y: float | None = Field(default=None, ge=0, le=1)
    zone: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)

    @field_validator("source_name")
    @classmethod
    def reject_blank_source(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source_name must not be blank")
        return value

    @field_validator("zone")
    @classmethod
    def valid_zone(cls, value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = value
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("zone must be an increasing normalized rectangle")
        return value


def utc_now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)
