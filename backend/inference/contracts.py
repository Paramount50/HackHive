"""The P0 visual model boundary, independent of optional GPU dependencies."""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Outcome = Literal["person_down", "crowd_gathering", "suspected_tailgating", "none", "unknown"]
PREDICATES = {
    "person_down": {"visible_person", "prone_posture", "persistence", "usable_context"},
    "crowd_gathering": {"people_in_zone", "threshold_reached", "duration_reached", "scene_visibility"},
    "suspected_tailgating": {"same_direction_passages", "temporal_proximity", "gate_visibility", "access_mismatch"},
    "none": set(), "unknown": set(),
}
REQUIRED_VISUAL_EVIDENCE = {
    "person_down": {"visible_person", "prone_posture"},
    "crowd_gathering": {"people_in_zone", "scene_visibility"},
    "suspected_tailgating": {"same_direction_passages", "gate_visibility"},
}


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    predicate: str = Field(min_length=1, max_length=64)
    supported: bool | None
    frame_indices: list[int] = Field(min_length=1, max_length=4)


class VisualObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    outcome: Outcome
    summary: str = Field(min_length=1, max_length=500)
    observations: list[Observation] = Field(default_factory=list, max_length=4)
    limitations: list[str] = Field(max_length=8)

    @model_validator(mode="after")
    def check_predicates(self):
        names = [item.predicate for item in self.observations]
        if len(names) != len(set(names)):
            raise ValueError("duplicate evidence predicate")
        if any(name not in PREDICATES[self.outcome] for name in names):
            raise ValueError("predicate is not allowed for this outcome")
        if any(len(value) > 300 for value in self.limitations):
            raise ValueError("limitation is too long")
        # Access-log reconciliation belongs to B's deterministic policy layer.
        if any(item.predicate == "access_mismatch" and item.supported is not None for item in self.observations):
            raise ValueError("visual output cannot establish access authorization")
        return self


class ObservationValidationError(ValueError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


def parse_observation(raw: str, frame_count: int) -> VisualObservation:
    """Accept only a whole JSON object; never repair failures into normal scenes."""
    if not 1 <= frame_count <= 4:
        raise ValueError("expected one to four frames")
    if len(raw) > 8_000:
        raise ValueError("model response is too long")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ObservationValidationError("json_parse", str(exc)) from exc
    try:
        value = VisualObservation.model_validate(parsed)
    except ValueError as exc:
        raise ObservationValidationError("schema_or_semantic_validation", str(exc)) from exc
    for observation in value.observations:
        if any(index < 0 or index >= frame_count for index in observation.frame_indices):
            raise ObservationValidationError("frame_validation", "evidence references a frame outside this request")
        if len(set(observation.frame_indices)) != len(observation.frame_indices):
            raise ObservationValidationError("frame_validation", "duplicate frame reference")
    return value


def verification_verdict(observation: VisualObservation, expected_outcome: str) -> str:
    """Interpret a visual observation without converting it into incident truth."""
    if expected_outcome not in REQUIRED_VISUAL_EVIDENCE:
        raise ValueError("expected outcome is not supported by visual verification")
    if observation.outcome == "unknown":
        return "inconclusive"
    if observation.outcome == "none":
        return "contradicted"
    if observation.outcome != expected_outcome:
        return "category_conflict"
    evidence = {item.predicate: item.supported for item in observation.observations}
    required = REQUIRED_VISUAL_EVIDENCE[expected_outcome]
    if any(predicate in evidence and evidence[predicate] is False for predicate in required):
        return "contradicted"
    if any(predicate in evidence and evidence[predicate] is None for predicate in required):
        return "inconclusive"
    if any(predicate not in evidence for predicate in required):
        return "insufficient_evidence"
    return "supported"


def generation_schema() -> dict:
    """Compact generation schema; semantic validation still runs afterward."""
    alternatives = []
    for outcome, predicates in PREDICATES.items():
        schema = VisualObservation.model_json_schema()
        schema["properties"]["outcome"] = {"type": "string", "enum": [outcome]}
        schema["properties"]["summary"]["maxLength"] = 160
        schema["properties"]["limitations"] = {
            "type": "array", "maxItems": 2, "items": {"type": "string", "maxLength": 120}}
        # The visual model has no access logs. Keep this predicate for the policy layer.
        allowed = sorted(predicates - {"access_mismatch"})
        observation = schema.pop("$defs")["Observation"]
        observation["properties"]["predicate"] = {"type": "string", "enum": allowed or ["unused"]}
        observation["properties"]["frame_indices"]["items"] = {
            "type": "integer", "minimum": 0, "maximum": 3}
        schema["properties"]["observations"] = {
            "type": "array", "maxItems": 2 if allowed else 0,
            "uniqueItems": True, "items": observation}
        if allowed:
            schema["required"] = ["outcome", "summary", "observations", "limitations"]
        else:
            # LMFE 0.11.3 can admit a first item despite maxItems=0. Omit this
            # field for abstentions/normal; validation supplies an empty list.
            del schema["properties"]["observations"]
        alternatives.append(schema)
    return {"anyOf": alternatives}
