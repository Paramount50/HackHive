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


def parse_observation(raw: str, frame_count: int) -> VisualObservation:
    """Accept only a whole JSON object; never repair failures into normal scenes."""
    if not 1 <= frame_count <= 4:
        raise ValueError("expected one to four frames")
    if len(raw) > 8_000:
        raise ValueError("model response is too long")
    value = VisualObservation.model_validate(json.loads(raw))
    for observation in value.observations:
        if any(index < 0 or index >= frame_count for index in observation.frame_indices):
            raise ValueError("evidence references a frame outside this request")
        if len(set(observation.frame_indices)) != len(observation.frame_indices):
            raise ValueError("duplicate frame reference")
    return value


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
            "type": "array", "maxItems": 2 if allowed else 0, "items": observation}
        if allowed:
            schema["required"] = ["outcome", "summary", "observations", "limitations"]
        else:
            # LMFE 0.11.3 can admit a first item despite maxItems=0. Omit this
            # field for abstentions/normal; validation supplies an empty list.
            del schema["properties"]["observations"]
        alternatives.append(schema)
    return {"anyOf": alternatives}
