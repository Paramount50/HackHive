"""The P0 visual model boundary, independent of optional GPU dependencies."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Outcome = Literal[
    "person_down", "crowd_gathering", "suspected_tailgating",
    "aggression_fight", "fire_smoke", "intrusion_unauthorized_access",
    "none", "unknown",
]
PREDICATES = {
    "person_down": {"visible_person", "prone_posture", "persistence", "usable_context"},
    "crowd_gathering": {"people_in_zone", "threshold_reached", "duration_reached", "scene_visibility"},
    "suspected_tailgating": {
        "distinct_people", "same_direction_passages", "temporal_proximity",
        "gate_visibility", "access_mismatch",
    },
    "aggression_fight": {
        "multiple_people_visible", "physical_contact", "aggressive_motion",
        "temporal_persistence",
    },
    "fire_smoke": {
        "flame_like_region", "smoke_like_region", "temporal_persistence",
        "scene_visibility",
    },
    "intrusion_unauthorized_access": {
        "person_in_restricted_zone", "restricted_boundary_crossing",
        "scene_visibility", "temporal_persistence",
    },
    "none": set(), "unknown": set(),
}

@dataclass(frozen=True)
class VerificationContract:
    """Application-owned rules for interpreting category-conditioned evidence."""

    required: frozenset[str]
    alternatives: tuple[frozenset[str], ...] = ()
    temporal: frozenset[str] = frozenset()
    tracker_corroboration: frozenset[str] = frozenset()
    ambiguity_context: frozenset[str] = frozenset()


VERIFICATION_CONTRACTS = {
    "person_down": VerificationContract(
        required=frozenset({"visible_person", "prone_posture"}),
        temporal=frozenset({"persistence"}),
        ambiguity_context=frozenset({"usable_context"}),
    ),
    "crowd_gathering": VerificationContract(
        required=frozenset({"people_in_zone", "scene_visibility"}),
        # Exact count and duration remain tracker facts. These visual answers
        # can prevent an automatic supported verdict, but cannot negate them.
        tracker_corroboration=frozenset({"threshold_reached", "duration_reached"}),
    ),
    "suspected_tailgating": VerificationContract(
        required=frozenset({
            "distinct_people", "same_direction_passages", "temporal_proximity",
            "gate_visibility",
        }),
    ),
    "aggression_fight": VerificationContract(
        required=frozenset({
            "multiple_people_visible", "physical_contact", "aggressive_motion",
        }),
        temporal=frozenset({"temporal_persistence"}),
    ),
    "fire_smoke": VerificationContract(
        required=frozenset({"scene_visibility"}),
        alternatives=(frozenset({"flame_like_region", "smoke_like_region"}),),
        temporal=frozenset({"temporal_persistence"}),
    ),
    "intrusion_unauthorized_access": VerificationContract(
        required=frozenset({
            "person_in_restricted_zone", "restricted_boundary_crossing", "scene_visibility",
        }),
        temporal=frozenset({"temporal_persistence"}),
    ),
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


def parse_candidate_observation(
    raw: str, frame_count: int, expected_outcome: str
) -> VisualObservation:
    """Parse the fixed per-predicate shape used by candidate verification."""
    if expected_outcome not in VERIFICATION_CONTRACTS:
        raise ValueError("expected outcome is not supported by visual verification")
    if len(raw) > 8_000:
        raise ValueError("model response is too long")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ObservationValidationError("json_parse", str(exc)) from exc
    predicates = sorted(PREDICATES[expected_outcome] - {"access_mismatch"})
    if not isinstance(parsed, dict):
        raise ObservationValidationError("schema_or_semantic_validation", "candidate output must be an object")
    if parsed.get("outcome") != expected_outcome:
        raise ObservationValidationError("schema_or_semantic_validation", "candidate outcome mismatch")
    evidence = parsed.get("evidence")
    evidence_frames = parsed.get("evidence_frames")
    if not isinstance(evidence, dict) or set(evidence) != set(predicates):
        raise ObservationValidationError(
            "schema_or_semantic_validation", "candidate evidence must answer every predicate"
        )
    if not isinstance(evidence_frames, dict) or set(evidence_frames) != set(predicates):
        raise ObservationValidationError(
            "schema_or_semantic_validation", "candidate evidence_frames must cite every predicate"
        )
    observations = []
    for predicate in predicates:
        supported = evidence[predicate]
        if supported is not None and type(supported) is not bool:
            raise ObservationValidationError(
                "schema_or_semantic_validation", "candidate evidence must be boolean or null"
            )
        indices = evidence_frames[predicate]
        if (
            not isinstance(indices, list)
            or not 1 <= len(indices) <= 4
            or any(type(index) is not int for index in indices)
            or len(set(indices)) != len(indices)
        ):
            raise ObservationValidationError(
                "frame_validation", "candidate evidence_frames must contain distinct frame indices"
            )
        observations.append({
            "predicate": predicate,
            "supported": supported,
            "frame_indices": indices,
        })
    normalized = {
        "outcome": parsed.get("outcome"),
        "summary": parsed.get("summary"),
        "observations": observations,
        "limitations": parsed.get("limitations"),
    }
    try:
        value = VisualObservation.model_validate(normalized)
    except ValueError as exc:
        raise ObservationValidationError("schema_or_semantic_validation", str(exc)) from exc
    for observation in value.observations:
        if any(index < 0 or index >= frame_count for index in observation.frame_indices):
            raise ObservationValidationError("frame_validation", "evidence references a frame outside this request")
    return value


def verification_verdict(observation: VisualObservation, expected_outcome: str) -> str:
    """Interpret a visual observation without converting it into incident truth."""
    if expected_outcome not in VERIFICATION_CONTRACTS:
        raise ValueError("expected outcome is not supported by visual verification")
    if observation.outcome == "unknown":
        return "inconclusive"
    if observation.outcome == "none":
        return "contradicted"
    if observation.outcome != expected_outcome:
        return "category_conflict"
    evidence = {item.predicate: item.supported for item in observation.observations}
    contract = VERIFICATION_CONTRACTS[expected_outcome]
    required = contract.required | contract.temporal
    if any(predicate in evidence and evidence[predicate] is False for predicate in required):
        return "contradicted"
    if any(predicate in evidence and evidence[predicate] is None for predicate in required):
        return "inconclusive"
    if any(predicate not in evidence for predicate in required):
        return "insufficient_evidence"
    for alternatives in contract.alternatives:
        if any(evidence.get(predicate) is True for predicate in alternatives):
            continue
        if alternatives.issubset(evidence) and all(
            evidence[predicate] is False for predicate in alternatives
        ):
            return "contradicted"
        if any(evidence.get(predicate) is None for predicate in alternatives if predicate in evidence):
            return "inconclusive"
        return "insufficient_evidence"
    # Visual estimates cannot replace exact tracker counts or durations. A
    # disagreement blocks support and remains visible for human review.
    if any(predicate not in evidence for predicate in contract.tracker_corroboration):
        return "insufficient_evidence"
    if any(evidence[predicate] is not True for predicate in contract.tracker_corroboration):
        return "inconclusive"
    # Context fields explain whether the primary evidence can be interpreted.
    # Their absence is non-decision evidence for legacy outputs; an explicit
    # false or unknown answer prevents a supported verdict.
    if any(
        evidence[predicate] is not True
        for predicate in contract.ambiguity_context
        if predicate in evidence
    ):
        return "inconclusive"
    return "supported"


def generation_schema(candidate_outcome: str | None = None) -> dict:
    """Compact generation schema; semantic validation still runs afterward."""
    alternatives = []
    outcomes = [candidate_outcome] if candidate_outcome is not None else list(PREDICATES)
    if candidate_outcome is not None and candidate_outcome not in PREDICATES:
        raise ValueError("candidate outcome is not supported by the visual contract")
    for outcome in outcomes:
        predicates = PREDICATES[outcome]
        schema = VisualObservation.model_json_schema()
        schema["properties"]["outcome"] = {"type": "string", "enum": [outcome]}
        schema["properties"]["summary"]["maxLength"] = 100 if candidate_outcome else 160
        schema["properties"]["limitations"] = {
            "type": "array", "maxItems": 2, "items": {"type": "string", "maxLength": 120}}
        if candidate_outcome is not None:
            evidence_properties = {
                predicate: {"type": ["boolean", "null"]} for predicate in sorted(
                    predicates - {"access_mismatch"}
                )
            }
            frame_properties = {
                predicate: {
                    "type": "array", "minItems": 1, "maxItems": 4,
                    "items": {"type": "integer", "enum": [0, 1, 2, 3]},
                } for predicate in sorted(predicates - {"access_mismatch"})
            }
            schema["properties"]["evidence"] = {
                "type": "object", "properties": evidence_properties,
                "required": sorted(evidence_properties), "additionalProperties": False,
            }
            schema["properties"]["evidence_frames"] = {
                "type": "object", "properties": frame_properties,
                "required": sorted(frame_properties), "additionalProperties": False,
            }
            schema["required"] = [
                "outcome", "summary", "evidence", "evidence_frames", "limitations"
            ]
            schema["properties"].pop("observations", None)
            schema.pop("$defs", None)
            alternatives.append(schema)
            continue
        # The visual model has no access logs. Keep this predicate for the policy layer.
        allowed = sorted(predicates - {"access_mismatch"})
        observation = schema.pop("$defs")["Observation"]
        observation["properties"]["predicate"] = {"type": "string", "enum": allowed or ["unused"]}
        observation["properties"]["frame_indices"]["items"] = {
            "type": "integer", "enum": [0, 1, 2, 3]}
        schema["properties"]["observations"] = {
            "type": "array", "maxItems": 2 if allowed else 0,
            "items": observation}
        if allowed:
            schema["required"] = ["outcome", "summary", "observations", "limitations"]
        else:
            # LMFE 0.11.3 can admit a first item despite maxItems=0. Omit this
            # field for abstentions/normal; validation supplies an empty list.
            del schema["properties"]["observations"]
        alternatives.append(schema)
    return {"anyOf": alternatives}
