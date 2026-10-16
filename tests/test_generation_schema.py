import json

import pytest

from backend.inference.contracts import (
    generation_schema, parse_candidate_observation, parse_observation, verification_verdict,
)
from backend.inference.smolvlm import candidate_prompt


def accepts(text):
    lmfe = pytest.importorskip("lmformatenforcer")
    from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig
    parser = lmfe.JsonSchemaParser(generation_schema(), config=CharacterLevelParserConfig(force_json_field_order=True))
    for character in text:
        if character not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(character)
    return parser.can_end()


def test_generation_schema_allows_unknown_and_visible_evidence():
    assert accepts(json.dumps({"outcome": "unknown", "summary": "Unclear scene.", "limitations": []}))
    assert accepts(json.dumps({"outcome": "person_down", "summary": "A prone person.", "observations": [
        {"predicate": "prone_posture", "supported": True, "frame_indices": [0, 3]}], "limitations": []}))


def test_generation_schema_excludes_visual_authorization_claims():
    assert not accepts(json.dumps({"outcome": "suspected_tailgating", "summary": "Entry.", "observations": [
        {"predicate": "access_mismatch", "supported": True, "frame_indices": [0]}], "limitations": []}))


def test_unknown_cannot_generate_fabricated_predicates():
    assert not accepts(json.dumps({"outcome": "unknown", "summary": "Unclear.", "observations": [
        {"predicate": "unused", "supported": True, "frame_indices": [0]}], "limitations": []}))


def test_category_must_be_chosen_before_its_predicates():
    assert not accepts(json.dumps({"summary": "A person.", "outcome": "person_down",
                                   "observations": [], "limitations": []}))


def test_generation_schema_does_not_claim_unenforced_array_uniqueness():
    for alternative in generation_schema()["anyOf"]:
        observations = alternative["properties"].get("observations")
        if observations is not None:
            assert "uniqueItems" not in observations


def test_candidate_schema_is_fixed_to_the_expected_category():
    alternatives = generation_schema("aggression_fight")["anyOf"]
    assert len(alternatives) == 1
    assert alternatives[0]["properties"]["outcome"]["enum"] == ["aggression_fight"]
    assert "intrusion_unauthorized_access" not in str(alternatives[0])
    assert set(alternatives[0]["properties"]["evidence"]["required"]) == {
        "multiple_people_visible", "physical_contact", "aggressive_motion",
        "temporal_persistence",
    }
    assert alternatives[0]["properties"]["evidence_frames"]["properties"][
        "physical_contact"
    ]["items"]["maximum"] == 3


def test_candidate_parser_requires_every_predicate_and_preserves_null():
    raw = json.dumps({
        "outcome": "aggression_fight",
        "summary": "A physical altercation is visible.",
        "evidence": {
            "multiple_people_visible": True,
            "physical_contact": True,
            "aggressive_motion": None,
            "temporal_persistence": False,
        },
        "evidence_frames": {
            "multiple_people_visible": [0],
            "physical_contact": [1, 2, 3],
            "aggressive_motion": [1],
            "temporal_persistence": [3],
        },
        "limitations": [],
    })
    parsed = parse_candidate_observation(raw, 4, "aggression_fight")
    assert {item.predicate: item.supported for item in parsed.observations} == {
        "multiple_people_visible": True,
        "physical_contact": True,
        "aggressive_motion": None,
        "temporal_persistence": False,
    }


def test_candidate_parser_rejects_out_of_range_frame_without_repair():
    raw = json.dumps({
        "outcome": "aggression_fight",
        "summary": "A physical altercation is visible.",
        "evidence": {
            "multiple_people_visible": True, "physical_contact": True,
            "aggressive_motion": None, "temporal_persistence": True,
        },
        "evidence_frames": {
            "multiple_people_visible": [0], "physical_contact": [4],
            "aggressive_motion": [1], "temporal_persistence": [3],
        },
        "limitations": [],
    })
    with pytest.raises(ValueError, match="outside this request"):
        parse_candidate_observation(raw, 4, "aggression_fight")


def test_candidate_prompt_uses_only_local_frame_indices():
    prompt = candidate_prompt("aggression_fight")
    assert "LOCAL FRAME INDEX" in prompt
    assert "source PTS" not in prompt


def test_truncated_candidate_json_remains_invalid():
    with pytest.raises(ValueError, match="Expecting value"):
        parse_candidate_observation(
            '{"outcome":"aggression_fight","summary":"A fight is visible","evidence":',
            4,
            "aggression_fight",
        )


def test_complete_aggression_candidate_is_supported():
    raw = json.dumps({
        "outcome": "aggression_fight",
        "summary": "People are physically fighting.",
        "evidence": {
            "aggressive_motion": True,
            "multiple_people_visible": True,
            "physical_contact": True,
            "temporal_persistence": True,
        },
        "evidence_frames": {
            "aggressive_motion": [1, 2, 3],
            "multiple_people_visible": [1, 2, 3],
            "physical_contact": [1, 2, 3],
            "temporal_persistence": [1, 2, 3],
        },
        "limitations": [],
    })
    observation = parse_candidate_observation(raw, 4, "aggression_fight")
    assert verification_verdict(observation, "aggression_fight") == "supported"


def test_duplicate_predicates_are_rejected_after_generation():
    duplicate = json.dumps({"outcome": "person_down", "summary": "A person.",
                            "observations": [
                                {"predicate": "visible_person", "supported": True, "frame_indices": [0]},
                                {"predicate": "visible_person", "supported": True, "frame_indices": [0]},
                            ], "limitations": []})
    # LMFE 0.11.3 does not enforce JSON Schema uniqueItems. The downstream
    # Pydantic contract must reject the raw generated output instead.
    assert accepts(duplicate)
    with pytest.raises(ValueError, match="duplicate evidence predicate"):
        parse_observation(duplicate, 4)
