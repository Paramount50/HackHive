import json

import pytest

from backend.inference.contracts import (
    generation_schema, normalize_candidate_frame_citations, parse_candidate_observation,
    parse_observation, verification_verdict,
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
    assert "observations" not in alternatives[0]["properties"]
    assert set(alternatives[0]["properties"]["evidence"]["required"]) == {
        "multiple_people_visible", "physical_contact", "aggressive_motion",
        "temporal_persistence",
    }
    assert alternatives[0]["properties"]["evidence_frames"]["properties"][
        "physical_contact"
    ]["items"]["enum"] == [0, 1, 2, 3]


def test_lmfe_candidate_schema_forces_contract_order_and_local_indices():
    lmfe = pytest.importorskip("lmformatenforcer")
    from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig

    parser = lmfe.JsonSchemaParser(
        generation_schema("aggression_fight"),
        config=CharacterLevelParserConfig(
            force_json_field_order=True,
            max_consecutive_whitespaces=2,
        ),
    )
    prefix = (
        '{"outcome":"aggression_fight","summary":"visible altercation",'
        '"evidence":{"aggressive_motion":true,"multiple_people_visible":true,'
        '"physical_contact":true,"temporal_persistence":true},'
        '"evidence_frames":{"aggressive_motion":['
    )
    for character in prefix:
        assert character in parser.get_allowed_characters()
        parser = parser.add_character(character)

    assert set("0123").issubset(parser.get_allowed_characters())
    parser = parser.add_character("1")
    assert "0" not in parser.get_allowed_characters()


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


def test_candidate_citation_normalization_collapses_only_valid_local_duplicates():
    crowd = {
        "outcome": "crowd_gathering",
        "summary": "Multiple people remain together.",
        "evidence": {
            "duration_reached": True,
            "people_in_zone": True,
            "scene_visibility": True,
            "threshold_reached": True,
        },
        "evidence_frames": {
            "duration_reached": [0, 0, 0],
            "people_in_zone": [0, 0, 0],
            "scene_visibility": [0, 0, 0],
            "threshold_reached": [0, 0, 0],
        },
        "limitations": [],
    }
    raw = json.dumps(crowd)
    normalized, notes = normalize_candidate_frame_citations(raw, 4)
    observation = parse_candidate_observation(normalized, 4, "crowd_gathering")
    assert notes == [
        "Repeated local frame citations were collapsed for: duration_reached, "
        "people_in_zone, scene_visibility, threshold_reached."
    ]
    assert all(item.frame_indices == [0] for item in observation.observations)
    assert json.loads(raw)["evidence_frames"]["duration_reached"] == [0, 0, 0]

    crowd["evidence_frames"]["duration_reached"] = [4, 4]
    invalid_raw = json.dumps(crowd)
    partially_normalized, invalid_notes = normalize_candidate_frame_citations(invalid_raw, 4)
    assert json.loads(partially_normalized)["evidence_frames"]["duration_reached"] == [4, 4]
    assert invalid_notes
    with pytest.raises(ValueError, match="outside this request"):
        parse_candidate_observation(partially_normalized, 4, "crowd_gathering")


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
