import json

import pytest

from backend.inference.contracts import generation_schema


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


def test_generation_schema_rejects_exact_duplicate_observation_objects():
    duplicate = json.dumps({"outcome": "person_down", "summary": "A person.",
                            "observations": [
                                {"predicate": "visible_person", "supported": True, "frame_indices": [0]},
                                {"predicate": "visible_person", "supported": True, "frame_indices": [0]},
                            ], "limitations": []})
    assert not accepts(duplicate)
