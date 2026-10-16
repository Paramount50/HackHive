import json

import unittest

from backend.inference.contracts import parse_observation
from backend.inference.hardware import gpu_budget_mib


def payload(**changes):
    return json.dumps({"outcome": "person_down", "summary": "A person is lying down.",
                       "observations": [{"predicate": "prone_posture", "supported": True, "frame_indices": [0, 3]}],
                       "limitations": ["Cause is unknown."], **changes})


INVALID_OUTPUTS = [
    "not json", "```json\n{}\n```", payload(outcome="fire_smoke"),
    payload(observations=[{"predicate": "diagnosis", "supported": True, "frame_indices": [0]}]),
    payload(observations=[{"predicate": "prone_posture", "supported": "true", "frame_indices": [0]}]),
    payload(observations=[{"predicate": "prone_posture", "supported": True, "frame_indices": [4]}]),
    payload(outcome="suspected_tailgating", observations=[
        {"predicate": "access_mismatch", "supported": True, "frame_indices": [0]}]),
]

def evidence_payload(outcome, observations):
    return json.dumps({
        "outcome": outcome,
        "summary": "Visible scene evidence.",
        "observations": observations,
        "limitations": [],
    })


class InferenceContractTests(unittest.TestCase):
    def test_valid_evidence_and_unknown_remain_distinct(self):
        self.assertEqual(parse_observation(payload(), 4).outcome, "person_down")
        self.assertEqual(parse_observation(payload(outcome="unknown", observations=[]), 4).outcome, "unknown")

    def test_invalid_model_output_is_rejected(self):
        for raw in INVALID_OUTPUTS:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_observation(raw, 4)

    def test_duplicate_predicates_remain_invalid(self):
        duplicate = payload(observations=[
            {"predicate": "visible_person", "supported": True, "frame_indices": [0]},
            {"predicate": "visible_person", "supported": True, "frame_indices": [1]},
        ])
        with self.assertRaises(ValueError):
            parse_observation(duplicate, 4)

    def test_expanded_categories_use_only_their_allowed_predicates(self):
        valid = [
            ("aggression_fight", [
                {"predicate": "multiple_people_visible", "supported": True, "frame_indices": [0, 1]},
                {"predicate": "physical_contact", "supported": True, "frame_indices": [1, 2]},
            ]),
            ("fire_smoke", [
                {"predicate": "flame_like_region", "supported": True, "frame_indices": [0, 1]},
                {"predicate": "scene_visibility", "supported": True, "frame_indices": [0, 1]},
            ]),
            ("intrusion_unauthorized_access", [
                {"predicate": "restricted_boundary_crossing", "supported": True, "frame_indices": [1, 2]},
                {"predicate": "scene_visibility", "supported": True, "frame_indices": [0, 1]},
            ]),
        ]
        for outcome, observations in valid:
            with self.subTest(outcome=outcome):
                self.assertEqual(parse_observation(evidence_payload(outcome, observations), 4).outcome, outcome)

        with self.assertRaises(ValueError):
            parse_observation(evidence_payload("aggression_fight", [
                {"predicate": "prone_posture", "supported": True, "frame_indices": [0]},
            ]), 4)

    def test_budget_reserves_memory_and_never_goes_negative(self):
        self.assertEqual(gpu_budget_mib(3600), 2576)
        self.assertEqual(gpu_budget_mib(8000), 4096)
        self.assertEqual(gpu_budget_mib(800), 0)
