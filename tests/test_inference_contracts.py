import json

import unittest

from backend.inference.contracts import parse_candidate_observation, parse_observation
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

    def test_candidate_verification_keeps_local_frame_bounds_strict(self):
        payload = {
            "outcome": "crowd_gathering",
            "summary": "People remain together.",
            "evidence": {
                "duration_reached": True,
                "people_in_zone": True,
                "scene_visibility": True,
                "threshold_reached": True,
            },
            "evidence_frames": {
                "duration_reached": [0, 0],
                "people_in_zone": [0],
                "scene_visibility": [0],
                "threshold_reached": [0],
            },
            "limitations": [],
        }
        from backend.inference.contracts import normalize_candidate_frame_citations

        normalized, notes = normalize_candidate_frame_citations(json.dumps(payload), 4)
        self.assertTrue(notes)
        self.assertEqual(
            parse_candidate_observation(normalized, 4, "crowd_gathering").observations[0].frame_indices,
            [0],
        )
        for invalid in ([4, 4], [-1, -1], [True, True], [], [0, 0, 0, 0, 0]):
            payload["evidence_frames"]["duration_reached"] = invalid
            normalized, _ = normalize_candidate_frame_citations(json.dumps(payload), 4)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_candidate_observation(normalized, 4, "crowd_gathering")

    def test_candidate_verification_rejects_more_than_four_frames(self):
        with self.assertRaisesRegex(ValueError, "one to four frames"):
            parse_candidate_observation("{}", 5, "crowd_gathering")

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
