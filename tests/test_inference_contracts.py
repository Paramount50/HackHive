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


class InferenceContractTests(unittest.TestCase):
    def test_valid_evidence_and_unknown_remain_distinct(self):
        self.assertEqual(parse_observation(payload(), 4).outcome, "person_down")
        self.assertEqual(parse_observation(payload(outcome="unknown", observations=[]), 4).outcome, "unknown")

    def test_invalid_model_output_is_rejected(self):
        for raw in INVALID_OUTPUTS:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_observation(raw, 4)

    def test_budget_reserves_memory_and_never_goes_negative(self):
        self.assertEqual(gpu_budget_mib(3600), 2576)
        self.assertEqual(gpu_budget_mib(8000), 4096)
        self.assertEqual(gpu_budget_mib(800), 0)
