import json
from threading import Event
from time import monotonic, sleep

import pytest
from PIL import Image

from backend.contracts import FixtureAnalysis
from backend.inference.contracts import VisualObservation, verification_verdict
from backend.inference.service import SemanticInferenceService, SemanticJob
from backend.storage import Store


def supported_observation() -> dict:
    return {
        "outcome": "person_down",
        "summary": "A visible person remains horizontally postured across the selected frames.",
        "observations": [
            {"predicate": "visible_person", "supported": True, "frame_indices": [0, 3]},
            {"predicate": "prone_posture", "supported": True, "frame_indices": [1, 2, 3]},
            {"predicate": "persistence", "supported": True, "frame_indices": [0, 1, 2, 3]},
            {"predicate": "usable_context", "supported": True, "frame_indices": [0, 3]},
        ],
        "limitations": ["Operator review is required."],
    }


def semantic_job(paths: tuple[str, str, str, str]) -> SemanticJob:
    return SemanticJob(
        event_id="event-1",
        camera_id="cam-1",
        candidate_key="candidate-1",
        expected_outcome="person_down",
        image_paths=paths,
        source_pts_ms=(0, 1_000, 2_000, 3_000),
    )


def test_verification_requires_category_specific_supported_evidence() -> None:
    observation = VisualObservation.model_validate(supported_observation())
    assert verification_verdict(observation, "person_down") == "supported"
    missing = observation.model_copy(update={"observations": observation.observations[:1]})
    assert verification_verdict(missing, "person_down") == "insufficient_evidence"
    normal = VisualObservation(
        outcome="none", summary="No supported incident is visible.", observations=[], limitations=[]
    )
    assert verification_verdict(normal, "person_down") == "contradicted"


def test_verification_distinguishes_false_unknown_and_missing_required_evidence() -> None:
    false_evidence = VisualObservation.model_validate({
        "outcome": "person_down",
        "summary": "A person is visible but not prone.",
        "observations": [
            {"predicate": "visible_person", "supported": True, "frame_indices": [0]},
            {"predicate": "prone_posture", "supported": False, "frame_indices": [0, 1]},
        ],
        "limitations": [],
    })
    assert verification_verdict(false_evidence, "person_down") == "contradicted"

    unknown_evidence = VisualObservation.model_validate({
        "outcome": "person_down",
        "summary": "The posture is unclear.",
        "observations": [
            {"predicate": "visible_person", "supported": True, "frame_indices": [0]},
            {"predicate": "prone_posture", "supported": None, "frame_indices": [0, 1]},
        ],
        "limitations": [],
    })
    assert verification_verdict(unknown_evidence, "person_down") == "inconclusive"

    missing_evidence = VisualObservation.model_validate({
        "outcome": "person_down",
        "summary": "A person is visible.",
        "observations": [
            {"predicate": "visible_person", "supported": True, "frame_indices": [0]},
        ],
        "limitations": [],
    })
    assert verification_verdict(missing_evidence, "person_down") == "insufficient_evidence"


def test_person_down_with_empty_evidence_is_not_supported() -> None:
    observation = VisualObservation.model_validate({
        "outcome": "person_down",
        "summary": "A person is standing near a checkout counter.",
        "observations": [],
        "limitations": ["The view does not show a prone posture."],
    })
    assert verification_verdict(observation, "person_down") == "insufficient_evidence"


DECISION_EVIDENCE = {
    "person_down": ["visible_person", "prone_posture", "persistence", "usable_context"],
    "crowd_gathering": [
        "people_in_zone", "threshold_reached", "duration_reached", "scene_visibility",
    ],
    "suspected_tailgating": [
        "distinct_people", "same_direction_passages", "temporal_proximity", "gate_visibility",
    ],
    "aggression_fight": [
        "multiple_people_visible", "physical_contact", "aggressive_motion", "temporal_persistence",
    ],
    "fire_smoke": [
        "flame_like_region", "smoke_like_region", "temporal_persistence", "scene_visibility",
    ],
    "intrusion_unauthorized_access": [
        "person_in_restricted_zone", "restricted_boundary_crossing",
        "scene_visibility", "temporal_persistence",
    ],
}


def category_observation(outcome: str, **answers: bool | None) -> VisualObservation:
    values = {predicate: True for predicate in DECISION_EVIDENCE[outcome]}
    values.update(answers)
    return VisualObservation.model_validate({
        "outcome": outcome,
        "summary": "Category-conditioned visual evidence.",
        "observations": [
            {"predicate": predicate, "supported": supported, "frame_indices": [0, 1]}
            for predicate, supported in values.items()
        ],
        "limitations": [],
    })


@pytest.mark.parametrize("outcome", DECISION_EVIDENCE)
def test_each_category_requires_a_complete_positive_decision_contract(outcome: str) -> None:
    assert verification_verdict(category_observation(outcome), outcome) == "supported"


@pytest.mark.parametrize("outcome,predicate", [
    ("person_down", "visible_person"),
    ("person_down", "prone_posture"),
    ("person_down", "persistence"),
    ("crowd_gathering", "people_in_zone"),
    ("crowd_gathering", "scene_visibility"),
    ("suspected_tailgating", "distinct_people"),
    ("suspected_tailgating", "same_direction_passages"),
    ("suspected_tailgating", "temporal_proximity"),
    ("suspected_tailgating", "gate_visibility"),
    ("aggression_fight", "multiple_people_visible"),
    ("aggression_fight", "physical_contact"),
    ("aggression_fight", "aggressive_motion"),
    ("aggression_fight", "temporal_persistence"),
    ("fire_smoke", "scene_visibility"),
    ("fire_smoke", "temporal_persistence"),
    ("intrusion_unauthorized_access", "person_in_restricted_zone"),
    ("intrusion_unauthorized_access", "restricted_boundary_crossing"),
    ("intrusion_unauthorized_access", "scene_visibility"),
    ("intrusion_unauthorized_access", "temporal_persistence"),
])
def test_false_or_null_required_evidence_cannot_support(
    outcome: str, predicate: str
) -> None:
    assert verification_verdict(category_observation(outcome, **{predicate: False}), outcome) == "contradicted"
    assert verification_verdict(category_observation(outcome, **{predicate: None}), outcome) == "inconclusive"


@pytest.mark.parametrize("predicate", ["threshold_reached", "duration_reached"])
def test_crowd_visual_disagreement_does_not_override_tracker_facts(predicate: str) -> None:
    assert verification_verdict(
        category_observation("crowd_gathering", **{predicate: False}), "crowd_gathering"
    ) == "inconclusive"
    assert verification_verdict(
        category_observation("crowd_gathering", **{predicate: None}), "crowd_gathering"
    ) == "inconclusive"


def test_person_down_unusable_or_unknown_context_is_inconclusive() -> None:
    for answer in (False, None):
        assert verification_verdict(
            category_observation("person_down", usable_context=answer), "person_down"
        ) == "inconclusive"


def test_fire_accepts_fire_or_smoke_but_not_absence_or_ambiguity_of_both() -> None:
    assert verification_verdict(category_observation(
        "fire_smoke", flame_like_region=True, smoke_like_region=False
    ), "fire_smoke") == "supported"
    assert verification_verdict(category_observation(
        "fire_smoke", flame_like_region=False, smoke_like_region=True
    ), "fire_smoke") == "supported"
    assert verification_verdict(category_observation(
        "fire_smoke", flame_like_region=False, smoke_like_region=False
    ), "fire_smoke") == "contradicted"
    assert verification_verdict(category_observation(
        "fire_smoke", flame_like_region=False, smoke_like_region=None
    ), "fire_smoke") == "inconclusive"


def test_none_and_unknown_remain_distinct_for_expanded_contract() -> None:
    none = VisualObservation(
        outcome="none", summary="Visible normal scene.", observations=[], limitations=[]
    )
    unknown = VisualObservation(
        outcome="unknown", summary="The view is obstructed.", observations=[], limitations=[]
    )
    assert verification_verdict(none, "aggression_fight") == "contradicted"
    assert verification_verdict(unknown, "aggression_fight") == "inconclusive"


def test_semantic_service_runs_one_bounded_background_worker(tmp_path) -> None:
    paths = []
    for index in range(4):
        path = tmp_path / f"frame-{index}.jpg"
        Image.new("RGB", (32, 24), (index * 40, 0, 0)).save(path)
        paths.append(str(path))

    class FakeWorker:
        def load(self):
            return {"model_id": "fake-vlm", "revision": "test"}

        def analyze(self, images, pts_ms):
            assert len(images) == 4
            assert pts_ms == [0, 1_000, 2_000, 3_000]
            return {"status": "completed", "observation": supported_observation()}

    completed = Event()
    results = []
    states = []
    service = SemanticInferenceService(
        FakeWorker(), lambda job, result: (results.append((job, result)), completed.set()),
        on_state=lambda job, result: states.append(result["status"]),
    )
    service.start()
    try:
        assert service.submit(semantic_job(tuple(paths)))
        assert completed.wait(2)
        assert results[0][1]["status"] == "completed"
        deadline = monotonic() + 1
        while service.snapshot()["pending"] and monotonic() < deadline:
            sleep(0.01)
        assert service.snapshot()["state"] == "ready"
        assert service.snapshot()["pending"] == 0
        assert states == ["pending", "running"]
    finally:
        service.close()


def test_semantic_service_uses_worker_frame_selector(tmp_path) -> None:
    paths = []
    for index in range(6):
        path = tmp_path / f"pool-{index}.jpg"
        Image.new("RGB", (32, 24), (index * 30, 0, 0)).save(path)
        paths.append(str(path))

    class SelectingWorker:
        def load(self):
            return {"model_id": "fake-vlm", "revision": "test"}

        def select_frames(self, images, target_frames=4):
            assert len(images) == 6
            assert target_frames == 4
            return [0, 2, 4, 5]

        def analyze(self, images, pts_ms):
            assert pts_ms == [0, 1_000, 2_000, 2_500]
            return {"status": "completed", "observation": supported_observation()}

    completed = Event()
    results = []
    service = SemanticInferenceService(
        SelectingWorker(), lambda job, result: (results.append(result), completed.set())
    )
    service.start()
    try:
        job = SemanticJob(
            event_id="event-pool",
            camera_id="cam-1",
            candidate_key="candidate-pool",
            expected_outcome="person_down",
            image_paths=tuple(paths),
            source_pts_ms=(0, 500, 1_000, 1_500, 2_000, 2_500),
        )
        assert service.submit(job)
        assert completed.wait(2)
        selection = results[0]["selection"]
        assert selection["strategy"] == "swift_latent_taylor_v1"
        assert selection["candidate_frame_count"] == 6
        assert selection["selected_indices"] == [0, 2, 4, 5]
        assert selection["selected_source_pts_ms"] == [0, 1_000, 2_000, 2_500]
        assert selection["target_frames"] == 4
        assert selection["taylor_order"] == 3
        assert selection["latency_ms"] >= 0
        assert selection["error"] is None
    finally:
        service.close()


def test_semantic_service_can_force_uniform_baseline(tmp_path) -> None:
    paths = []
    for index in range(6):
        path = tmp_path / f"uniform-{index}.jpg"
        Image.new("RGB", (32, 24), (index * 30, 0, 0)).save(path)
        paths.append(str(path))

    class Worker:
        def load(self):
            return {"model_id": "fake-vlm", "revision": "test"}

        def select_frames(self, images, target_frames=4):
            raise AssertionError("uniform baseline must bypass Swift selection")

        def analyze(self, images, pts_ms):
            assert pts_ms == [0, 1_000, 1_500, 2_500]
            return {"status": "completed", "observation": supported_observation()}

    completed = Event()
    results = []
    service = SemanticInferenceService(
        Worker(), lambda job, result: (results.append(result), completed.set()),
        selection_strategy="uniform",
    )
    service.start()
    try:
        assert service.submit(SemanticJob(
            event_id="uniform-event",
            camera_id="cam-1",
            candidate_key="uniform-candidate",
            expected_outcome="person_down",
            image_paths=tuple(paths),
            source_pts_ms=(0, 500, 1_000, 1_500, 2_000, 2_500),
        ))
        assert completed.wait(2)
        assert results[0]["selection"]["strategy"] == "uniform_requested"
        assert service.snapshot()["selection_strategy"] == "uniform"
    finally:
        service.close()


def test_store_attaches_supported_vlm_result_without_auto_accepting() -> None:
    store = Store()
    session = store.create_session(10_000)
    event_id = store.record_analysis(
        session,
        10_000,
        13_000,
        FixtureAnalysis(
            outcome="person_down",
            summary="CPU candidate",
            observations=[{"predicate": "cpu_candidate_rule", "supported": True}],
            frame_indices=[3],
            model_id="candidate://cpu-rules",
        ),
        "candidate-1",
        start_pts_ms=0,
    )
    assert event_id
    job = SemanticJob(
        event_id=event_id,
        camera_id="cam-1",
        candidate_key="candidate-1",
        expected_outcome="person_down",
        image_paths=("0.jpg", "1.jpg", "2.jpg", "3.jpg"),
        source_pts_ms=(0, 1_000, 2_000, 3_000),
    )
    verdict = store.record_vlm_result(job, {
        "status": "completed",
        "observation": supported_observation(),
        "raw_output": "{}",
        "model_id": "fake-vlm",
        "model_revision": "test",
        "prompt_version": "test-v1",
        "timings_ms": {"total": 12.5},
    })

    assert verdict == "supported"
    event = store.get_event_row(event_id)
    assert event["decision_state"] == "needs_review"
    assert event["summary"] == (
        "Possible person-down incident. Visual verification supports a visible person, "
        "prone posture, posture persistence and usable scene context."
    )
    window_id = json.loads(event["window_ids_json"])[-1]
    window = store.connection.execute(
        "SELECT reason,status,model_outcome,observations_json FROM analysis_windows WHERE window_id=?",
        (window_id,)
    ).fetchone()
    assert window["reason"] == "vlm_verification"
    assert window["status"] == "completed"
    assert window["model_outcome"] == "person_down"
    audit_payload = json.loads(window["observations_json"])
    assert audit_payload["observation"]["summary"].startswith("A visible person")
    assert audit_payload["raw_output"] == "{}"
    projected = store.get_event(event_id)["verification"]
    assert projected["status"] == "completed"
    assert projected["verdict"] == "supported"


def test_semantic_states_upsert_one_visible_attempt() -> None:
    store = Store()
    session = store.create_session(10_000)
    event_id = store.record_analysis(
        session, 10_000, 13_000,
        FixtureAnalysis(
            outcome="person_down", summary="CPU candidate",
            observations=[{"predicate": "cpu_candidate_rule", "supported": True}],
            model_id="candidate://cpu-rules",
        ),
        "state-candidate", start_pts_ms=0,
    )
    job = SemanticJob(
        event_id=event_id, camera_id="cam-1", candidate_key="state-candidate",
        expected_outcome="person_down", image_paths=("0", "1", "2", "3"),
        source_pts_ms=(0, 1_000, 2_000, 3_000), verification_id="attempt-1",
    )
    assert store.record_vlm_result(job, {"status": "pending"}) == "pending"
    assert store.get_event(event_id)["verification"]["status"] == "pending"
    assert store.record_vlm_result(job, {
        "status": "failed", "error_code": "model_load_failed", "error": "fixture failure",
    }) == "failed"
    verification = store.get_event(event_id)["verification"]
    assert verification["status"] == "failed"
    assert verification["error_code"] == "model_load_failed"
    count = store.connection.execute(
        "SELECT COUNT(*) FROM analysis_windows WHERE window_id='attempt-1'"
    ).fetchone()[0]
    assert count == 1


def test_semantic_service_exposes_coalescing_without_leaking_old_job() -> None:
    states = []
    service = SemanticInferenceService(
        object(), lambda job, result: None,
        on_state=lambda job, result: states.append((job.verification_id, result)),
    )
    old = semantic_job(("0", "1", "2", "3"))
    newer = semantic_job(("4", "5", "6", "7"))
    assert service.submit(old)
    assert service.submit(newer)
    assert [result["status"] for _, result in states] == ["pending", "rejected", "pending"]
    assert states[1][1]["error_code"] == "coalesced_by_newer_request"
    assert len(service._jobs) == 1


def test_semantic_service_exposes_expiry_after_slow_load() -> None:
    class SlowLoadingWorker:
        def load(self):
            sleep(0.05)
            return {"model_id": "fixture", "revision": "test"}

    terminal = Event()
    states = []

    def capture(job, result):
        states.append(result)
        if result["status"] == "expired":
            terminal.set()

    service = SemanticInferenceService(
        SlowLoadingWorker(), lambda job, result: None,
        deadline_seconds=0.01, on_state=capture,
    )
    assert service.submit(semantic_job(("0", "1", "2", "3")))
    service.start()
    try:
        assert terminal.wait(2)
        assert [result["status"] for result in states] == ["pending", "expired"]
        assert states[-1]["error_code"] == "deadline_exceeded"
    finally:
        service.close()
