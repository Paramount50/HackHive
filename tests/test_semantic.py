import json
from threading import Event
from time import monotonic, sleep

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
    service = SemanticInferenceService(
        FakeWorker(), lambda job, result: (results.append((job, result)), completed.set())
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
    assert event["summary"].startswith("A visible person")
    window_id = json.loads(event["window_ids_json"])[-1]
    window = store.connection.execute(
        "SELECT reason,status,model_outcome FROM analysis_windows WHERE window_id=?", (window_id,)
    ).fetchone()
    assert dict(window) == {
        "reason": "vlm_verification", "status": "completed", "model_outcome": "person_down"
    }
