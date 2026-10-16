from __future__ import annotations

import json
import hashlib
from pathlib import Path
from time import monotonic, sleep

import pytest
from fastapi.testclient import TestClient

from backend.contracts import FixtureAnalysis, QueryIntent, ReplayStartRequest, ReviewRequest
from backend.media import MediaArtifact
from backend.storage import Store
from backend.ingest.manager import ReplayManager
from backend.ingest.replay import BoundedFrameBuffer, FixtureReplay, FramePacket, VideoReplay


def make_store() -> Store:
    store = Store()
    store.seed_camera()
    return store


def write_test_video(path: Path, frame_count: int = 10) -> None:
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48)
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.full((48, 64, 3), (index * 20) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_unknown_is_not_an_incident() -> None:
    store = make_store()
    session = store.create_session(1_000)
    analysis = FixtureAnalysis(outcome="unknown", summary="Frames unavailable", limitations=["timeout"])
    assert store.record_analysis(session, 2_000, 3_000, analysis, "unknown-1") is None
    assert store.list_events(QueryIntent()).__len__() == 0


def test_repeated_observation_merges_into_one_event() -> None:
    store = make_store()
    session = store.create_session(1_000)
    analysis = FixtureAnalysis(
        outcome="person_down",
        summary="A person remains prone near the doorway.",
        observations=[{"predicate": "prone_posture", "supported": True}],
    )
    first = store.record_analysis(session, 2_000, 3_000, analysis, "person-1")
    second = store.record_analysis(session, 3_000, 4_000, analysis, "person-1")
    assert first == second
    result = store.list_events(QueryIntent())
    assert len(result) == 1
    assert result[0]["lifecycle_state"] == "active"
    assert result[0]["end_utc_ms"] is None
    assert len(json.loads(result[0]["window_ids_json"])) == 2


def test_quiet_gap_closes_event_and_recurrence_gets_new_id() -> None:
    store = make_store()
    session = store.create_session(1_000)
    analysis = FixtureAnalysis(
        outcome="person_down",
        summary="A person remains prone near the doorway.",
        observations=[{"predicate": "prone_posture", "supported": True}],
    )
    first = store.record_analysis(session, 2_000, 3_000, analysis, "person-quiet")
    assert store.close_inactive_events(observed_utc_ms=8_001, merge_gap_ms=5_000) == [first]
    closed = next(item for item in store.list_events(QueryIntent()) if item["event_id"] == first)
    assert closed["lifecycle_state"] == "ended"
    assert closed["end_utc_ms"] == 3_000

    second = store.record_analysis(session, 9_000, 10_000, analysis, "person-quiet")
    assert second != first
    assert len(store.list_events(QueryIntent())) == 2


def test_review_is_idempotent_and_separate_from_system_decision() -> None:
    store = make_store()
    session = store.create_session(1_000)
    analysis = FixtureAnalysis(
        outcome="crowd_gathering",
        summary="Four people remain in the marked zone.",
        observations=[{"predicate": "people_in_zone", "supported": True}],
    )
    event_id = store.record_analysis(session, 2_000, 3_000, analysis, "crowd-1")
    request = ReviewRequest(action="confirm", actor_label="tester", request_key="review-1")
    store.review(event_id, request)
    store.review(event_id, request)
    result = store.list_events(QueryIntent())[0]
    assert result["decision_state"] == "accepted"
    assert result["review_status"] == "confirmed"
    assert len(result["reviews"]) == 1
    assert store.list_events(QueryIntent(review_statuses=["confirmed"]))[0]["event_id"] == event_id
    assert store.list_events(QueryIntent(review_statuses=["unreviewed"])) == []


def test_review_action_changes_use_distinct_idempotency_keys() -> None:
    store = make_store()
    session = store.create_session(1_000)
    analysis = FixtureAnalysis(
        outcome="person_down", summary="A person remains prone.",
        observations=[{"predicate": "prone_posture", "supported": True}],
    )
    event_id = store.record_analysis(session, 2_000, 3_000, analysis, "review-cycle")
    for action, request_key in (
        ("confirm", "cycle-confirm-1"),
        ("reject", "cycle-reject"),
        ("confirm", "cycle-confirm-2"),
    ):
        store.review(
            event_id,
            ReviewRequest(action=action, actor_label="tester", request_key=request_key),
        )
    result = store.get_event(event_id)
    assert result["review_status"] == "confirmed"
    assert [entry["action"] for entry in result["reviews"]] == ["confirm", "reject", "confirm"]
    with pytest.raises(ValueError, match="already used"):
        store.review(
            event_id,
            ReviewRequest(action="reject", actor_label="tester", request_key="cycle-confirm-1"),
        )


def test_unreviewed_list_prioritizes_uncertain_vlm_results() -> None:
    store = make_store()
    session = store.create_session(1_000)
    analysis = FixtureAnalysis(
        outcome="person_down", summary="A person is visible.",
        observations=[{"predicate": "visible_person", "supported": True}],
    )
    supported = store.record_analysis(session, 5_000, 5_500, analysis, "priority-supported")
    inconclusive = store.record_analysis(session, 3_000, 3_500, analysis, "priority-inconclusive")
    contradicted = store.record_analysis(session, 2_000, 2_500, analysis, "priority-contradicted")

    for event_id, verdict in (
        (supported, "supported"),
        (inconclusive, "inconclusive"),
        (contradicted, "contradicted"),
    ):
        row = store.get_event_row(event_id)
        window_id = f"vlm-{event_id}"
        store.connection.execute(
            """INSERT INTO analysis_windows
               (window_id,session_id,start_utc_ms,end_utc_ms,frame_count,reason,status,
                model_outcome,model_id,model_revision,prompt_version,observations_json,timings_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                window_id, session, 2_000, 3_000, 4, "vlm_verification", "completed",
                "person_down", "fixture://vlm", "v1", "test",
                json.dumps({"verdict": verdict}), "{}",
            ),
        )
        window_ids = json.loads(row["window_ids_json"])
        window_ids.append(window_id)
        store.connection.execute(
            "UPDATE events SET window_ids_json=? WHERE event_id=?",
            (json.dumps(window_ids), event_id),
        )
    store.connection.commit()

    ordered = store.list_events(QueryIntent(review_statuses=["unreviewed"]))
    assert [event["event_id"] for event in ordered[:2]] == [inconclusive, contradicted]
    assert ordered[2]["event_id"] == supported
    recent_first = store.list_events(QueryIntent())
    assert [event["event_id"] for event in recent_first] == [supported, inconclusive, contradicted]


def test_source_offsets_filter_by_recording_and_report_durable_source_name() -> None:
    store = make_store()
    started = 10_000
    session = store.create_session(started, source_name="entrance.mp4")
    event_id = store.record_analysis(
        session,
        started + 42_400,
        started + 49_200,
        FixtureAnalysis(
            outcome="person_down", summary="A person remained on the ground.",
            observations=[{"predicate": "prone_posture", "supported": True}],
        ),
        "offset-event",
        start_pts_ms=42_400,
    )
    matches = store.list_events(QueryIntent(
        source_name="entrance.mp4", start_pts_ms=42_000, end_pts_ms=50_000,
    ))
    assert [event["event_id"] for event in matches] == [event_id]
    assert matches[0]["source_name"] == "entrance.mp4"
    assert store.count_events(QueryIntent(
        source_name="entrance.mp4", start_pts_ms=50_000, end_pts_ms=60_000,
    )) == 0


def test_event_lookup_and_review_do_not_depend_on_list_limit() -> None:
    store = make_store()
    session = store.create_session(1_000)
    analysis = FixtureAnalysis(
        outcome="person_down",
        summary="A person remains prone.",
        observations=[{"predicate": "prone_posture", "supported": True}],
    )
    oldest = None
    for index in range(105):
        event_id = store.record_analysis(
            session, 2_000 + index * 10, 2_005 + index * 10,
            analysis, f"limit-event-{index}",
        )
        oldest = oldest or event_id
    assert oldest not in {item["event_id"] for item in store.list_events(QueryIntent(limit=100))}
    assert store.get_event(oldest)["event_id"] == oldest
    reviewed = store.review(
        oldest,
        ReviewRequest(action="confirm", actor_label="tester", request_key="oldest-review"),
    )
    assert reviewed["review_status"] == "confirmed"


def test_query_uses_half_open_time_filters() -> None:
    store = make_store()
    session = store.create_session(1_000)
    analysis = FixtureAnalysis(
        outcome="person_down", summary="A person is lying on the floor.",
        observations=[{"predicate": "prone_posture", "supported": True}],
    )
    store.record_analysis(session, 2_000, 3_000, analysis, "time-1")
    assert len(store.list_events(QueryIntent(start_utc_ms=3_000))) == 1
    assert len(store.list_events(QueryIntent(end_utc_ms=2_000))) == 0


def test_replay_policy_is_validated_and_persisted() -> None:
    request = ReplayStartRequest(source_name="entrance.avi", gate_line_y=0.55, zone=(0.1, 0.2, 0.8, 0.9))
    assert request.gate_line_y == 0.55
    try:
        ReplayStartRequest(source_name="entrance.avi", zone=(0.8, 0.2, 0.1, 0.9))
    except ValueError:
        pass
    else:
        raise AssertionError("unordered zone should be rejected")

    store = make_store()
    session_id = store.create_session(
        1_000, policy_version="replay-policy-v1",
        policy_json={"gate_line_y": request.gate_line_y, "zone": request.zone},
    )
    row = store.connection.execute(
        "SELECT policy_version,policy_json FROM stream_sessions WHERE session_id=?",
        (session_id,),
    ).fetchone()
    assert row["policy_version"] == "replay-policy-v1"
    assert json.loads(row["policy_json"]) == {"gate_line_y": 0.55, "zone": [0.1, 0.2, 0.8, 0.9]}


def test_visual_tailgating_candidate_does_not_claim_simulated_access_evidence() -> None:
    store = make_store()
    session_id = store.create_session(1_000)
    candidate = FixtureAnalysis(
        outcome="suspected_tailgating",
        summary="Two people crossed the configured gate in the same direction.",
        model_id="candidate://cpu-rules",
        observations=[{"predicate": "same_direction_passages", "supported": True}],
    )
    event_id = store.record_analysis(session_id, 2_000, 3_000, candidate, "visual-only-tailgate")
    event = store.get_event_row(event_id)
    assert event["authorization_state"] == "unknown"
    assert event["access_evidence_source"] is None


def test_visual_intrusion_candidate_keeps_authorization_unknown() -> None:
    store = make_store()
    session_id = store.create_session(1_000)
    candidate = FixtureAnalysis(
        outcome="intrusion_unauthorized_access",
        summary="A person crossed a configured restricted boundary.",
        model_id="candidate://cpu-rules",
        observations=[{"predicate": "restricted_boundary_crossing", "supported": True}],
    )
    event_id = store.record_analysis(session_id, 2_000, 3_000, candidate, "visual-only-intrusion")
    event = store.get_event_row(event_id)
    assert event["authorization_state"] == "unknown"
    assert event["access_evidence_source"] is None


def test_sliding_xclip_windows_reuse_the_active_followup_occurrence() -> None:
    store = make_store()
    session_id = store.create_session(1_000)
    candidate = FixtureAnalysis(
        outcome="aggression_fight", summary="XCLIP candidate.",
        model_id="xclip://microsoft/xclip-base-patch32",
        observations=[{"predicate": "xclip_class_scores", "supported": True}],
    )
    base_key = "session:aggression_fight:camera-window"
    first = store.record_analysis(session_id, 2_000, 3_000, candidate, base_key)
    store.close_inactive_events(10_000, merge_gap_ms=0, session_id=session_id)
    second = store.record_analysis(session_id, 20_000, 21_000, candidate, base_key)
    third = store.record_analysis(session_id, 21_000, 22_000, candidate, base_key)
    assert first != second
    assert third == second
    assert len(store.list_events(QueryIntent())) == 2


def test_api_fixture_and_citations() -> None:
    from backend import main

    with TestClient(main.app) as client:
        response = client.post("/api/fixtures/run")
        assert response.status_code == 200
        assert response.json()["fixture"] is True
        stored = main.store.get_event_row(response.json()["event_id"])
        assert stored["access_evidence_source"] == "simulated"
        query = client.post("/api/query", json={"operation": "list"})
        assert query.status_code == 200
        assert query.json()["citations"]


def test_media_api_transcodes_legacy_mp4_clip_for_browser(tmp_path, monkeypatch) -> None:
    import cv2
    import numpy as np
    from backend import main

    media_root = (tmp_path / "media").resolve()
    relative_path = Path("events") / "event" / "legacy.mp4"
    clip_path = media_root / relative_path
    clip_path.parent.mkdir(parents=True)
    writer = cv2.VideoWriter(
        str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48)
    )
    assert writer.isOpened()
    for index in range(8):
        writer.write(np.full((48, 64, 3), index * 20, dtype=np.uint8))
    writer.release()

    store = Store(media_root=media_root)
    store.seed_camera()
    session = store.create_session(1_000)
    event_id = store.record_analysis(
        session, 2_000, 3_000,
        FixtureAnalysis(outcome="person_down", summary="Person prone."),
        "legacy-media-event",
    )
    artifact = MediaArtifact(
        kind="event_clip", relative_path=relative_path.as_posix(), mime_type="video/mp4",
        coverage_start_utc_ms=1_000, coverage_end_utc_ms=2_000,
        requested_start_utc_ms=1_000, requested_end_utc_ms=3_000,
        byte_size=clip_path.stat().st_size,
        sha256=hashlib.sha256(clip_path.read_bytes()).hexdigest(),
    )
    media_id = store.add_media_artifact(event_id, artifact)
    store.connection.execute(
        "UPDATE media SET status='missing',failure_reason='recorded missing before file recheck' WHERE media_id=?",
        (media_id,),
    )
    store.connection.commit()
    monkeypatch.setattr(main, "store", store)
    monkeypatch.setattr(main, "media_root", media_root)

    with TestClient(main.app) as client:
        response = client.get(f"/api/media/{media_id}", headers={"Range": "bytes=0-1023"})

    assert response.status_code == 206
    assert response.headers["content-type"] == "video/webm"
    assert response.content.startswith(bytes.fromhex("1a45dfa3"))
    assert response.headers["content-range"].startswith("bytes 0-")
    updated = store.get_media(media_id)
    assert updated["mime_type"] == "video/webm"
    assert updated["relative_path"].endswith(".webm")


def test_chat_status_reports_structured_search_ready_without_optional_models(tmp_path, monkeypatch) -> None:
    from backend import main

    monkeypatch.setattr(main, "chat_enabled", True)
    monkeypatch.setattr(main, "chat_model_dir", tmp_path / "qwen")
    monkeypatch.setattr(main, "embedding_model_dir", tmp_path / "minilm")
    monkeypatch.setattr(main, "chat_model", None)
    monkeypatch.setattr(main, "semantic_ranker", None)

    query_status = main.status()["query"]
    assert query_status["state"] == "ready"
    assert query_status["semantic_models_state"] == "not_configured"
    assert query_status["enabled"] is True


def test_perception_fixture_runs_rules_before_storage() -> None:
    from backend.main import app

    with TestClient(app) as client:
        response = client.post("/api/fixtures/perception-run")
        assert response.status_code == 200
        body = response.json()
        assert body["candidate"]
        assert body["event_id"]


def test_replay_preserves_order_and_bounds_context() -> None:
    packets = [
        FramePacket("cam-1", "s", index, index * 1000, 1000 + index, 640, 360, f"fixture://{index}")
        for index in range(4)
    ]
    buffer = BoundedFrameBuffer(max_frames=2)
    for packet in FixtureReplay(packets).frames():
        buffer.append(packet)
    assert len(buffer) == 2
    assert buffer.latest.frame_id == 3
    assert [item.frame_id for item in buffer.between(2_000, 3_000)] == [2, 3]


def test_grant_consumption_is_retry_safe() -> None:
    store = make_store()
    session = store.create_session(1_000)
    store.add_grant(session, "grant-1", "door", 1_000, 10_000)
    first = store.record_passage(session, "door", "track-1", 2_000, 1_000, dedupe_key="passage-1")
    retry = store.record_passage(session, "door", "track-1", 2_000, 1_000, dedupe_key="passage-1")
    assert first["passage_id"] == retry["passage_id"]
    grant = store.connection.execute("SELECT consumed_passages FROM access_grants").fetchone()
    assert grant["consumed_passages"] == 1


def test_real_video_replay_emits_sampled_timestamped_assets(tmp_path) -> None:
    video = tmp_path / "source.avi"
    write_test_video(video)

    replay = VideoReplay(
        video,
        camera_id="cam-1",
        session_id="session-real",
        session_started_utc_ms=1_000_000,
        asset_dir=tmp_path / "frames",
        sample_fps=5,
    )
    packets = list(replay.frames())

    assert 4 <= len(packets) <= 6
    assert [packet.source_pts_ms for packet in packets] == sorted(
        packet.source_pts_ms for packet in packets
    )
    assert all(packet.width == 64 and packet.height == 48 for packet in packets)
    assert all(packet.captured_utc_ms == 1_000_000 + packet.source_pts_ms for packet in packets)
    assert all(packet.fixture is False for packet in packets)
    assert all(Path(packet.asset_id).is_file() for packet in packets)


def test_real_video_replay_rejects_missing_file(tmp_path) -> None:
    try:
        VideoReplay(
            tmp_path / "missing.mp4",
            "cam-1",
            "session-real",
            1_000,
            tmp_path / "frames",
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing video should be rejected")


def test_grab_sampling_matches_sequential_frame_count_and_timestamps(tmp_path) -> None:
    video = tmp_path / "source.avi"
    write_test_video(video, frame_count=30)
    outputs = {}
    for strategy in ("sequential", "grab"):
        replay = VideoReplay(
            video, "cam-1", f"session-{strategy}", 1_000,
            tmp_path / strategy, sample_fps=2, decode_strategy=strategy,
        )
        outputs[strategy] = list(replay.frames())
    assert [packet.source_pts_ms for packet in outputs["grab"]] == [
        packet.source_pts_ms for packet in outputs["sequential"]
    ]
    assert len(outputs["grab"]) == len(outputs["sequential"])


def test_replay_manager_decodes_in_background_and_reports_latest_frame(tmp_path) -> None:
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    write_test_video(recordings / "hallway.avi")
    manager = ReplayManager(recordings, tmp_path / "assets", context_frames=3)

    initial = manager.start(
        session_id="session-background",
        camera_id="cam-1",
        source_name="hallway.avi",
        started_utc_ms=5_000,
        sample_fps=5,
        pace=False,
        max_frames=None,
    )
    assert initial["status"] in {"starting", "running", "completed"}

    deadline = monotonic() + 3
    result = manager.status("session-background")
    while result["status"] not in {"completed", "failed"} and monotonic() < deadline:
        sleep(0.01)
        result = manager.status("session-background")

    assert result["status"] == "completed"
    assert 4 <= result["frame_count"] <= 6
    assert result["latest_frame"]["fixture"] is False
    assert Path(result["latest_frame"]["asset_id"]).is_file()


def test_replay_manager_processes_frames_and_finishes_analysis(tmp_path) -> None:
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    write_test_video(recordings / "hallway.avi")
    seen = []
    finished = []
    manager = ReplayManager(recordings, tmp_path / "assets", context_frames=3)

    manager.start(
        session_id="session-analysis",
        camera_id="cam-1",
        source_name="hallway.avi",
        started_utc_ms=5_000,
        sample_fps=5,
        pace=False,
        max_frames=None,
        on_frame=lambda packet: seen.append(packet.frame_id) or ["same-event"],
        on_finish=finished.append,
    )
    deadline = monotonic() + 3
    result = manager.status("session-analysis")
    while result["status"] not in {"completed", "failed"} and monotonic() < deadline:
        sleep(0.01)
        result = manager.status("session-analysis")

    assert result["status"] == "completed"
    assert result["analysis"] == "active"
    assert result["events_created"] == 1
    assert seen == list(range(result["frame_count"]))
    assert len(finished) == 1
    assert finished[0].frame_id == result["frame_count"] - 1


def test_replay_api_starts_background_job_and_rejects_path_escape(tmp_path, monkeypatch) -> None:
    from backend import main

    recordings = tmp_path / "recordings"
    recordings.mkdir()
    write_test_video(recordings / "entrance.avi")
    manager = ReplayManager(recordings, tmp_path / "assets")
    monkeypatch.setattr(main, "replay_manager", manager)
    monkeypatch.setattr(main, "detector_weights", tmp_path / "missing-yolo.pt")
    monkeypatch.setattr(main, "get_action_classifier", lambda: None)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/replays",
            json={"source_name": "entrance.avi", "pace": False, "max_frames": 2},
        )
        assert response.status_code == 202
        session_id = response.json()["session_id"]
        deadline = monotonic() + 3
        status = client.get(f"/api/replays/{session_id}")
        while status.json()["latest_frame"] is None and monotonic() < deadline:
            sleep(0.01)
            status = client.get(f"/api/replays/{session_id}")
        assert status.status_code == 200
        assert "asset_id" not in status.json()["latest_frame"]
        assert status.json()["latest_frame"]["frame_url"].endswith("/frame")
        frame = client.get(f"/api/replays/{session_id}/frame")
        assert frame.status_code == 200
        assert frame.headers["content-type"] == "image/jpeg"

        escaped = client.post("/api/replays", json={"source_name": "../outside.avi"})
        assert escaped.status_code == 400


def test_video_upload_streams_validates_and_registers_recording(tmp_path, monkeypatch) -> None:
    from backend import main

    source = tmp_path / "source.avi"
    write_test_video(source)
    upload_root = tmp_path / "uploads"
    monkeypatch.setattr(main, "recordings_root", upload_root)
    monkeypatch.setattr(main, "max_upload_bytes", 10 * 1024 * 1024)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/uploads",
            files={"file": ("../../entrance.avi", source.read_bytes(), "video/x-msvideo")},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["original_name"] == "entrance.avi"
        assert body["source_name"].endswith(".avi")
        assert body["frame_count"] == 10
        assert body["width"] == 64 and body["height"] == 48
        assert len(body["sha256"]) == 64
        assert (upload_root / body["source_name"]).is_file()
        listed = client.get("/api/uploads").json()["recordings"]
        assert any(item["recording_id"] == body["recording_id"] for item in listed)


def test_video_upload_rejects_unsupported_and_oversized_files(tmp_path, monkeypatch) -> None:
    from backend import main

    monkeypatch.setattr(main, "recordings_root", tmp_path / "uploads")
    with TestClient(main.app) as client:
        unsupported = client.post(
            "/api/uploads", files={"file": ("notes.txt", b"not video", "text/plain")}
        )
        assert unsupported.status_code == 422
        monkeypatch.setattr(main, "max_upload_bytes", 4)
        oversized = client.post(
            "/api/uploads", files={"file": ("large.avi", b"more-than-four", "video/x-msvideo")}
        )
        assert oversized.status_code == 413
    assert not list((tmp_path / "uploads").glob("*"))


def test_upload_to_analysis_creates_incident_and_evidence(tmp_path, monkeypatch) -> None:
    from backend import main
    from backend.perception.pipeline import BoxDetection

    source = tmp_path / "source.avi"
    write_test_video(source, frame_count=30)
    upload_root = tmp_path / "recordings"
    weights = tmp_path / "fake-yolo.pt"
    weights.write_bytes(b"test boundary only")

    class FixedDetector:
        def __init__(self, *args, **kwargs):
            pass

        def detect(self, image_path):
            return [BoxDetection(5, 15, 55, 30, 0.9, "person")]

    monkeypatch.setattr(main, "recordings_root", upload_root)
    monkeypatch.setattr(main, "replay_manager", ReplayManager(upload_root, tmp_path / "frames"))
    media_root = (tmp_path / "media").resolve()
    monkeypatch.setattr(main, "media_root", media_root)
    monkeypatch.setattr(main.store, "media_root", media_root)
    monkeypatch.setattr(main, "detector_weights", weights)
    monkeypatch.setattr(main, "YoloDetector", FixedDetector)
    monkeypatch.setattr(main, "get_action_classifier", lambda: None)
    monkeypatch.setattr(main.importlib.util, "find_spec", lambda name: object())

    with TestClient(main.app) as client:
        uploaded = client.post(
            "/api/uploads",
            files={"file": ("person-down.avi", source.read_bytes(), "video/x-msvideo")},
        )
        assert uploaded.status_code == 201
        started = client.post(
            "/api/replays",
            json={
                "source_name": uploaded.json()["source_name"],
                "sample_fps": 1,
                "pace": False,
                "max_frames": 3,
            },
        )
        assert started.status_code == 202
        session_id = started.json()["session_id"]
        deadline = monotonic() + 5
        status = client.get(f"/api/replays/{session_id}").json()
        while status["status"] not in {"completed", "failed"} and monotonic() < deadline:
            sleep(0.01)
            status = client.get(f"/api/replays/{session_id}").json()

        assert status["status"] == "completed"
        assert status["analysis"] == "active"
        assert status["events_created"] == 1
        timeline = client.get("/api/events", params={"category": "person_down"}).json()["events"]
        event = next(item for item in timeline if item["session_id"] == session_id)
        assert event["decision_state"] == "needs_review"
        assert event["thumbnail_media_id"]
        detail = client.get(f"/api/events/{event['event_id']}").json()
        assert {item["kind"] for item in detail["media"] if item["status"] == "ready"} == {
            "keyframe", "event_clip"
        }
