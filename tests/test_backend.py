from __future__ import annotations

from pathlib import Path
from time import monotonic, sleep

from fastapi.testclient import TestClient

from backend.contracts import FixtureAnalysis, QueryIntent, ReviewRequest
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
        frame = np.full((48, 64, 3), index * 20, dtype=np.uint8)
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
    assert len(store.list_events(QueryIntent())) == 1


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


def test_api_fixture_and_citations() -> None:
    from backend.main import app

    with TestClient(app) as client:
        response = client.post("/api/fixtures/run")
        assert response.status_code == 200
        assert response.json()["fixture"] is True
        query = client.post("/api/query", json={"operation": "list"})
        assert query.status_code == 200
        assert query.json()["citations"]


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


def test_replay_api_starts_background_job_and_rejects_path_escape(tmp_path, monkeypatch) -> None:
    from backend import main

    recordings = tmp_path / "recordings"
    recordings.mkdir()
    write_test_video(recordings / "entrance.avi")
    manager = ReplayManager(recordings, tmp_path / "assets")
    monkeypatch.setattr(main, "replay_manager", manager)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/replays",
            json={"source_name": "entrance.avi", "pace": False, "max_frames": 2},
        )
        assert response.status_code == 202
        session_id = response.json()["session_id"]
        assert client.get(f"/api/replays/{session_id}").status_code == 200

        escaped = client.post("/api/replays", json={"source_name": "../outside.avi"})
        assert escaped.status_code == 400
