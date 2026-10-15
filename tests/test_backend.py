from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.contracts import FixtureAnalysis, QueryIntent, ReviewRequest
from backend.storage import Store
from backend.ingest.replay import BoundedFrameBuffer, FixtureReplay, FramePacket, VideoReplay


def make_store() -> Store:
    store = Store()
    store.seed_camera()
    return store


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
    import cv2
    import numpy as np

    video = tmp_path / "source.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48)
    )
    assert writer.isOpened()
    for index in range(10):
        frame = np.full((48, 64, 3), index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()

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
