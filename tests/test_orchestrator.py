from pathlib import Path

import cv2
import numpy as np

from backend.contracts import QueryIntent
from backend.ingest.replay import FramePacket
from backend.media import MediaExporter
from backend.orchestrator import EventOrchestrator
from backend.perception.pipeline import BoxDetection, PerceptionPipeline
from backend.perception.rules import CandidateRules
from backend.storage import Store


class ScriptedDetector:
    def __init__(self, frames):
        self.frames = iter(frames)

    def detect(self, image_path):
        return next(self.frames)


def test_cpu_candidate_becomes_review_incident_with_media_and_end(tmp_path) -> None:
    images = []
    packets = []
    for index in range(9):
        path = tmp_path / f"frame-{index}.jpg"
        assert cv2.imwrite(str(path), np.full((48, 64, 3), index * 20, dtype=np.uint8))
        images.append(path)
        packets.append(FramePacket(
            "cam-1", "session-1", index, index * 1_000, 10_000 + index * 1_000,
            64, 48, str(path), False,
        ))
    horizontal = BoxDetection(5, 15, 55, 35, 0.9, "person")
    detector = ScriptedDetector([[horizontal], [horizontal], [horizontal], [], [], [], [], [], []])
    store = Store()
    store.seed_camera()
    store.connection.execute(
        """INSERT INTO stream_sessions
        (session_id,camera_id,mode,started_utc_ms,policy_version,policy_json)
        VALUES ('session-1','cam-1','replay',10000,'cam-1-v1','{}')"""
    )
    store.connection.commit()
    orchestrator = EventOrchestrator(
        session_id="session-1", session_started_utc_ms=10_000,
        pipeline=PerceptionPipeline(detector), store=store,
        media=MediaExporter(tmp_path / "media"),
    )
    touched = []
    for packet in packets:
        touched.extend(orchestrator.process(packet))

    assert len(set(touched)) == 1
    event = store.get_event_row(touched[0])
    assert event["decision_state"] == "needs_review"
    assert event["lifecycle_state"] == "ended"
    assert event["end_utc_ms"] == 12_000
    media = store.connection.execute(
        "SELECT kind,status,relative_path FROM media WHERE event_id=? ORDER BY kind", (event["event_id"],)
    ).fetchall()
    assert {(item["kind"], item["status"]) for item in media} == {
        ("event_clip", "ready"), ("keyframe", "ready")
    }
    assert all((tmp_path / "media" / item["relative_path"]).is_file() for item in media)


def test_crowd_occurrence_merges_when_tracker_ids_change_and_finishes(tmp_path) -> None:
    packets = []
    detector_frames = []
    for frame_index in range(5):
        path = tmp_path / f"crowd-{frame_index}.jpg"
        assert cv2.imwrite(str(path), np.full((100, 320, 3), 40, dtype=np.uint8))
        pts = frame_index * 500
        packets.append(FramePacket("cam-1", "crowd-session", frame_index, pts, 20_000 + pts,
                                   320, 100, str(path), False))
        shift = frame_index * 25
        detector_frames.append([
            BoxDetection(10 + shift + person_index * 50, 20, 30 + shift + person_index * 50, 70, 0.9, "person")
            for person_index in range(4)
        ])
    store = Store()
    store.seed_camera()
    store.connection.execute(
        """INSERT INTO stream_sessions
        (session_id,camera_id,mode,started_utc_ms,policy_version,policy_json)
        VALUES ('crowd-session','cam-1','replay',20000,'cam-1-v1','{}')"""
    )
    store.connection.commit()
    orchestrator = EventOrchestrator(
        session_id="crowd-session",
        session_started_utc_ms=20_000,
        pipeline=PerceptionPipeline(
            ScriptedDetector(detector_frames),
            rules=CandidateRules(crowd_minimum_people=4, crowd_duration_ms=1_000),
        ),
        store=store,
        media=MediaExporter(tmp_path / "crowd-media"),
    )
    touched = []
    for packet in packets:
        touched.extend(orchestrator.process(packet))
    orchestrator.finish(packets[-1])

    assert len(set(touched)) == 1
    events = store.list_events(QueryIntent())
    assert len(events) == 1
    assert events[0]["category"] == "crowd_gathering"
    assert events[0]["lifecycle_state"] == "ended"
    media = store.connection.execute(
        "SELECT kind,status FROM media WHERE event_id=?", (events[0]["event_id"],)
    ).fetchall()
    assert {(item["kind"], item["status"]) for item in media} == {
        ("event_clip", "ready"), ("keyframe", "ready")
    }
