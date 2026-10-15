from pathlib import Path

import cv2
import numpy as np

from backend.ingest.replay import FramePacket
from backend.media import MediaExporter
from backend.orchestrator import EventOrchestrator
from backend.perception.pipeline import BoxDetection, PerceptionPipeline
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
