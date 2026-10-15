from pathlib import Path

import cv2
import numpy as np

from backend.contracts import FixtureAnalysis, MediaStatus, QueryIntent
from backend.ingest.replay import FramePacket
from backend.media import MediaExporter
from backend.storage import Store


def make_packets(root: Path) -> list[FramePacket]:
    packets = []
    for index in range(5):
        path = root / f"{index}.jpg"
        assert cv2.imwrite(str(path), np.full((48, 64, 3), index * 30, dtype=np.uint8))
        packets.append(FramePacket("cam-1", "session", index, index * 500, 10_000 + index * 500, 64, 48, str(path), False))
    return packets


def test_media_export_is_atomic_and_records_actual_coverage(tmp_path) -> None:
    packets = make_packets(tmp_path)
    exporter = MediaExporter(tmp_path / "media")
    keyframe = exporter.export_keyframe("event-1", packets[2])
    clip = exporter.export_clip(
        "event-1", packets,
        requested_start_utc_ms=9_000,
        requested_end_utc_ms=13_000,
    )
    assert (exporter.root / keyframe.relative_path).is_file()
    assert keyframe.byte_size > 0 and len(keyframe.sha256) == 64
    clip_path = exporter.root / clip.relative_path
    assert clip_path.is_file() and clip.byte_size > 0
    assert clip.coverage_start_utc_ms == 10_000
    assert clip.coverage_end_utc_ms == 12_000
    capture = cv2.VideoCapture(str(clip_path))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    capture.release()
    assert not list((exporter.root / "event-1").glob("*.tmp*"))


def test_ready_artifact_and_missing_fixture_have_distinct_states(tmp_path) -> None:
    store = Store()
    session = store.create_session(1_000)
    event_id = store.record_analysis(
        session, 2_000, 3_000,
        FixtureAnalysis(outcome="person_down", summary="Person prone."),
        "media-event",
    )
    packet = make_packets(tmp_path)[0]
    artifact = MediaExporter(tmp_path / "media").export_keyframe(event_id, packet)
    ready_id = store.add_media_artifact(event_id, artifact)
    missing_id = store.add_media(event_id)
    assert store.get_media(ready_id)["status"] == MediaStatus.READY
    assert store.get_media(ready_id)["sha256"] == artifact.sha256
    assert store.get_media(missing_id)["status"] == MediaStatus.MISSING
    listed = store.list_events(QueryIntent())
    assert listed[0]["thumbnail_media_id"] == ready_id
    assert listed[0]["has_ready_media"] == 1
