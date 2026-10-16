from pathlib import Path

import cv2
import numpy as np

from backend.contracts import FixtureAnalysis, MediaStatus, QueryIntent
from backend.ingest.replay import FramePacket
from backend.media import MediaExporter, transcode_legacy_mp4_clip
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
    assert clip_path.suffix == ".webm"
    assert clip.mime_type == "video/webm"
    assert clip.coverage_start_utc_ms == 10_000
    assert clip.coverage_end_utc_ms == 12_000
    capture = cv2.VideoCapture(str(clip_path))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    capture.release()
    assert not list((exporter.root / "event-1").glob("*.tmp*"))


def test_legacy_mp4_clip_is_transcoded_to_browser_playable_webm(tmp_path) -> None:
    legacy = tmp_path / "legacy.mp4"
    writer = cv2.VideoWriter(
        str(legacy), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48)
    )
    assert writer.isOpened()
    for index in range(8):
        writer.write(np.full((48, 64, 3), index * 20, dtype=np.uint8))
    writer.release()

    converted, size, digest = transcode_legacy_mp4_clip(legacy)

    assert converted.suffix == ".webm"
    assert converted.read_bytes().startswith(bytes.fromhex("1a45dfa3"))
    assert size == converted.stat().st_size and size > 0
    assert len(digest) == 64
    capture = cv2.VideoCapture(str(converted))
    assert capture.isOpened()
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    assert decoded == 8


def test_ready_artifact_and_missing_fixture_have_distinct_states(tmp_path) -> None:
    media_root = tmp_path / "media"
    exporter = MediaExporter(media_root)
    store = Store(media_root=media_root)
    session = store.create_session(1_000)
    event_id = store.record_analysis(
        session, 2_000, 3_000,
        FixtureAnalysis(outcome="person_down", summary="Person prone."),
        "media-event",
    )
    packet = make_packets(tmp_path)[0]
    artifact = exporter.export_keyframe(event_id, packet)
    clip_artifact = exporter.export_clip(
        event_id, make_packets(tmp_path),
        requested_start_utc_ms=9_000,
        requested_end_utc_ms=13_000,
    )
    ready_id = store.add_media_artifact(event_id, artifact)
    clip_id = store.add_media_artifact(event_id, clip_artifact)
    missing_id = store.add_media(event_id)
    assert store.get_media(ready_id)["status"] == MediaStatus.READY
    assert store.get_media(ready_id)["sha256"] == artifact.sha256
    assert store.get_media(missing_id)["status"] == MediaStatus.MISSING
    listed = store.list_events(QueryIntent())
    assert listed[0]["thumbnail_media_id"] == ready_id
    assert listed[0]["has_ready_media"] == 1
    assert listed[0]["has_clip"] == 1
    assert store.list_events(QueryIntent(has_ready_media=True))[0]["event_id"] == event_id
    assert store.list_events(QueryIntent(has_ready_media=False)) == []

    (media_root / artifact.relative_path).unlink()
    (media_root / clip_artifact.relative_path).unlink()
    stale = store.list_events(QueryIntent())[0]
    assert stale["thumbnail_media_id"] is None
    assert stale["has_ready_media"] == 0
    assert stale["has_keyframe"] == 0
    assert stale["has_clip"] == 0
    assert store.list_events(QueryIntent(has_ready_media=True)) == []
    assert store.list_events(QueryIntent(has_ready_media=False))[0]["event_id"] == event_id
    assert store.get_media(ready_id)["status"] == MediaStatus.MISSING
    assert store.get_media(clip_id)["failure_reason"] == (
        "artifact file is missing from local media storage"
    )


def test_database_store_defaults_media_root_next_to_database(tmp_path):
    database = tmp_path / "data" / "hackhive.sqlite"

    store = Store(database)

    assert store.media_root == (tmp_path / "data" / "media").resolve()
