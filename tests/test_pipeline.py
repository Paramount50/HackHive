from pathlib import Path

from backend.ingest.replay import FramePacket
from backend.perception.pipeline import BoxDetection, PerceptionPipeline
from backend.perception.rules import CandidateCategory, CandidateRules


class ScriptedDetector:
    def __init__(self, frames: list[list[BoxDetection]]) -> None:
        self.frames = iter(frames)

    def detect(self, image_path: str | Path) -> list[BoxDetection]:
        return next(self.frames)


def packet(frame_id: int, pts: int) -> FramePacket:
    return FramePacket("cam-1", "session-1", frame_id, pts, 10_000 + pts, 100, 100, f"frame-{frame_id}.jpg", False)


def person(x1: float, y1: float, x2: float, y2: float) -> BoxDetection:
    return BoxDetection(x1, y1, x2, y2, 0.9, "person")


def test_iou_tracker_preserves_id_and_emits_gate_crossing_once() -> None:
    pipeline = PerceptionPipeline(ScriptedDetector([
        [person(40, 20, 60, 40)],
        [person(40, 35, 60, 55)],
        [person(40, 50, 60, 70)],
        [person(40, 55, 60, 75)],
    ]))
    observations = [pipeline.process(packet(index, index * 500))[0][0] for index in range(4)]
    assert len({item.track_id for item in observations}) == 1
    assert [item.direction for item in observations] == [None, None, "entry", None]


def test_horizontal_person_persistence_produces_candidate() -> None:
    horizontal = person(10, 40, 70, 70)
    pipeline = PerceptionPipeline(
        ScriptedDetector([[horizontal], [horizontal], [horizontal]]),
        rules=CandidateRules(person_down_persistence_ms=2_000),
    )
    candidates = []
    for index, pts in enumerate((0, 1_000, 2_000)):
        _, candidates = pipeline.process(packet(index, pts))
    person_down = [item for item in candidates if item.category == CandidateCategory.PERSON_DOWN]
    assert len(person_down) == 1
    assert person_down[0].start_pts_ms == 0
    assert person_down[0].trigger_pts_ms == 2_000


def test_track_expires_after_configured_misses() -> None:
    pipeline = PerceptionPipeline(ScriptedDetector([
        [person(10, 10, 30, 40)], [], [], [], [person(10, 10, 30, 40)]
    ]))
    first = pipeline.process(packet(0, 0))[0][0].track_id
    for index in range(1, 4):
        pipeline.process(packet(index, index * 500))
    later = pipeline.process(packet(4, 2_000))[0][0].track_id
    assert later != first


def test_posture_uses_pixel_aspect_ratio_on_widescreen_source():
    pipeline = PerceptionPipeline(ScriptedDetector([[person(10, 10, 130, 110)]]))
    frame = FramePacket("cam-1", "session-1", 0, 0, 10000, 1920, 1080, "frame.jpg", False)
    detections, _ = pipeline.process(frame)
    assert detections[0].horizontal_posture  # 120/100, not (120/1920)/(100/1080)
