from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..ingest.replay import FramePacket
from .rules import Candidate, CandidateRules, Detection


@dataclass(frozen=True)
class BoxDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_name: str

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class Detector(Protocol):
    def detect(self, image_path: str | Path) -> list[BoxDetection]: ...


class YoloDetector:
    """Optional Ultralytics adapter; model installation and weights are a GPU-owner handoff."""

    def __init__(self, weights: str | Path, *, device: str = "cpu", image_size: int = 416) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("YOLO requires the optional ultralytics package") from exc
        self._model = YOLO(str(weights))
        self.device = device
        self.image_size = image_size

    def detect(self, image_path: str | Path) -> list[BoxDetection]:
        results = self._model.predict(
            source=str(image_path), device=self.device, imgsz=self.image_size, verbose=False
        )
        detections: list[BoxDetection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls.item())
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                detections.append(BoxDetection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    score=float(box.conf.item()), class_name=str(names[class_id]),
                ))
        return detections


@dataclass
class _Track:
    track_id: str
    box: BoxDetection
    missed: int = 0
    side: str | None = None


def _iou(left: BoxDetection, right: BoxDetection) -> float:
    x1, y1 = max(left.x1, right.x1), max(left.y1, right.y1)
    x2, y2 = min(left.x2, right.x2), min(left.y2, right.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
    right_area = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


class IoUTracker:
    """Small CPU tracker for integration and fallback; not a ByteTrack accuracy substitute."""

    def __init__(self, *, iou_threshold: float = 0.1, max_missed: int = 2) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._tracks: dict[str, _Track] = {}
        self._next_id = 1

    def update(self, boxes: list[BoxDetection]) -> list[tuple[_Track, BoxDetection]]:
        unmatched_tracks = set(self._tracks)
        matches: list[tuple[_Track, BoxDetection]] = []
        for box in sorted(boxes, key=lambda item: item.score, reverse=True):
            candidates = [
                (track_id, _iou(self._tracks[track_id].box, box))
                for track_id in unmatched_tracks
                if self._tracks[track_id].box.class_name == box.class_name
            ]
            track_id, overlap = max(candidates, key=lambda item: item[1], default=(None, 0.0))
            if track_id is None or overlap < self.iou_threshold:
                track_id = f"track-{self._next_id}"
                self._next_id += 1
                track = _Track(track_id, box)
                self._tracks[track_id] = track
            else:
                track = self._tracks[track_id]
                unmatched_tracks.remove(track_id)
                track.box = box
                track.missed = 0
            matches.append((track, box))
        for track_id in unmatched_tracks:
            self._tracks[track_id].missed += 1
            if self._tracks[track_id].missed > self.max_missed:
                del self._tracks[track_id]
        return matches


@dataclass(frozen=True)
class ScenePolicy:
    gate_line_y: float = 0.5
    zone: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)


class PerceptionPipeline:
    def __init__(
        self,
        detector: Detector,
        *,
        tracker: IoUTracker | None = None,
        rules: CandidateRules | None = None,
        policy: ScenePolicy | None = None,
        history_frames: int = 60,
    ) -> None:
        self.detector = detector
        self.tracker = tracker or IoUTracker()
        self.rules = rules or CandidateRules()
        self.policy = policy or ScenePolicy()
        self.history: deque[list[Detection]] = deque(maxlen=history_frames)

    def process(self, packet: FramePacket) -> tuple[list[Detection], list[Candidate]]:
        boxes = self.detector.detect(packet.asset_id)
        detections: list[Detection] = []
        for track, box in self.tracker.update(boxes):
            center_x_px, center_y_px = box.center
            center_x = center_x_px / packet.width
            center_y = center_y_px / packet.height
            side = "inside" if center_y >= self.policy.gate_line_y else "outside"
            direction = None
            if track.side == "outside" and side == "inside":
                direction = "entry"
            elif track.side == "inside" and side == "outside":
                direction = "exit"
            track.side = side
            zone_x1, zone_y1, zone_x2, zone_y2 = self.policy.zone
            detections.append(Detection(
                frame_id=packet.frame_id,
                source_pts_ms=packet.source_pts_ms,
                track_id=track.track_id,
                class_name=box.class_name,
                score=box.score,
                center_x=center_x,
                center_y=center_y,
                width=(box.x2 - box.x1) / packet.width,
                height=(box.y2 - box.y1) / packet.height,
                in_zone=zone_x1 <= center_x <= zone_x2 and zone_y1 <= center_y <= zone_y2,
                side=side,
                direction=direction,
            ))
        self.history.append(detections)
        return detections, self.rules.evaluate(list(self.history))
