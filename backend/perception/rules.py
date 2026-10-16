from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CandidateCategory(StrEnum):
    PERSON_DOWN = "person_down"
    CROWD_GATHERING = "crowd_gathering"
    SUSPECTED_TAILGATING = "suspected_tailgating"
    AGGRESSION_FIGHT = "aggression_fight"
    FIRE_SMOKE = "fire_smoke"
    INTRUSION_UNAUTHORIZED_ACCESS = "intrusion_unauthorized_access"


@dataclass(frozen=True)
class Detection:
    frame_id: int
    source_pts_ms: int
    track_id: str
    class_name: str
    score: float
    center_x: float
    center_y: float
    width: float
    height: float
    in_zone: bool = False
    side: str | None = None
    direction: str | None = None
    pixel_aspect_ratio: float | None = None

    @property
    def horizontal_posture(self) -> bool:
        ratio = self.pixel_aspect_ratio
        if ratio is None:
            ratio = self.width / self.height if self.height > 0 else 0
        return ratio >= 1.15


@dataclass(frozen=True)
class Candidate:
    category: CandidateCategory
    start_pts_ms: int
    trigger_pts_ms: int
    track_ids: tuple[str, ...]
    reason: str


class CandidateRules:
    def __init__(
        self,
        *,
        person_down_persistence_ms: int = 2_000,
        crowd_minimum_people: int = 4,
        crowd_duration_ms: int = 3_000,
        tailgating_window_ms: int = 2_000,
        max_observation_gap_ms: int = 1_500,
    ) -> None:
        self.person_down_persistence_ms = person_down_persistence_ms
        self.crowd_minimum_people = crowd_minimum_people
        self.crowd_duration_ms = crowd_duration_ms
        self.tailgating_window_ms = tailgating_window_ms
        self.max_observation_gap_ms = max_observation_gap_ms

    def evaluate(self, frames: list[list[Detection]]) -> list[Candidate]:
        nonempty = [frame for frame in frames if frame]
        if any(
            right[0].source_pts_ms < left[0].source_pts_ms
            for left, right in zip(nonempty, nonempty[1:])
        ):
            raise ValueError("detection frames must be ordered by source PTS")
        candidates: list[Candidate] = []
        candidates.extend(self._person_down(frames))
        candidates.extend(self._crowd(frames))
        candidates.extend(self._tailgating(frames))
        return candidates

    def _person_down(self, frames: list[list[Detection]]) -> list[Candidate]:
        by_track: dict[str, list[Detection]] = {}
        for frame in frames:
            if not frame:
                # An empty detection frame has no timestamp in this boundary.
                # Preserve runs until a later timestamped observation can
                # distinguish a tolerated miss from a long gap.
                continue
            frame_pts_ms = frame[0].source_pts_ms
            observed_people = {
                detection.track_id: detection
                for detection in frame
                if detection.class_name == "person"
            }
            for track_id, observations in list(by_track.items()):
                detection = observed_people.get(track_id)
                if detection is not None and not detection.horizontal_posture:
                    del by_track[track_id]
                elif (
                    detection is None
                    and frame_pts_ms - observations[-1].source_pts_ms > self.max_observation_gap_ms
                ):
                    del by_track[track_id]
            for detection in observed_people.values():
                if not detection.horizontal_posture:
                    continue
                previous = by_track.get(detection.track_id)
                if (
                    previous
                    and detection.source_pts_ms - previous[-1].source_pts_ms
                    > self.max_observation_gap_ms
                ):
                    del by_track[detection.track_id]
                by_track.setdefault(detection.track_id, []).append(detection)
        result = []
        for track_id, observations in by_track.items():
            if observations[-1].source_pts_ms - observations[0].source_pts_ms >= self.person_down_persistence_ms:
                result.append(Candidate(
                    CandidateCategory.PERSON_DOWN, observations[0].source_pts_ms,
                    observations[-1].source_pts_ms, (track_id,),
                    "person remains horizontally postured for the configured persistence",
                ))
        return result

    def _crowd(self, frames: list[list[Detection]]) -> list[Candidate]:
        qualifying = []
        for frame in frames:
            tracks = tuple(sorted({d.track_id for d in frame if d.class_name == "person" and d.in_zone}))
            if len(tracks) < self.crowd_minimum_people:
                qualifying = []
                continue
            pts = frame[0].source_pts_ms
            if qualifying and pts - qualifying[-1][0] > self.max_observation_gap_ms:
                qualifying = []
            qualifying.append((pts, tracks))
        if not qualifying or qualifying[-1][0] - qualifying[0][0] < self.crowd_duration_ms:
            return []
        return [Candidate(
            CandidateCategory.CROWD_GATHERING, qualifying[0][0], qualifying[-1][0], qualifying[-1][1],
            "configured person count remains in the marked zone for the configured duration",
        )]

    def _tailgating(self, frames: list[list[Detection]]) -> list[Candidate]:
        crossings = [
            detection for frame in frames for detection in frame
            if detection.class_name == "person" and detection.direction == "entry"
        ]
        crossings.sort(key=lambda item: item.source_pts_ms)
        result = []
        for first, second in zip(crossings, crossings[1:]):
            if first.track_id != second.track_id and second.source_pts_ms - first.source_pts_ms <= self.tailgating_window_ms:
                result.append(Candidate(
                    CandidateCategory.SUSPECTED_TAILGATING, first.source_pts_ms, second.source_pts_ms,
                    (first.track_id, second.track_id),
                    "two distinct same-direction crossings occur within the configured window",
                ))
        return result
