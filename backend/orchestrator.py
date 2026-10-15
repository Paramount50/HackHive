from __future__ import annotations

from collections import deque

from .contracts import FixtureAnalysis, MediaStatus
from .ingest.replay import FramePacket
from .media import MediaExporter
from .perception.pipeline import PerceptionPipeline
from .storage import Store


SUMMARIES = {
    "person_down": "A tracked person remained horizontally postured for the configured duration.",
    "crowd_gathering": "The configured number of tracked people remained inside the monitored zone.",
    "suspected_tailgating": "Two distinct tracked passages crossed the gate in the same direction close together.",
}


class EventOrchestrator:
    """CPU event path. Candidate output remains needs-review until VLM confirmation."""

    def __init__(
        self,
        *,
        session_id: str,
        session_started_utc_ms: int,
        pipeline: PerceptionPipeline,
        store: Store,
        media: MediaExporter,
        context_frames: int = 300,
        merge_gap_ms: int = 5_000,
        pre_roll_ms: int = 10_000,
        post_roll_ms: int = 10_000,
    ) -> None:
        self.session_id = session_id
        self.session_started_utc_ms = session_started_utc_ms
        self.pipeline = pipeline
        self.store = store
        self.media = media
        self.frames: deque[FramePacket] = deque(maxlen=context_frames)
        self.merge_gap_ms = merge_gap_ms
        self.pre_roll_ms = pre_roll_ms
        self.post_roll_ms = post_roll_ms

    def process(self, packet: FramePacket) -> list[str]:
        if packet.session_id != self.session_id:
            raise ValueError("packet belongs to another replay session")
        self.frames.append(packet)
        _, candidates = self.pipeline.process(packet)
        touched: list[str] = []
        for candidate in candidates:
            if candidate.trigger_pts_ms != packet.source_pts_ms:
                continue
            category = str(candidate.category)
            candidate_key = f"{self.session_id}:{category}:{','.join(sorted(candidate.track_ids))}"
            analysis = FixtureAnalysis(
                outcome=category,
                summary=SUMMARIES[category],
                observations=[{
                    "predicate": "cpu_candidate_rule",
                    "supported": True,
                    "frame_indices": [packet.frame_id],
                }],
                limitations=[
                    "This incident was created by CPU candidate rules and has not been confirmed by a VLM."
                ],
                frame_indices=[packet.frame_id],
                model_id="candidate://cpu-rules",
                model_revision="candidate-v1",
                prompt_version="none",
            )
            event_id = self.store.record_analysis(
                self.session_id,
                self.session_started_utc_ms + candidate.start_pts_ms,
                self.session_started_utc_ms + candidate.trigger_pts_ms,
                analysis,
                candidate_key,
                start_pts_ms=candidate.start_pts_ms,
            )
            if event_id:
                touched.append(event_id)
                has_keyframe = self.store.connection.execute(
                    "SELECT 1 FROM media WHERE event_id=? AND kind='keyframe'", (event_id,)
                ).fetchone()
                if not has_keyframe:
                    self.store.add_media_artifact(event_id, self.media.export_keyframe(event_id, packet))

        closed = self.store.close_inactive_events(
            packet.captured_utc_ms, self.merge_gap_ms, session_id=self.session_id
        )
        for event_id in closed:
            has_clip = self.store.connection.execute(
                "SELECT 1 FROM media WHERE event_id=? AND kind='event_clip'", (event_id,)
            ).fetchone()
            event = self.store.get_event_row(event_id)
            if has_clip or event is None:
                continue
            requested_start = event["start_utc_ms"] - self.pre_roll_ms
            requested_end = event["end_utc_ms"] + self.post_roll_ms
            try:
                artifact = self.media.export_clip(
                    event_id, list(self.frames),
                    requested_start_utc_ms=requested_start,
                    requested_end_utc_ms=requested_end,
                )
            except (ValueError, RuntimeError):
                self.store.add_media(event_id, MediaStatus.FAILED, kind="event_clip")
            else:
                self.store.add_media_artifact(event_id, artifact)
        return touched
