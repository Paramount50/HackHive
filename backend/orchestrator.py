from __future__ import annotations

from collections import deque
from typing import Callable

from .contracts import FixtureAnalysis, MediaStatus
from .ingest.replay import FramePacket
from .inference.service import SemanticJob
from .media import MediaExporter
from .perception.action import ActionClassification, aggregate_action_classifications
from .perception.pipeline import PerceptionPipeline
from .perception.rules import Candidate, CandidateCategory
from .storage import Store


SUMMARIES = {
    "person_down": "A tracked person remained horizontally postured for the configured duration.",
    "crowd_gathering": "The configured number of tracked people remained inside the monitored zone.",
    "suspected_tailgating": "Two distinct tracked passages crossed the gate in the same direction close together.",
    "aggression_fight": "Possible physical altercation visible in the replay; operator confirmation is required.",
}


class ActionClassifier:
    def classify(self, image_paths: list[str]) -> ActionClassification:
        raise NotImplementedError


class EventOrchestrator:
    """Replay event path; model candidates remain needs-review until operator review."""

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
        action_classifier: ActionClassifier | None = None,
        action_window_ms: int = 2_000,
        action_stride_ms: int = 1_000,
        semantic_submit: Callable[[SemanticJob], bool] | None = None,
        semantic_pool_frames: int = 16,
    ) -> None:
        self.session_id = session_id
        self.session_started_utc_ms = session_started_utc_ms
        self.pipeline = pipeline
        self.store = store
        self.media = media
        self.frames: deque[FramePacket] = deque(maxlen=context_frames)
        self.replay_frames: list[FramePacket] = []
        self.merge_gap_ms = merge_gap_ms
        self.pre_roll_ms = pre_roll_ms
        self.post_roll_ms = post_roll_ms
        if action_window_ms <= 0 or action_stride_ms <= 0:
            raise ValueError("action window and stride must be positive")
        self.action_classifier = action_classifier
        self.action_window_ms = action_window_ms
        self.action_stride_ms = action_stride_ms
        self.last_action_pts_ms: int | None = None
        self.last_action_frame_indices: list[int] = []
        self.action_predictions: list[ActionClassification] = []
        self.semantic_submit = semantic_submit
        self.semantic_submitted_events: set[str] = set()
        self.pending_semantic_candidates: dict[str, tuple[Candidate, str]] = {}
        if semantic_pool_frames < 4:
            raise ValueError("semantic_pool_frames must be at least four")
        self.semantic_pool_frames = semantic_pool_frames

    def process(self, packet: FramePacket) -> list[str]:
        if packet.session_id != self.session_id:
            raise ValueError("packet belongs to another replay session")
        self.frames.append(packet)
        self.replay_frames.append(packet)
        _, candidates = self.pipeline.process(packet)
        window_prediction = self._classify_window(packet)
        if window_prediction is not None:
            self.action_predictions.append(window_prediction)
        action_prediction = (
            aggregate_action_classifications(self.action_predictions)
            if self.action_predictions else None
        )
        if action_prediction is not None:
            action_category = action_prediction.outcome
            if action_category in {"person_down", "crowd_gathering", "aggression_fight"}:
                if action_category == "aggression_fight":
                    candidates = [
                        candidate for candidate in candidates
                        if candidate.category != CandidateCategory.CROWD_GATHERING
                    ]
                if window_prediction is not None:
                    matching_candidate = next(
                        (candidate for candidate in candidates
                         if candidate.category == CandidateCategory(action_category)),
                        None,
                    )
                    if matching_candidate is not None:
                        candidates.remove(matching_candidate)
                        candidates.append(Candidate(
                            matching_candidate.category,
                            matching_candidate.start_pts_ms,
                            matching_candidate.trigger_pts_ms,
                            matching_candidate.track_ids,
                            f"XCLIP zero-shot action ranking; {matching_candidate.reason}",
                        ))
                    else:
                        candidates.append(Candidate(
                            CandidateCategory(action_category),
                            max(0, packet.source_pts_ms - self.action_window_ms),
                            packet.source_pts_ms,
                            (),
                            "XCLIP zero-shot action ranking",
                        ))
            if window_prediction is not None and action_prediction.outcome in {"none", "unknown"}:
                self._record_action_window(packet, action_prediction)
        touched: list[str] = []
        for candidate in candidates:
            if candidate.trigger_pts_ms != packet.source_pts_ms:
                continue
            category = str(candidate.category)
            is_action = candidate.reason.startswith("XCLIP zero-shot")
            identity = "temporal-action" if is_action else (
                "default-zone" if category == "crowd_gathering"
                else ",".join(sorted(candidate.track_ids))
            )
            if not identity:
                identity = "camera-window"
            candidate_key = f"{self.session_id}:{category}:{identity}"
            if is_action:
                candidate_key = f"{self.session_id}:temporal-action"
            analysis = FixtureAnalysis(
                outcome=category,
                summary=(
                    (
                        f"Possible {category.replace('_', ' ')} flagged by replay-level temporal action ranking "
                        f"(uncalibrated; operator review required; peak score "
                        f"{action_prediction.normalized_scores[category]:.3f} across "
                        f"{action_prediction.window_count} action window(s))."
                    ) if is_action and action_prediction else SUMMARIES[category]
                ),
                observations=[{
                    "predicate": "xclip_class_scores" if is_action else "cpu_candidate_rule",
                    "supported": True,
                    "frame_indices": self.last_action_frame_indices if is_action else [packet.frame_id],
                    **({
                        "details": {
                            "top_category": action_prediction.top_category,
                            "class_scores": action_prediction.class_scores,
                            "normalized_scores": action_prediction.normalized_scores,
                            "margin": action_prediction.margin,
                            "latency_ms": action_prediction.latency_ms,
                            "aggregation_method": action_prediction.aggregation_method,
                            "window_count": action_prediction.window_count,
                            "calibrated": False,
                        }
                    } if is_action and action_prediction else {}),
                }],
                limitations=[
                    (
                        "XCLIP zero-shot action ranking is uncalibrated for this camera and has not been "
                        "human-confirmed."
                        if is_action else
                        "This incident was created by CPU candidate rules and has not been confirmed by a VLM."
                    )
                ],
                frame_indices=self.last_action_frame_indices if is_action else [packet.frame_id],
                model_id="xclip://microsoft/xclip-base-patch32" if is_action else "candidate://cpu-rules",
                model_revision="a2e27a78a2b5d802e894b8a1ef14f3a8ce490963" if is_action else "candidate-v1",
                prompt_version="cctv-actions-v1" if is_action else "none",
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
                self.pending_semantic_candidates[event_id] = (candidate, candidate_key)
                has_keyframe = self.store.connection.execute(
                    "SELECT 1 FROM media WHERE event_id=? AND kind='keyframe'", (event_id,)
                ).fetchone()
                if not has_keyframe:
                    self.store.add_media_artifact(event_id, self.media.export_keyframe(event_id, packet))

        closed = self.store.close_inactive_events(
            packet.captured_utc_ms, self.merge_gap_ms, session_id=self.session_id
        )
        self._export_closed(closed)
        return touched

    def _submit_final_verifications(self, last_packet: FramePacket) -> None:
        for event_id, (candidate, candidate_key) in self.pending_semantic_candidates.items():
            if event_id in self.semantic_submitted_events:
                continue
            event = self.store.get_event_row(event_id)
            if event is None:
                continue
            event_start_pts_ms = int(
                event["start_pts_ms"]
                if event["start_pts_ms"] is not None else candidate.start_pts_ms
            )
            event_end_pts_ms = int(
                event["end_pts_ms"]
                if event["end_pts_ms"] is not None else last_packet.source_pts_ms
            )
            verification_frames = [
                frame for frame in self.replay_frames
                if max(0, event_start_pts_ms - 2_000)
                <= frame.source_pts_ms
                <= event_end_pts_ms + 2_000
            ]
            final_candidate = Candidate(
                CandidateCategory(event["category"]),
                event_start_pts_ms,
                event_end_pts_ms,
                candidate.track_ids,
                "full-incident temporal verification",
            )
            self._submit_semantic_verification(
                event_id,
                final_candidate,
                candidate_key,
                frames=verification_frames,
            )

    def _submit_semantic_verification(
        self,
        event_id: str,
        candidate: Candidate,
        candidate_key: str,
        *,
        frames: list[FramePacket] | None = None,
    ) -> None:
        if event_id in self.semantic_submitted_events:
            return
        if str(candidate.category) not in {
            "person_down", "crowd_gathering", "suspected_tailgating", "aggression_fight"
        }:
            return
        available = frames if frames is not None else [
            frame for frame in self.frames
            if max(0, candidate.start_pts_ms - 2_000) <= frame.source_pts_ms <= candidate.trigger_pts_ms
        ]
        if len(available) > self.semantic_pool_frames:
            pool_indices = [
                round(index * (len(available) - 1) / (self.semantic_pool_frames - 1))
                for index in range(self.semantic_pool_frames)
            ]
            available = [available[index] for index in pool_indices]
        source_pts_ms = tuple(frame.source_pts_ms for frame in available)
        if not source_pts_ms:
            source_pts_ms = (candidate.start_pts_ms, candidate.trigger_pts_ms)
        job = SemanticJob(
            event_id=event_id,
            camera_id=available[-1].camera_id,
            candidate_key=candidate_key,
            expected_outcome=str(candidate.category),
            image_paths=tuple(frame.asset_id for frame in available),
            source_pts_ms=source_pts_ms,
        )
        if self.semantic_submit is None:
            self.store.record_vlm_result(job, {
                "status": "not_configured",
                "error_code": "vlm_not_configured",
                "error": "No local VLM service is configured.",
            })
            self.semantic_submitted_events.add(event_id)
            return
        if len(available) < 4:
            self.store.record_vlm_result(job, {
                "status": "rejected",
                "error_code": "insufficient_candidate_frames",
                "error": "At least four candidate frames are required.",
            })
            self.semantic_submitted_events.add(event_id)
            return
        if self.semantic_submit(job):
            self.semantic_submitted_events.add(event_id)

    def _classify_window(self, packet: FramePacket) -> ActionClassification | None:
        if self.action_classifier is None:
            return None
        if self.last_action_pts_ms is not None and packet.source_pts_ms - self.last_action_pts_ms < self.action_stride_ms:
            return None
        window_start = packet.source_pts_ms - self.action_window_ms
        window = [frame for frame in self.frames if window_start <= frame.source_pts_ms <= packet.source_pts_ms]
        if len(window) < 8 or window[-1].source_pts_ms - window[0].source_pts_ms < self.action_window_ms * 0.75:
            return None
        targets = [
            window_start + round(index * self.action_window_ms / 7)
            for index in range(8)
        ]
        selected = []
        for target in targets:
            frame = min(window, key=lambda item: abs(item.source_pts_ms - target))
            if not selected or frame.frame_id != selected[-1].frame_id:
                selected.append(frame)
        if len(selected) != 8:
            return None
        self.last_action_frame_indices = [frame.frame_id for frame in selected]
        try:
            result = self.action_classifier.classify([frame.asset_id for frame in selected])
        except Exception as exc:
            self.store.record_analysis_failure(
                self.session_id,
                self.session_started_utc_ms + selected[0].source_pts_ms,
                self.session_started_utc_ms + selected[-1].source_pts_ms,
                reason="xclip_action",
                model_id="xclip://microsoft/xclip-base-patch32",
                model_revision="a2e27a78a2b5d802e894b8a1ef14f3a8ce490963",
                prompt_version="cctv-actions-v1",
                error_code="action_inference_failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            self.last_action_pts_ms = packet.source_pts_ms
            return None
        self.last_action_pts_ms = packet.source_pts_ms
        return result

    def _record_action_window(self, packet: FramePacket, prediction: ActionClassification) -> None:
        category = prediction.outcome
        analysis = FixtureAnalysis(
            outcome=category,
            summary=(
                f"Top zero-shot action label: {prediction.top_category}; "
                f"normalized score {prediction.normalized_scores[prediction.top_category]:.3f}, "
                f"margin {prediction.margin:.3f}."
            ),
            observations=[{
                "predicate": "xclip_class_scores",
                "supported": category not in {"none", "unknown"},
                "frame_indices": self.last_action_frame_indices,
                "details": {
                    "top_category": prediction.top_category,
                    "class_scores": prediction.class_scores,
                    "normalized_scores": prediction.normalized_scores,
                    "margin": prediction.margin,
                    "latency_ms": prediction.latency_ms,
                    "calibrated": False,
                },
            }],
            limitations=[
                "Zero-shot action scores are not calibrated and must not be treated as verified incident truth."
            ],
            frame_indices=self.last_action_frame_indices,
            model_id="xclip://microsoft/xclip-base-patch32",
            model_revision="a2e27a78a2b5d802e894b8a1ef14f3a8ce490963",
            prompt_version="cctv-actions-v1",
        )
        self.store.record_analysis(
            self.session_id,
            self.session_started_utc_ms + max(0, packet.source_pts_ms - self.action_window_ms),
            self.session_started_utc_ms + packet.source_pts_ms,
            analysis,
            f"{self.session_id}:xclip-window:{packet.frame_id}",
            start_pts_ms=max(0, packet.source_pts_ms - self.action_window_ms),
        )

    def finish(self, last_packet: FramePacket | None) -> None:
        if last_packet is None:
            return
        closed = self.store.close_inactive_events(
            last_packet.captured_utc_ms + self.merge_gap_ms + 1,
            self.merge_gap_ms,
            session_id=self.session_id,
        )
        self._export_closed(closed)
        self._submit_final_verifications(last_packet)

    def _export_closed(self, closed: list[str]) -> None:
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
