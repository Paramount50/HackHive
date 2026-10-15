from pathlib import Path

import cv2
import numpy as np

from backend.contracts import QueryIntent
from backend.ingest.replay import FramePacket
from backend.media import MediaExporter
from backend.orchestrator import EventOrchestrator
from backend.perception.action import ActionClassification, PROMPTS, classify_logits
from backend.perception.pipeline import PerceptionPipeline
from backend.storage import Store


def test_classify_logits_averages_prompts_and_abstains_on_ties() -> None:
    fight_scores = [4.0] * len(PROMPTS["aggression_fight"])
    other_scores = [0.0] * sum(len(prompts) for name, prompts in PROMPTS.items()
                               if name != "aggression_fight")

    result = classify_logits(fight_scores + other_scores)

    assert result.outcome == "aggression_fight"
    assert result.top_category == "aggression_fight"
    assert result.normalized_scores["aggression_fight"] > 0.35
    assert result.class_scores["aggression_fight"] == 4.0
    assert classify_logits([0.0] * sum(map(len, PROMPTS.values()))).outcome == "unknown"


def test_classify_logits_rejects_invalid_model_output() -> None:
    expected = sum(map(len, PROMPTS.values()))
    try:
        classify_logits([0.0] * (expected - 1))
    except ValueError as exc:
        assert "finite prompt scores" in str(exc)
    else:
        raise AssertionError("wrong-sized output was accepted")


class EmptyDetector:
    def detect(self, image_path):
        return []


class FixedActionClassifier:
    def __init__(self, outcome: str) -> None:
        self.outcome = outcome
        self.calls: list[list[str]] = []

    def classify(self, image_paths: list[str]) -> ActionClassification:
        self.calls.append(image_paths)
        return ActionClassification(
            self.outcome,
            "aggression_fight" if self.outcome == "unknown" else self.outcome,
            {"aggression_fight": 0.70, "person_down": 0.10, "crowd_gathering": 0.10, "none": 0.10},
            {"aggression_fight": 3.0, "person_down": 0.0, "crowd_gathering": 0.0, "none": 0.0},
            0.60,
        )


def make_session(store: Store, session_id: str, started: int) -> None:
    store.seed_camera()
    store.connection.execute(
        """INSERT INTO stream_sessions
        (session_id,camera_id,mode,started_utc_ms,policy_version,policy_json)
        VALUES (?, 'cam-1', 'replay', ?, 'cam-1-v1', '{}')""",
        (session_id, started),
    )
    store.connection.commit()


def test_xclip_window_creates_reviewable_action_event_with_model_scores(tmp_path: Path) -> None:
    store = Store()
    make_session(store, "action-session", 30_000)
    classifier = FixedActionClassifier("aggression_fight")
    packets = []
    for index in range(11):
        path = tmp_path / f"action-{index}.jpg"
        assert cv2.imwrite(str(path), np.full((48, 64, 3), index * 10, dtype=np.uint8))
        pts = index * 200
        packets.append(FramePacket(
            "cam-1", "action-session", index, pts, 30_000 + pts, 64, 48, str(path), False
        ))
    orchestrator = EventOrchestrator(
        session_id="action-session",
        session_started_utc_ms=30_000,
        pipeline=PerceptionPipeline(EmptyDetector()),
        store=store,
        media=MediaExporter(tmp_path / "media"),
        action_classifier=classifier,
    )

    touched = []
    for packet in packets:
        touched.extend(orchestrator.process(packet))
    orchestrator.finish(packets[-1])

    assert len(classifier.calls) == 1
    assert len(classifier.calls[0]) == 8
    assert len(set(touched)) == 1
    events = store.list_events(QueryIntent())
    assert len(events) == 1
    event = events[0]
    assert event["category"] == "aggression_fight"
    assert event["decision_state"] == "needs_review"
    assert event["lifecycle_state"] == "ended"
    provenance = store.connection.execute(
        """SELECT model_id,model_revision,prompt_version FROM analysis_windows
        WHERE session_id='action-session'"""
    ).fetchone()
    assert tuple(provenance) == (
        "xclip://microsoft/xclip-base-patch32",
        "a2e27a78a2b5d802e894b8a1ef14f3a8ce490963",
        "cctv-actions-v1",
    )
    action_predicate = next(item for item in event["predicates"] if item["name"] == "xclip_class_scores")
    assert action_predicate["details"]["calibrated"] is False
    assert action_predicate["details"]["class_scores"]["aggression_fight"] == 3.0
    assert len(action_predicate["frame_indices"]) == 8
    assert action_predicate["frame_indices"] == sorted(action_predicate["frame_indices"])


def test_unknown_action_windows_are_logged_without_creating_incidents(tmp_path: Path) -> None:
    store = Store()
    make_session(store, "unknown-session", 40_000)
    classifier = FixedActionClassifier("unknown")
    packets = []
    for index in range(11):
        path = tmp_path / f"unknown-{index}.jpg"
        assert cv2.imwrite(str(path), np.zeros((48, 64, 3), dtype=np.uint8))
        pts = index * 200
        packets.append(FramePacket(
            "cam-1", "unknown-session", index, pts, 40_000 + pts, 64, 48, str(path), False
        ))
    orchestrator = EventOrchestrator(
        session_id="unknown-session",
        session_started_utc_ms=40_000,
        pipeline=PerceptionPipeline(EmptyDetector()),
        store=store,
        media=MediaExporter(tmp_path / "unknown-media"),
        action_classifier=classifier,
    )

    for packet in packets:
        assert orchestrator.process(packet) == []

    assert store.list_events(QueryIntent()) == []
    windows = store.connection.execute(
        "SELECT model_outcome,observations_json FROM analysis_windows WHERE session_id='unknown-session'"
    ).fetchall()
    assert len(windows) == len(classifier.calls) == 1
    assert windows[0]["model_outcome"] == "unknown"
    assert "class_scores" in windows[0]["observations_json"]
