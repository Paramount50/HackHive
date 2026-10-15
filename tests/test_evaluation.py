from backend.contracts import Category
from backend.evaluation import LabeledEvent, PredictedEvent, score_events, temporal_iou


def test_temporal_iou_uses_half_open_intervals() -> None:
    assert temporal_iou(0, 1_000, 1_000, 2_000) == 0
    assert temporal_iou(0, 1_000, 500, 1_500) == 1 / 3


def test_event_score_is_one_to_one_and_reports_raw_counts() -> None:
    labels = [
        LabeledEvent("video-1", Category.PERSON_DOWN, 1_000, 5_000),
        LabeledEvent("video-1", Category.SUSPECTED_TAILGATING, 10_000, 12_000),
    ]
    predictions = [
        PredictedEvent("video-1", Category.PERSON_DOWN, 2_000, 5_000, 3_000),
        PredictedEvent("video-1", Category.PERSON_DOWN, 2_500, 4_500, 3_500),
        PredictedEvent("video-1", Category.CROWD_GATHERING, 20_000, 24_000, 21_000),
    ]
    metrics = score_events(labels, predictions, evaluated_duration_ms=3_600_000)
    assert metrics["raw_counts"] == {"tp": 1, "fp": 2, "fn": 1, "labels": 2, "predictions": 3}
    assert metrics["precision"] == 1 / 3
    assert metrics["recall"] == 1 / 2
    assert metrics["false_alerts_per_hour"] == 2
    assert metrics["alert_latency_ms"] == {"count": 1, "p50": 2_000, "p95": 2_000}
    assert metrics["per_class"]["person_down"]["tp"] == 1
    assert metrics["per_class"]["person_down"]["fp"] == 1


def test_score_rejects_invalid_duration() -> None:
    try:
        score_events([], [], evaluated_duration_ms=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("zero evaluation duration must be rejected")
