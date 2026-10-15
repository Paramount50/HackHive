from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable

from .contracts import Category


@dataclass(frozen=True)
class LabeledEvent:
    source_id: str
    category: Category
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class PredictedEvent:
    source_id: str
    category: Category
    start_ms: int
    end_ms: int
    alert_ms: int


def temporal_iou(left_start: int, left_end: int, right_start: int, right_end: int) -> float:
    intersection = max(0, min(left_end, right_end) - max(left_start, right_start))
    union = max(left_end, right_end) - min(left_start, right_start)
    return intersection / union if union > 0 else 0.0


def _percentile(values: list[int], percentage: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, ceil(percentage * len(ordered)) - 1)]


def score_events(
    labels: Iterable[LabeledEvent],
    predictions: Iterable[PredictedEvent],
    *,
    evaluated_duration_ms: int,
    minimum_iou: float = 0.1,
) -> dict:
    if evaluated_duration_ms <= 0:
        raise ValueError("evaluated_duration_ms must be positive")
    if not 0 < minimum_iou <= 1:
        raise ValueError("minimum_iou must be in (0, 1]")
    label_list = list(labels)
    prediction_list = list(predictions)
    for event in [*label_list, *prediction_list]:
        if event.end_ms <= event.start_ms:
            raise ValueError("event end_ms must be after start_ms")

    possible: list[tuple[float, int, int]] = []
    for label_index, label in enumerate(label_list):
        for prediction_index, prediction in enumerate(prediction_list):
            if label.source_id != prediction.source_id or label.category != prediction.category:
                continue
            overlap = temporal_iou(label.start_ms, label.end_ms, prediction.start_ms, prediction.end_ms)
            if overlap >= minimum_iou:
                possible.append((overlap, label_index, prediction_index))
    possible.sort(reverse=True)
    matched_labels: set[int] = set()
    matched_predictions: set[int] = set()
    latencies: list[int] = []
    for _, label_index, prediction_index in possible:
        if label_index in matched_labels or prediction_index in matched_predictions:
            continue
        matched_labels.add(label_index)
        matched_predictions.add(prediction_index)
        latencies.append(max(0, prediction_list[prediction_index].alert_ms - label_list[label_index].start_ms))

    per_class = {}
    for category in Category:
        label_indices = {i for i, item in enumerate(label_list) if item.category == category}
        prediction_indices = {i for i, item in enumerate(prediction_list) if item.category == category}
        tp = len(label_indices & matched_labels)
        fp = len(prediction_indices - matched_predictions)
        fn = len(label_indices - matched_labels)
        per_class[str(category)] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        }

    tp = len(matched_labels)
    fp = len(prediction_list) - len(matched_predictions)
    fn = len(label_list) - len(matched_labels)
    return {
        "raw_counts": {"tp": tp, "fp": fp, "fn": fn, "labels": len(label_list), "predictions": len(prediction_list)},
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "false_alerts_per_hour": fp / (evaluated_duration_ms / 3_600_000),
        "alert_latency_ms": {
            "count": len(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
        "per_class": per_class,
        "matching": {"minimum_temporal_iou": minimum_iou, "one_to_one": True},
    }
