from __future__ import annotations

from math import comb, factorial

import numpy as np


def _backward_coefficients(order: int) -> np.ndarray:
    return np.asarray([
        sum(((-1) ** offset) * comb(term, offset) / factorial(term)
            for term in range(offset, order + 1))
        for offset in range(order + 1)
    ], dtype=np.float64)


def _forward_coefficients(order: int) -> np.ndarray:
    matrix = np.asarray([
        [(1 + offset) ** power for offset in range(order + 1)]
        for power in range(order + 1)
    ], dtype=np.float64)
    target = np.zeros(order + 1, dtype=np.float64)
    target[0] = 1.0
    return np.linalg.solve(matrix, target)


def taylor_residuals(frame_features: np.ndarray, order: int = 3) -> np.ndarray:
    """Return the relative Taylor prediction residual for every chronological frame."""
    features = np.asarray(frame_features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("frame_features must have shape [frames, dimensions]")
    frame_count = features.shape[0]
    if not 1 <= order < frame_count:
        raise ValueError("Taylor order must be positive and smaller than the frame count")
    if not np.isfinite(features).all():
        raise ValueError("frame_features must contain only finite values")
    norms = np.linalg.norm(features, axis=1)
    residuals = np.zeros(frame_count, dtype=np.float64)
    for current_order in range(order + 1):
        start = current_order + 1
        if start >= frame_count:
            break
        coefficients = _backward_coefficients(current_order)
        prediction = sum(
            coefficients[offset] * features[start - 1 - offset:frame_count - 1 - offset]
            for offset in range(current_order + 1)
        )
        relative = np.linalg.norm(features[start:] - prediction, axis=1) / (norms[start:] + 1e-8)
        if current_order < order:
            residuals[start] = relative[0]
        else:
            residuals[start:] = relative
    forward = _forward_coefficients(order)
    for index in range(min(order, frame_count - order - 1)):
        prediction = sum(
            forward[offset] * features[index + 1 + offset]
            for offset in range(order + 1)
        )
        residuals[index] = np.linalg.norm(features[index] - prediction) / (norms[index] + 1e-8)
    return residuals


def select_temporal_surprises(
    frame_features: np.ndarray, target_frames: int = 4, order: int = 3
) -> list[int]:
    """Select chronological local Taylor-residual maxima, then fill by residual rank."""
    features = np.asarray(frame_features)
    if features.ndim != 2:
        raise ValueError("frame_features must have shape [frames, dimensions]")
    frame_count = features.shape[0]
    if not 1 <= target_frames <= frame_count:
        raise ValueError("target_frames must be between one and the frame count")
    if frame_count <= target_frames:
        return list(range(frame_count))
    residuals = taylor_residuals(features, order=order)
    peaks = [
        index for index in range(1, frame_count - 1)
        if residuals[index] > residuals[index - 1] and residuals[index] >= residuals[index + 1]
    ]
    ranked_peaks = sorted(peaks, key=lambda index: (-residuals[index], index))
    selected = ranked_peaks[:target_frames]
    if len(selected) < target_frames:
        remainder = sorted(
            (index for index in range(frame_count) if index not in selected),
            key=lambda index: (-residuals[index], index),
        )
        selected.extend(remainder[:target_frames - len(selected)])
    return sorted(selected)
