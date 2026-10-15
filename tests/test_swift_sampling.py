import numpy as np
import pytest

from backend.inference.swift_sampling import select_temporal_surprises, taylor_residuals


def test_swift_sampling_keeps_abrupt_temporal_surprise() -> None:
    features = np.asarray([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 5.0],
        [0.0, 5.0],
        [0.0, 5.0],
        [0.0, 5.0],
    ])
    residuals = taylor_residuals(features, order=3)
    selected = select_temporal_surprises(features, target_frames=4, order=3)

    assert residuals[5] > residuals[3]
    assert 5 in selected
    assert selected == sorted(selected)
    assert len(selected) == len(set(selected)) == 4


def test_swift_sampling_validates_budget_and_features() -> None:
    features = np.ones((5, 3))
    assert select_temporal_surprises(features, target_frames=5) == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="target_frames"):
        select_temporal_surprises(features, target_frames=6)
    with pytest.raises(ValueError, match="finite"):
        taylor_residuals(np.asarray([[1.0], [2.0], [np.nan], [4.0]]))
