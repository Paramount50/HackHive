import cv2
import numpy as np
import pytest

from backend.inference.smolvlm import SmolVLMWorker
from scripts.smoke_vlm import sample_window


def test_sampling_four_distinct_frames_preserves_source_order(tmp_path):
    path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    for index in range(60):
        writer.write(np.full((48, 64, 3), index * 3, dtype=np.uint8))
    writer.release()
    images, pts, indices = sample_window(path, .5)
    assert len(images) == len(set(indices)) == 4
    assert all(left < right for left, right in zip(pts, pts[1:]))
    assert pts[-1] - pts[0] == 4000
    assert images[0].getpixel((10, 10))[0] < images[-1].getpixel((10, 10))[0]


def test_missing_video_is_an_input_failure(tmp_path):
    with pytest.raises(ValueError, match="cannot sample"):
        sample_window(tmp_path / "missing.avi", .5)


def test_worker_rejects_missing_and_nonchronological_inputs_without_loading_gpu(tmp_path):
    worker = SmolVLMWorker(tmp_path)
    for pts in ([0, 1, 2], [0, 1, 1, 2], [3, 2, 1, 0]):
        with pytest.raises(ValueError, match="four distinct chronological"):
            worker.analyze([object()] * len(pts), pts)
    with pytest.raises(RuntimeError, match="load the worker"):
        worker.analyze([object()] * 4, [0, 1000, 2000, 3000])
