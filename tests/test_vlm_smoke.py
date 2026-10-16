import cv2
import numpy as np
import pytest

from backend.inference.smolvlm import (
    SmolVLMWorker, candidate_prompt, model_config, validate_provenance,
    vision_sampling_compatibility,
)
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


def test_vlm_variants_have_independent_pinned_configs_and_directories():
    five_hundred = model_config("500m")
    two_point_two = model_config("2.2b")
    assert five_hundred.model_id.endswith("500M-Video-Instruct")
    assert two_point_two.model_id.endswith("2.2B-Instruct")
    assert five_hundred.revision != two_point_two.revision
    assert five_hundred.default_dir != two_point_two.default_dir
    assert two_point_two.minimum_budget_mib > five_hundred.minimum_budget_mib
    assert two_point_two.budget_cap_mib > five_hundred.budget_cap_mib


def test_invalid_vlm_variant_and_provenance_are_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        model_config("fight")
    config = model_config("2.2b")
    validate_provenance({"model_id": config.model_id, "revision": config.revision}, "2.2b")
    with pytest.raises(ValueError, match="provenance"):
        validate_provenance({"model_id": config.model_id, "revision": "wrong"}, "2.2b")


def test_swift_sampling_requires_expected_vision_internals():
    class Compatible:
        class Layer:
            layer_norm1 = object()
            self_attn = type("Attention", (), {"k_proj": object()})()

        model = type("Core", (), {
            "vision_model": type("Vision", (), {
                "encoder": type("Encoder", (), {"layers": [Layer()]})()
            })()
        })()

    assert vision_sampling_compatibility(Compatible()) == (True, None)
    compatible, error = vision_sampling_compatibility(object())
    assert compatible is False
    assert "AttributeError" in error


def test_candidate_prompt_is_category_conditioned():
    prompt = candidate_prompt("aggression_fight")
    assert "physical_contact" in prompt
    assert "intrusion_unauthorized_access" not in prompt
    assert "Do not choose another category" in prompt


def test_candidate_prompt_rejects_audit_outcomes():
    with pytest.raises(ValueError, match="controlled incident"):
        candidate_prompt("none")
