from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from threading import Lock


MODEL_ID = "microsoft/xclip-base-patch32"
MODEL_REVISION = "a2e27a78a2b5d802e894b8a1ef14f3a8ce490963"
PROMPT_VERSION = "cctv-actions-v1"
PROMPTS = {
    "aggression_fight": (
        "A CCTV video of people fighting physically.",
        "A CCTV video of people pushing and striking one another.",
        "A CCTV video of people wrestling in a physical altercation.",
    ),
    "person_down": (
        "A CCTV video of a person lying on the floor.",
        "A CCTV video of a person falling down.",
        "A CCTV video of a person prone on the ground.",
    ),
    "crowd_gathering": (
        "A CCTV video of a crowd gathered together in one area.",
        "A CCTV video of a group of people standing together.",
        "A CCTV video of a crowd gathered in a public area.",
    ),
    "none": (
        "A CCTV video of people walking calmly in a public place.",
        "A CCTV video of people working normally.",
        "A CCTV video of people shopping normally.",
    ),
}
EVENT_CATEGORIES = tuple(name for name in PROMPTS if name != "none")
MIN_PROBABILITY = 0.35
MIN_MARGIN = 0.10


@dataclass(frozen=True)
class ActionClassification:
    outcome: str
    top_category: str
    normalized_scores: dict[str, float]
    class_scores: dict[str, float]
    margin: float
    latency_ms: float = 0.0


def classify_logits(logits: list[float], *, latency_ms: float = 0.0) -> ActionClassification:
    expected = sum(len(prompts) for prompts in PROMPTS.values())
    if len(logits) != expected or any(not math.isfinite(value) for value in logits):
        raise ValueError(f"expected {expected} finite prompt scores")
    class_scores = {}
    offset = 0
    for category, prompts in PROMPTS.items():
        class_scores[category] = sum(logits[offset:offset + len(prompts)]) / len(prompts)
        offset += len(prompts)

    maximum = max(class_scores.values())
    exponentials = {key: math.exp(value - maximum) for key, value in class_scores.items()}
    denominator = sum(exponentials.values())
    probabilities = {key: value / denominator for key, value in exponentials.items()}
    ranked = sorted(probabilities, key=probabilities.get, reverse=True)
    top, second = ranked[:2]
    margin = probabilities[top] - probabilities[second]
    outcome = top if probabilities[top] >= MIN_PROBABILITY and margin >= MIN_MARGIN else "unknown"
    return ActionClassification(outcome, top, probabilities, class_scores, margin, latency_ms)


class XClipActionClassifier:
    """Local zero-shot action ranking; predictions remain uncalibrated and reviewable."""

    def __init__(self, model_dir: str | Path, *, device: str = "cuda:0") -> None:
        self.model_dir = Path(model_dir).resolve()
        self.device = device
        provenance_path = self.model_dir / "hackhive-action-model.json"
        if not provenance_path.is_file():
            raise RuntimeError("local XCLIP model provenance is missing; run download_action_model.py")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance != {"model_id": MODEL_ID, "revision": MODEL_REVISION}:
            raise ValueError("local XCLIP model provenance does not match the tested revision")
        try:
            import torch
            from transformers import AutoProcessor, XCLIPModel
        except ImportError as exc:
            raise RuntimeError("XCLIP requires the pinned Transformers and PyTorch dependencies") from exc
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for the local action classifier")
        self.processor = AutoProcessor.from_pretrained(
            self.model_dir, local_files_only=True, trust_remote_code=False
        )
        self.model = XCLIPModel.from_pretrained(
            self.model_dir, local_files_only=True, trust_remote_code=False
        ).to(device).eval()
        self._torch = torch
        self._prompts = [prompt for prompts in PROMPTS.values() for prompt in prompts]
        self._inference_lock = Lock()

    def classify(self, image_paths: list[str | Path]) -> ActionClassification:
        if len(image_paths) != 8:
            raise ValueError("XCLIP action classification requires exactly eight ordered frames")
        from PIL import Image

        started = time.perf_counter()
        frames = []
        for path in image_paths:
            with Image.open(path) as image:
                frames.append(image.convert("RGB"))
        inputs = self.processor(
            text=self._prompts, images=[frames], return_tensors="pt", padding=True
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with self._inference_lock, self._torch.inference_mode():
            logits = self.model(**inputs).logits_per_video[0].float().cpu().tolist()
        return classify_logits(logits, latency_ms=(time.perf_counter() - started) * 1000)
