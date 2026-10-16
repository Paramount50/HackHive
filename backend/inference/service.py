from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic, perf_counter
from typing import Any, Callable, Protocol
import uuid

from PIL import Image

from .scheduler import BoundedScheduler, InferenceRequest, RequestReason, RequestStatus


@dataclass(frozen=True)
class SemanticJob:
    event_id: str
    camera_id: str
    candidate_key: str
    expected_outcome: str
    image_paths: tuple[str, ...]
    source_pts_ms: tuple[int, ...]


class VisualWorker(Protocol):
    def load(self) -> dict[str, Any]: ...
    def analyze(self, images: list[Image.Image], pts_ms: list[int]) -> dict[str, Any]: ...
    def analyze_candidate(
        self, images: list[Image.Image], pts_ms: list[int], candidate_category: str
    ) -> dict[str, Any]: ...


ResultHandler = Callable[[SemanticJob, dict[str, Any]], None]


class SemanticInferenceService:
    """One background VLM worker behind bounded, deadline-aware admission."""

    def __init__(
        self,
        worker: VisualWorker,
        on_result: ResultHandler,
        *,
        max_pending: int = 3,
        deadline_seconds: float = 20.0,
        selection_strategy: str = "swift",
    ) -> None:
        if deadline_seconds <= 0:
            raise ValueError("deadline_seconds must be positive")
        if selection_strategy not in {"swift", "uniform"}:
            raise ValueError("selection_strategy must be swift or uniform")
        self.worker = worker
        self.on_result = on_result
        self.scheduler = BoundedScheduler(max_pending=max_pending)
        self.deadline_seconds = deadline_seconds
        self.selection_strategy = selection_strategy
        self._jobs: dict[str, SemanticJob] = {}
        self._jobs_lock = Lock()
        self._wake = Event()
        self._stop = Event()
        self._thread: Thread | None = None
        self.state = "stopped"
        self.error: str | None = None
        self.metadata: dict[str, Any] = {}

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.state = "loading"
        self._thread = Thread(target=self._run, name="hackhive-vlm", daemon=True)
        self._thread.start()

    def submit(self, job: SemanticJob) -> bool:
        if len(job.source_pts_ms) < 4 or len(job.image_paths) != len(job.source_pts_ms) or any(
            right <= left for left, right in zip(job.source_pts_ms, job.source_pts_ms[1:])
        ):
            raise ValueError("semantic job requires at least four distinct chronological frames")
        if self.state == "failed":
            return False
        request_id = str(uuid.uuid4())
        request = InferenceRequest(
            request_id=request_id,
            reason=RequestReason.CANDIDATE,
            deadline_monotonic=monotonic() + self.deadline_seconds,
            camera_id=job.camera_id,
            candidate_key=job.candidate_key,
            urgent=True,
        )
        with self._jobs_lock:
            self._jobs[request_id] = job
        admitted = self.scheduler.submit(request)
        if not admitted:
            with self._jobs_lock:
                self._jobs.pop(request_id, None)
            return False
        self._wake.set()
        return True

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "error": self.error,
            "pending": len(self.scheduler),
            "counters": dict(self.scheduler.counters),
            "model_id": self.metadata.get("model_id"),
            "model_revision": self.metadata.get("revision"),
            "selection_strategy": self.selection_strategy,
        }

    def close(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        try:
            self.metadata = self.worker.load()
        except Exception as exc:
            self.state = "failed"
            self.error = f"{type(exc).__name__}: {exc}"[:500]
            return
        self.state = "ready"
        while not self._stop.is_set():
            request = self.scheduler.run_next(self._handle)
            if request is None:
                self._wake.wait(0.5)
                self._wake.clear()

    def _handle(self, request: InferenceRequest) -> None:
        with self._jobs_lock:
            job = self._jobs.pop(request.request_id, None)
        if job is None:
            request.status = RequestStatus.FAILED
            request.error = "semantic_job_missing"
            return
        images: list[Image.Image] = []
        try:
            for path in job.image_paths:
                with Image.open(Path(path)) as image:
                    images.append(image.convert("RGB"))
            selection_started = perf_counter()
            selection_error = None
            if len(images) == 4:
                selected_indices = list(range(4))
                strategy = "all"
            elif self.selection_strategy == "swift" and hasattr(self.worker, "select_frames"):
                gaps = [right - left for left, right in zip(job.source_pts_ms, job.source_pts_ms[1:])]
                mean_gap = sum(gaps) / len(gaps)
                regular = all(abs(gap - mean_gap) <= max(1, mean_gap * 0.25) for gap in gaps)
                try:
                    if not regular:
                        raise ValueError("candidate timestamps are too irregular for Taylor sampling")
                    selected_indices = self.worker.select_frames(images, target_frames=4)
                    if (
                        len(selected_indices) != 4
                        or selected_indices != sorted(set(selected_indices))
                        or any(index < 0 or index >= len(images) for index in selected_indices)
                    ):
                        raise ValueError("selector returned invalid frame indices")
                except Exception as exc:
                    selection_error = f"{type(exc).__name__}: {exc}"[:300]
                    selected_indices = [round(index * (len(images) - 1) / 3) for index in range(4)]
                    strategy = "uniform_fallback"
                else:
                    strategy = "swift_latent_taylor_v1"
            else:
                selected_indices = [round(index * (len(images) - 1) / 3) for index in range(4)]
                strategy = "uniform_requested"
            selected_images = [images[index] for index in selected_indices]
            selected_pts = [job.source_pts_ms[index] for index in selected_indices]
            selection_ms = (perf_counter() - selection_started) * 1000
            if hasattr(self.worker, "analyze_candidate"):
                result = self.worker.analyze_candidate(
                    selected_images, selected_pts, job.expected_outcome
                )
            else:
                result = self.worker.analyze(selected_images, selected_pts)
            result["selection"] = {
                "strategy": strategy,
                "candidate_frame_count": len(images),
                "selected_indices": selected_indices,
                "selected_source_pts_ms": selected_pts,
                "target_frames": 4,
                "taylor_order": 3 if strategy == "swift_latent_taylor_v1" else None,
                "latency_ms": selection_ms,
                "error": selection_error,
            }
            self.on_result(job, result)
            if result.get("status") != "completed":
                request.status = RequestStatus.FAILED
                request.error = str(result.get("error_code") or "vlm_inference_failed")
        except Exception as exc:
            request.status = RequestStatus.FAILED
            request.error = f"{type(exc).__name__}: {exc}"[:200]
            self.on_result(job, {
                "status": "failed",
                "observation": None,
                "error_code": "service_failure",
                "error": request.error,
                "source_pts_ms": list(job.source_pts_ms),
                "timings_ms": {},
            })
