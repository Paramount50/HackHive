from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

from .replay import BoundedFrameBuffer, VideoReplay


@dataclass
class ReplayJob:
    session_id: str
    camera_id: str
    source_name: str
    started_utc_ms: int
    sample_fps: float
    pace: bool
    max_frames: int | None
    buffer: BoundedFrameBuffer
    status: str = "starting"
    frame_count: int = 0
    error: str | None = None
    stop: Event = field(default_factory=Event)
    thread: Thread | None = None


class ReplayManager:
    """Own replay workers while keeping video decoding off request threads."""

    def __init__(
        self,
        recordings_root: str | Path,
        assets_root: str | Path,
        context_frames: int = 150,
        max_active_jobs: int = 4,
    ) -> None:
        self.recordings_root = Path(recordings_root).resolve()
        self.assets_root = Path(assets_root).resolve()
        self.context_frames = context_frames
        self.max_active_jobs = max_active_jobs
        self._jobs: dict[str, ReplayJob] = {}
        self._lock = Lock()

    def resolve_source(self, source_name: str) -> Path:
        relative = Path(source_name)
        if relative.is_absolute():
            raise ValueError("source_name must be relative to the recordings directory")
        source = (self.recordings_root / relative).resolve()
        if not source.is_relative_to(self.recordings_root):
            raise ValueError("source_name escapes the recordings directory")
        if not source.is_file():
            raise FileNotFoundError(f"recording not found: {source_name}")
        return source

    def start(
        self,
        *,
        session_id: str,
        camera_id: str,
        source_name: str,
        started_utc_ms: int,
        sample_fps: float,
        pace: bool,
        max_frames: int | None,
    ) -> dict[str, Any]:
        source = self.resolve_source(source_name)
        with self._lock:
            active = sum(job.status in {"starting", "running"} for job in self._jobs.values())
            if active >= self.max_active_jobs:
                raise RuntimeError("replay worker limit reached")
            if session_id in self._jobs:
                raise ValueError("session already has a replay job")
            job = ReplayJob(
                session_id=session_id,
                camera_id=camera_id,
                source_name=source_name,
                started_utc_ms=started_utc_ms,
                sample_fps=sample_fps,
                pace=pace,
                max_frames=max_frames,
                buffer=BoundedFrameBuffer(self.context_frames),
            )
            thread = Thread(
                target=self._run,
                args=(job, source),
                name=f"replay-{session_id[:8]}",
                daemon=True,
            )
            job.thread = thread
            self._jobs[session_id] = job
            thread.start()
        return self.status(session_id)

    def _run(self, job: ReplayJob, source: Path) -> None:
        try:
            replay = VideoReplay(
                source,
                camera_id=job.camera_id,
                session_id=job.session_id,
                session_started_utc_ms=job.started_utc_ms,
                asset_dir=self.assets_root / job.session_id,
                sample_fps=job.sample_fps,
                pace=job.pace,
                sleep=job.stop.wait,
            )
            with self._lock:
                job.status = "running"
            for packet in replay.frames(max_frames=job.max_frames):
                if job.stop.is_set():
                    break
                with self._lock:
                    job.buffer.append(packet)
                    job.frame_count += 1
            with self._lock:
                job.status = "stopped" if job.stop.is_set() else "completed"
        except Exception as exc:  # worker failures are exposed through job status
            with self._lock:
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"

    def status(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(session_id)
            if job is None:
                raise KeyError(session_id)
            latest = asdict(job.buffer.latest) if job.buffer.latest else None
            return {
                "session_id": job.session_id,
                "camera_id": job.camera_id,
                "source_name": job.source_name,
                "status": job.status,
                "frame_count": job.frame_count,
                "latest_frame": latest,
                "error": job.error,
            }

    def stop(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(session_id)
            if job is None:
                raise KeyError(session_id)
            job.stop.set()
            if job.status in {"starting", "running"}:
                job.status = "stopping"
        return self.status(session_id)

    def latest_asset(self, session_id: str) -> Path | None:
        with self._lock:
            job = self._jobs.get(session_id)
            if job is None:
                raise KeyError(session_id)
            latest = job.buffer.latest
            return Path(latest.asset_id) if latest else None
