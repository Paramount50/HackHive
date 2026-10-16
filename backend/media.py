from __future__ import annotations

import hashlib
import math
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from .ingest.replay import FramePacket


@dataclass(frozen=True)
class MediaArtifact:
    kind: str
    relative_path: str
    mime_type: str
    coverage_start_utc_ms: int
    coverage_end_utc_ms: int
    requested_start_utc_ms: int
    requested_end_utc_ms: int
    byte_size: int
    sha256: str


class MediaExporter:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    @staticmethod
    def _digest(path: Path) -> tuple[int, str]:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return path.stat().st_size, digest.hexdigest()

    def export_keyframe(self, event_id: str, packet: FramePacket) -> MediaArtifact:
        directory = self.root / event_id
        directory.mkdir(parents=True, exist_ok=True)
        name = f"{uuid.uuid4()}.jpg"
        destination = directory / name
        temporary = directory / f".{name}.tmp"
        shutil.copyfile(packet.asset_id, temporary)
        with temporary.open("rb+") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        size, digest = self._digest(destination)
        return MediaArtifact(
            kind="keyframe",
            relative_path=destination.relative_to(self.root).as_posix(),
            mime_type="image/jpeg",
            coverage_start_utc_ms=packet.captured_utc_ms,
            coverage_end_utc_ms=packet.captured_utc_ms,
            requested_start_utc_ms=packet.captured_utc_ms,
            requested_end_utc_ms=packet.captured_utc_ms,
            byte_size=size,
            sha256=digest,
        )

    def export_clip(
        self,
        event_id: str,
        packets: list[FramePacket],
        *,
        requested_start_utc_ms: int,
        requested_end_utc_ms: int,
    ) -> MediaArtifact:
        selected = sorted(
            (packet for packet in packets if requested_start_utc_ms <= packet.captured_utc_ms <= requested_end_utc_ms),
            key=lambda packet: packet.source_pts_ms,
        )
        if not selected:
            raise ValueError("no sampled frames cover the requested clip interval")
        import cv2

        first = cv2.imread(selected[0].asset_id)
        if first is None:
            raise ValueError("first sampled frame cannot be decoded")
        height, width = first.shape[:2]
        duration_seconds = max(0.001, (selected[-1].source_pts_ms - selected[0].source_pts_ms) / 1000)
        fps = min(30.0, max(1.0, (len(selected) - 1) / duration_seconds))
        directory = self.root / event_id
        directory.mkdir(parents=True, exist_ok=True)
        name = f"{uuid.uuid4()}.webm"
        destination = directory / name
        temporary = directory / f".{name}.tmp.webm"
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"VP90"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("OpenCV VP9/WebM writer is unavailable")
        try:
            for packet in selected:
                frame = cv2.imread(packet.asset_id)
                if frame is None:
                    raise ValueError(f"sampled frame cannot be decoded: {packet.frame_id}")
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
        finally:
            writer.release()
        if not temporary.is_file() or temporary.stat().st_size == 0:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("VP9/WebM clip export produced no playable data")
        os.replace(temporary, destination)
        validation = cv2.VideoCapture(str(destination))
        try:
            ok, _ = validation.read()
            if not ok:
                destination.unlink(missing_ok=True)
                raise RuntimeError("exported incident clip failed playback verification")
        finally:
            validation.release()
        size, digest = self._digest(destination)
        return MediaArtifact(
            kind="event_clip",
            relative_path=destination.relative_to(self.root).as_posix(),
            mime_type="video/webm",
            coverage_start_utc_ms=selected[0].captured_utc_ms,
            coverage_end_utc_ms=selected[-1].captured_utc_ms,
            requested_start_utc_ms=requested_start_utc_ms,
            requested_end_utc_ms=requested_end_utc_ms,
            byte_size=size,
            sha256=digest,
        )


def transcode_legacy_mp4_clip(source: str | Path) -> tuple[Path, int, str]:
    source_path = Path(source).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"legacy incident clip is missing: {source_path}")
    import cv2

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("legacy incident clip cannot be decoded")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0 or fps > 120:
        fps = 10.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("legacy incident clip has invalid dimensions")

    destination = source_path.with_name(f"{source_path.stem}-{uuid.uuid4().hex}.webm")
    temporary = destination.with_name(f".{destination.name}.tmp.webm")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"VP90"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("OpenCV VP9/WebM writer is unavailable for legacy clip conversion")
    frames_written = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            frames_written += 1
    finally:
        capture.release()
        writer.release()
    if frames_written == 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("legacy incident clip conversion produced no playable frames")
    os.replace(temporary, destination)
    converted = cv2.VideoCapture(str(destination))
    try:
        ok, _ = converted.read()
        if not ok:
            destination.unlink(missing_ok=True)
            raise RuntimeError("converted incident clip failed playback verification")
    finally:
        converted.release()
    size, digest = MediaExporter._digest(destination)
    return destination, size, digest
