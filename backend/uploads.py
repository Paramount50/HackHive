from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from .contracts import utc_now_ms


ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}
CHUNK_BYTES = 1024 * 1024


class UploadValidationError(ValueError):
    pass


async def save_recording_upload(
    upload: UploadFile,
    root: str | Path,
    *,
    max_bytes: int,
) -> dict[str, Any]:
    original_name = Path(upload.filename or "").name
    suffix = Path(original_name).suffix.lower()
    if not original_name or suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise UploadValidationError("supported video extensions are .mp4, .mov, .avi, and .mkv")
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")

    recording_id = str(uuid.uuid4())
    stored_name = f"{recording_id}{suffix}"
    directory = Path(root).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / stored_name
    temporary = directory / f".{recording_id}.upload{suffix}"
    size = 0
    digest = hashlib.sha256()
    try:
        with temporary.open("xb") as stream:
            while chunk := await upload.read(CHUNK_BYTES):
                size += len(chunk)
                if size > max_bytes:
                    raise UploadValidationError(f"video exceeds the {max_bytes // (1024 * 1024)} MiB upload limit")
                digest.update(chunk)
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if size == 0:
            raise UploadValidationError("uploaded video is empty")

        import cv2

        capture = cv2.VideoCapture(str(temporary))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ok, _ = capture.read()
            if not capture.isOpened() or not ok or fps <= 0 or frame_count < 1 or width < 1 or height < 1:
                raise UploadValidationError("uploaded file is not a decodable video")
        finally:
            capture.release()
        duration_ms = max(1, round(frame_count * 1000 / fps))
        os.replace(temporary, destination)
        return {
            "recording_id": recording_id,
            "original_name": original_name,
            "stored_name": stored_name,
            "mime_type": upload.content_type or "application/octet-stream",
            "byte_size": size,
            "sha256": digest.hexdigest(),
            "duration_ms": duration_ms,
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "created_utc_ms": utc_now_ms(),
        }
    except Exception:
        temporary.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
