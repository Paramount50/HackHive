from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Callable, Iterable


@dataclass(frozen=True)
class FramePacket:
    camera_id: str
    session_id: str
    frame_id: int
    source_pts_ms: int
    captured_utc_ms: int
    width: int
    height: int
    asset_id: str
    fixture: bool = True


class BoundedFrameBuffer:
    """Latest-frame slot plus a bounded timestamped context ring."""

    def __init__(self, max_frames: int = 60) -> None:
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        self._ring: deque[FramePacket] = deque(maxlen=max_frames)
        self._latest: FramePacket | None = None

    def append(self, packet: FramePacket) -> None:
        if self._latest and packet.source_pts_ms < self._latest.source_pts_ms:
            raise ValueError("frame source PTS must be monotonic")
        self._latest = packet
        self._ring.append(packet)

    @property
    def latest(self) -> FramePacket | None:
        return self._latest

    def between(self, start_pts_ms: int, end_pts_ms: int) -> list[FramePacket]:
        if end_pts_ms < start_pts_ms:
            raise ValueError("end PTS must be after start PTS")
        return [item for item in self._ring if start_pts_ms <= item.source_pts_ms <= end_pts_ms]

    def __len__(self) -> int:
        return len(self._ring)


class FixtureReplay:
    def __init__(self, packets: Iterable[FramePacket], pace: bool = False) -> None:
        self.packets = list(packets)
        if any(b.source_pts_ms < a.source_pts_ms for a, b in zip(self.packets, self.packets[1:])):
            raise ValueError("replay packets must be ordered by source PTS")
        self.pace = pace

    def frames(self) -> Iterable[FramePacket]:
        previous_pts: int | None = None
        previous_clock = monotonic()
        for packet in self.packets:
            if self.pace and previous_pts is not None:
                target = (packet.source_pts_ms - previous_pts) / 1000
                elapsed = monotonic() - previous_clock
                if target > elapsed:
                    import time

                    time.sleep(target - elapsed)
            yield packet
            previous_pts = packet.source_pts_ms
            previous_clock = monotonic()


class VideoReplay:
    """Decode a local video into sampled frame packets with source-relative PTS.

    OpenCV is imported lazily so fixture-only development remains possible without
    the decoder dependency. Decoded frame assets are bounded by the caller-owned
    directory lifecycle; they are evidence inputs, not permanent event media.
    """

    def __init__(
        self,
        path: str | Path,
        camera_id: str,
        session_id: str,
        session_started_utc_ms: int,
        asset_dir: str | Path,
        sample_fps: float = 5.0,
        pace: bool = False,
        decode_strategy: str = "sequential",
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"video file not found: {self.path}")
        if sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        if decode_strategy not in {"sequential", "grab"}:
            raise ValueError("decode_strategy must be sequential or grab")
        self.camera_id = camera_id
        self.session_id = session_id
        self.session_started_utc_ms = session_started_utc_ms
        self.asset_dir = Path(asset_dir)
        self.sample_fps = sample_fps
        self.pace = pace
        self.decode_strategy = decode_strategy
        self._sleep = sleep

    @staticmethod
    def _opencv():
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - exercised only in incomplete installs
            raise RuntimeError(
                "real replay requires opencv-python-headless; install requirements.txt"
            ) from exc
        return cv2

    def frames(self, max_frames: int | None = None) -> Iterable[FramePacket]:
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be positive")
        cv2 = self._opencv()
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"unable to decode video: {self.path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = self.sample_fps
        interval_ms = 1000.0 / self.sample_fps
        next_sample_pts = 0.0
        emitted = 0
        decoded_index = 0
        previous_pts = -1
        replay_clock = monotonic()
        first_pts: int | None = None
        sleep = self._sleep
        if sleep is None:
            import time

            sleep = time.sleep

        self.asset_dir.mkdir(parents=True, exist_ok=True)
        try:
            while True:
                if self.decode_strategy == "grab":
                    ok = capture.grab()
                    image = None
                else:
                    ok, image = capture.read()
                if not ok:
                    break
                reported_pts = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                fallback_pts = decoded_index * 1000.0 / fps
                absolute_pts = reported_pts if reported_pts >= 0 else fallback_pts
                if decoded_index > 0 and absolute_pts <= previous_pts:
                    absolute_pts = fallback_pts
                source_pts_ms = max(previous_pts + 1, int(round(absolute_pts)))
                previous_pts = source_pts_ms
                decoded_index += 1

                if source_pts_ms + 0.5 < next_sample_pts:
                    continue
                while next_sample_pts <= source_pts_ms + 0.5:
                    next_sample_pts += interval_ms

                if image is None:
                    ok, image = capture.retrieve()
                    if not ok:
                        raise ValueError(f"unable to retrieve sampled frame at {source_pts_ms} ms")

                if first_pts is None:
                    first_pts = source_pts_ms
                if self.pace:
                    target_elapsed = (source_pts_ms - first_pts) / 1000.0
                    remaining = target_elapsed - (monotonic() - replay_clock)
                    if remaining > 0:
                        sleep(remaining)

                frame_id = emitted
                asset_path = self.asset_dir / f"{frame_id:08d}.jpg"
                if not cv2.imwrite(str(asset_path), image):
                    raise OSError(f"failed to write decoded frame: {asset_path}")
                height, width = image.shape[:2]
                yield FramePacket(
                    camera_id=self.camera_id,
                    session_id=self.session_id,
                    frame_id=frame_id,
                    source_pts_ms=source_pts_ms,
                    captured_utc_ms=self.session_started_utc_ms + source_pts_ms,
                    width=int(width),
                    height=int(height),
                    asset_id=str(asset_path.resolve()),
                    fixture=False,
                )
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    break
        finally:
            capture.release()
