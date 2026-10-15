from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import monotonic
from typing import Iterable


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
