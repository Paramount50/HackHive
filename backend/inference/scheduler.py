from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from time import monotonic
from typing import Callable


class RequestReason(StrEnum):
    CANDIDATE = "candidate"
    AUDIT = "audit"
    CHAT = "chat"


class RequestStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    EXPIRED = "expired"
    FAILED = "failed"


@dataclass
class InferenceRequest:
    request_id: str
    reason: RequestReason
    deadline_monotonic: float
    status: RequestStatus = RequestStatus.PENDING
    error: str | None = None


class BoundedScheduler:
    """A single-worker queue with explicit expiry and bounded admission."""

    def __init__(self, max_pending: int = 3) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self.max_pending = max_pending
        self._pending: deque[InferenceRequest] = deque()

    def submit(self, request: InferenceRequest) -> bool:
        if len(self._pending) >= self.max_pending:
            request.status = RequestStatus.EXPIRED
            request.error = "scheduler_queue_full"
            return False
        self._pending.append(request)
        return True

    def pop(self, now: float | None = None) -> InferenceRequest | None:
        current = monotonic() if now is None else now
        while self._pending:
            request = self._pending.popleft()
            if request.deadline_monotonic <= current:
                request.status = RequestStatus.EXPIRED
                request.error = "deadline_exceeded"
                continue
            return request
        return None

    def run_next(
        self,
        handler: Callable[[InferenceRequest], None],
        now: float | None = None,
    ) -> InferenceRequest | None:
        request = self.pop(now)
        if request is None:
            return None
        try:
            handler(request)
        except Exception as error:
            request.status = RequestStatus.FAILED
            request.error = str(error)[:200]
        else:
            request.status = RequestStatus.COMPLETED
        return request

    def __len__(self) -> int:
        return len(self._pending)
