from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from enum import StrEnum
from time import monotonic
from threading import Lock
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
    SUPERSEDED = "superseded"


@dataclass
class InferenceRequest:
    request_id: str
    reason: RequestReason
    deadline_monotonic: float
    status: RequestStatus = RequestStatus.PENDING
    error: str | None = None
    camera_id: str = "cam-1"
    candidate_key: str | None = None
    urgent: bool = False
    submitted_monotonic: float = 0


class BoundedScheduler:
    """A single-worker queue with explicit expiry and bounded admission."""

    def __init__(self, max_pending: int = 3) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self.max_pending = max_pending
        self._pending: deque[InferenceRequest] = deque()
        self._lock = Lock()
        self._worker_lock = Lock()
        self._candidate_streak = 0
        self.counters: Counter[str] = Counter()

    def submit(self, request: InferenceRequest, now: float | None = None) -> bool:
        with self._lock:
            if request.status != RequestStatus.PENDING:
                raise ValueError("only a fresh pending request can be submitted")
            request.submitted_monotonic = monotonic() if now is None else now
            self._expire(request.submitted_monotonic)
            if request.deadline_monotonic <= request.submitted_monotonic:
                self._reject(request, "deadline_exceeded")
                return False
            same_camera = [item for item in self._pending if item.camera_id == request.camera_id]
            for old in same_camera:
                same_candidate = (request.reason == old.reason == RequestReason.CANDIDATE
                                  and request.candidate_key is not None
                                  and request.candidate_key == old.candidate_key)
                if same_candidate or request.reason == old.reason == RequestReason.AUDIT:
                    self._pending.remove(old)
                    old.status, old.error = RequestStatus.SUPERSEDED, "coalesced_by_newer_request"
                    self.counters["superseded"] += 1
                    break
            if len(self._pending) >= self.max_pending:
                self._reject(request, "scheduler_queue_full")
                return False
            count = sum(item.camera_id == request.camera_id and item.reason == RequestReason.CANDIDATE
                        for item in self._pending)
            if request.reason == RequestReason.CANDIDATE and count >= 2:
                self._reject(request, "camera_candidate_limit")
                return False
            self._pending.append(request)
            return True

    def _reject(self, request: InferenceRequest, reason: str) -> None:
        request.status, request.error = RequestStatus.EXPIRED, reason
        self.counters[f"{request.reason}:{reason}"] += 1

    def _expire(self, current: float) -> None:
        for request in list(self._pending):
            if request.deadline_monotonic <= current:
                self._pending.remove(request)
                self._reject(request, "deadline_exceeded")

    def pop(self, now: float | None = None) -> InferenceRequest | None:
        current = monotonic() if now is None else now
        with self._lock:
            self._expire(current)
            audits = [item for item in self._pending if item.reason == RequestReason.AUDIT]
            candidates = [item for item in self._pending if item.reason == RequestReason.CANDIDATE]
            urgent = [item for item in candidates if item.urgent]
            aged = [item for item in audits if current - item.submitted_monotonic >= 5]
            if audits and self._candidate_streak >= 4:
                selected = audits[0]
            else:
                pool = urgent or aged or candidates or audits or list(self._pending)
                if not pool:
                    return None
                selected = pool[0]
            self._pending.remove(selected)
            self._candidate_streak = self._candidate_streak + 1 if selected.reason == RequestReason.CANDIDATE else 0
            return selected

    def run_next(
        self,
        handler: Callable[[InferenceRequest], None],
        now: float | None = None,
    ) -> InferenceRequest | None:
        with self._worker_lock:
            request = self.pop(now)
            if request is None:
                return None
            try:
                handler(request)
            except Exception as error:
                request.status = RequestStatus.FAILED
                request.error = str(error)[:200]
            else:
                # Handlers may report an explicit model/output failure without raising.
                if request.status == RequestStatus.PENDING:
                    request.status = RequestStatus.COMPLETED
            return request

    def __len__(self) -> int:
        with self._lock:
            return len(self._pending)
