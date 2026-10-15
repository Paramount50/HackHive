from backend.inference.scheduler import (
    BoundedScheduler,
    InferenceRequest,
    RequestReason,
    RequestStatus,
)


def request(name: str, deadline: float = 10) -> InferenceRequest:
    return InferenceRequest(name, RequestReason.CANDIDATE, deadline)


def test_scheduler_bounds_admission_and_marks_overload() -> None:
    scheduler = BoundedScheduler(max_pending=1)
    first, second = request("first"), request("second")
    assert scheduler.submit(first, now=0)
    assert not scheduler.submit(second, now=0)
    assert second.status == RequestStatus.EXPIRED
    assert second.error == "scheduler_queue_full"


def test_scheduler_expiry_is_visible_and_not_normal() -> None:
    scheduler = BoundedScheduler()
    expired = request("expired", deadline=2)
    scheduler.submit(expired, now=0)
    assert scheduler.pop(now=3) is None
    assert expired.status == RequestStatus.EXPIRED
    assert expired.error == "deadline_exceeded"


def test_scheduler_preserves_handler_failures() -> None:
    scheduler = BoundedScheduler()
    failed = request("failed", deadline=10)
    scheduler.submit(failed, now=0)
    result = scheduler.run_next(lambda _: (_ for _ in ()).throw(RuntimeError("model timeout")), now=1)
    assert result is failed
    assert failed.status == RequestStatus.FAILED
    assert failed.error == "model timeout"


def test_candidate_updates_coalesce_without_evicting_audit():
    scheduler = BoundedScheduler(max_pending=2)
    first, replacement = request("old", 100), request("new", 100)
    first.candidate_key = replacement.candidate_key = "session:event"
    audit = InferenceRequest("audit", RequestReason.AUDIT, 100)
    for item in (first, audit, replacement):
        assert scheduler.submit(item, now=0)
    assert first.status == RequestStatus.SUPERSEDED
    assert len(scheduler) == 2
    assert scheduler.pop(now=1) is replacement
    assert scheduler.pop(now=1) is audit


def test_waiting_audit_gets_service_after_four_urgent_candidates():
    scheduler = BoundedScheduler()
    audit = InferenceRequest("audit", RequestReason.AUDIT, 100)
    scheduler.submit(audit, now=0)
    for index in range(4):
        candidate = request(str(index), 100)
        candidate.urgent = True
        scheduler.submit(candidate, now=0)
        assert scheduler.pop(now=1) is candidate
    next_candidate = request("fifth", 100)
    next_candidate.urgent = True
    scheduler.submit(next_candidate, now=0)
    assert scheduler.pop(now=1) is audit


def test_expired_audit_is_counted_and_candidate_limit_is_per_camera():
    scheduler = BoundedScheduler(max_pending=8)
    audit = InferenceRequest("audit", RequestReason.AUDIT, 1)
    scheduler.submit(audit, now=0)
    assert scheduler.pop(now=2) is None
    assert scheduler.counters["audit:deadline_exceeded"] == 1
    assert scheduler.submit(request("one", 100), now=2)
    assert scheduler.submit(request("two", 100), now=2)
    assert not scheduler.submit(request("three", 100), now=2)
    other = request("other", 100)
    other.camera_id = "cam-2"
    assert scheduler.submit(other, now=2)


def test_explicit_handler_failure_is_not_overwritten_as_completed():
    scheduler = BoundedScheduler()
    item = request("bad-json", 100)
    scheduler.submit(item, now=0)
    def handler(job):
        job.status = RequestStatus.FAILED
        job.error = "invalid_model_output"
    scheduler.run_next(handler, now=1)
    assert item.status == RequestStatus.FAILED
