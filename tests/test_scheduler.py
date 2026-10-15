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
    assert scheduler.submit(first)
    assert not scheduler.submit(second)
    assert second.status == RequestStatus.EXPIRED
    assert second.error == "scheduler_queue_full"


def test_scheduler_expiry_is_visible_and_not_normal() -> None:
    scheduler = BoundedScheduler()
    expired = request("expired", deadline=2)
    scheduler.submit(expired)
    assert scheduler.pop(now=3) is None
    assert expired.status == RequestStatus.EXPIRED
    assert expired.error == "deadline_exceeded"


def test_scheduler_preserves_handler_failures() -> None:
    scheduler = BoundedScheduler()
    failed = request("failed", deadline=10)
    scheduler.submit(failed)
    result = scheduler.run_next(lambda _: (_ for _ in ()).throw(RuntimeError("model timeout")), now=1)
    assert result is failed
    assert failed.status == RequestStatus.FAILED
    assert failed.error == "model timeout"
