from __future__ import annotations

from fastapi.testclient import TestClient

from backend.contracts import FixtureAnalysis, QueryIntent, ReviewRequest
from backend.storage import Store


def make_store() -> Store:
    store = Store()
    store.seed_camera()
    return store


def test_unknown_is_not_an_incident() -> None:
    store = make_store()
    session = store.create_session(1_000)
    analysis = FixtureAnalysis(outcome="unknown", summary="Frames unavailable", limitations=["timeout"])
    assert store.record_analysis(session, 2_000, 3_000, analysis, "unknown-1") is None
    assert store.list_events(QueryIntent()).__len__() == 0


def test_repeated_observation_merges_into_one_event() -> None:
    store = make_store()
    session = store.create_session(1_000)
    analysis = FixtureAnalysis(
        outcome="person_down",
        summary="A person remains prone near the doorway.",
        observations=[{"predicate": "prone_posture", "supported": True}],
    )
    first = store.record_analysis(session, 2_000, 3_000, analysis, "person-1")
    second = store.record_analysis(session, 3_000, 4_000, analysis, "person-1")
    assert first == second
    assert len(store.list_events(QueryIntent())) == 1


def test_review_is_idempotent_and_separate_from_system_decision() -> None:
    store = make_store()
    session = store.create_session(1_000)
    analysis = FixtureAnalysis(
        outcome="crowd_gathering",
        summary="Four people remain in the marked zone.",
        observations=[{"predicate": "people_in_zone", "supported": True}],
    )
    event_id = store.record_analysis(session, 2_000, 3_000, analysis, "crowd-1")
    request = ReviewRequest(action="confirm", actor_label="tester", request_key="review-1")
    store.review(event_id, request)
    store.review(event_id, request)
    result = store.list_events(QueryIntent())[0]
    assert result["decision_state"] == "accepted"
    assert result["review_status"] == "confirmed"
    assert len(result["reviews"]) == 1


def test_query_uses_half_open_time_filters() -> None:
    store = make_store()
    session = store.create_session(1_000)
    analysis = FixtureAnalysis(
        outcome="person_down", summary="A person is lying on the floor.",
        observations=[{"predicate": "prone_posture", "supported": True}],
    )
    store.record_analysis(session, 2_000, 3_000, analysis, "time-1")
    assert len(store.list_events(QueryIntent(start_utc_ms=3_000))) == 1
    assert len(store.list_events(QueryIntent(end_utc_ms=2_000))) == 0


def test_api_fixture_and_citations() -> None:
    from backend.main import app

    with TestClient(app) as client:
        response = client.post("/api/fixtures/run")
        assert response.status_code == 200
        assert response.json()["fixture"] is True
        query = client.post("/api/query", json={"operation": "list"})
        assert query.status_code == 200
        assert query.json()["citations"]
