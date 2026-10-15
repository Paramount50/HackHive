from datetime import datetime, timezone

import pytest

from backend.contracts import NaturalQueryRequest
from backend.query import QueryClarification, parse_natural_query


NOW = int(datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)


def test_tailgating_today_becomes_closed_intent() -> None:
    intent = parse_natural_query(NaturalQueryRequest(
        question="Show all tailgating events from camera 2 today", now_utc_ms=NOW
    ))
    assert intent.operation == "list"
    assert intent.categories == ["suspected_tailgating"]
    assert intent.camera_ids == ["cam-2"]
    start = datetime.fromtimestamp(intent.start_utc_ms / 1000, timezone.utc)
    end = datetime.fromtimestamp(intent.end_utc_ms / 1000, timezone.utc)
    assert start == datetime(2026, 10, 14, 18, 30, tzinfo=timezone.utc)
    assert end == datetime(2026, 10, 15, 18, 30, tzinfo=timezone.utc)


def test_count_person_down_in_last_two_hours() -> None:
    intent = parse_natural_query(NaturalQueryRequest(
        question="How many person down incidents were there in the last 2 hours?", now_utc_ms=NOW
    ))
    assert intent.operation == "count"
    assert intent.categories == ["person_down"]
    assert intent.end_utc_ms - intent.start_utc_ms == 2 * 60 * 60 * 1000


def test_unsupported_category_requests_clarification() -> None:
    with pytest.raises(QueryClarification, match="Unsupported category 'gun'"):
        parse_natural_query(NaturalQueryRequest(question="Show gun incidents today", now_utc_ms=NOW))


def test_chatbot_maps_natural_synonyms_to_supported_categories() -> None:
    person_intent = parse_natural_query(NaturalQueryRequest(
        question="Did someone fall on the ground today?", now_utc_ms=NOW
    ))
    crowd_intent = parse_natural_query(NaturalQueryRequest(
        question="Are there large groups of people at the entrance?", now_utc_ms=NOW
    ))
    tailgate_intent = parse_natural_query(NaturalQueryRequest(
        question="Any unauthorized access or people following through the door?", now_utc_ms=NOW
    ))
    assert person_intent.categories == ["person_down"]
    assert crowd_intent.categories == ["crowd_gathering"]
    assert tailgate_intent.categories == ["suspected_tailgating"]


def test_query_api_returns_clarification_instead_of_guessing() -> None:
    from fastapi.testclient import TestClient
    from backend.main import app

    with TestClient(app) as client:
        response = client.post("/api/query", json={"question": "Was the building safe?"})
    assert response.status_code == 200
    assert response.json()["needs_clarification"] is True
    assert response.json()["events"] == []
