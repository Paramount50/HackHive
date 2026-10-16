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


def test_explicit_date_and_clock_range_are_applied_in_requested_timezone() -> None:
    intent = parse_natural_query(NaturalQueryRequest(
        question="Show person down between 09:00 and 10:00 on 2026-10-16",
        timezone="Asia/Kolkata",
    ))
    assert intent.end_utc_ms - intent.start_utc_ms == 60 * 60 * 1000
    assert datetime.fromtimestamp(intent.start_utc_ms / 1000, timezone.utc) == datetime(
        2026, 10, 16, 3, 30, tzinfo=timezone.utc
    )


def test_clock_time_without_date_clarifies_instead_of_being_ignored() -> None:
    with pytest.raises(QueryClarification, match="need a date"):
        parse_natural_query(NaturalQueryRequest(
            question="Show person down around 09:00",
        ))


def test_explicit_human_review_and_multiple_cameras_become_filters() -> None:
    intent = parse_natural_query(NaturalQueryRequest(
        question="Show human confirmed person down events on camera 1 and camera 2",
    ))
    assert intent.camera_ids == ["cam-1", "cam-2"]
    assert intent.review_statuses == ["confirmed"]

def test_vlm_verdict_filter_is_not_silently_treated_as_semantic_text() -> None:
    with pytest.raises(QueryClarification, match="VLM verdict filters are not supported"):
        parse_natural_query(NaturalQueryRequest(
            question="Show VLM-supported incidents",
        ))


def test_recording_offset_query_requires_source_and_becomes_filter() -> None:
    intent = parse_natural_query(NaturalQueryRequest(
        question="Show incidents from entrance.mp4 from 00:42.400 to 00:49.200",
    ))
    assert intent.source_name == "entrance.mp4"
    assert intent.start_pts_ms == 42_400
    assert intent.end_pts_ms == 49_200
    with pytest.raises(QueryClarification, match="recording filename"):
        parse_natural_query(NaturalQueryRequest(
            question="Show incidents at video offset 00:42",
        ))


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
        question="Any people following through the door?", now_utc_ms=NOW
    ))
    aggression_intent = parse_natural_query(NaturalQueryRequest(
        question="Were people fighting today?", now_utc_ms=NOW
    ))
    fire_intent = parse_natural_query(NaturalQueryRequest(
        question="Show fire or smoke incidents today", now_utc_ms=NOW
    ))
    intrusion_intent = parse_natural_query(NaturalQueryRequest(
        question="List intruders and trespassing events", now_utc_ms=NOW
    ))
    assert person_intent.categories == ["person_down"]
    assert crowd_intent.categories == ["crowd_gathering"]
    assert tailgate_intent.categories == ["suspected_tailgating"]
    assert aggression_intent.categories == ["aggression_fight"]
    assert fire_intent.categories == ["fire_smoke"]
    assert intrusion_intent.categories == ["intrusion_unauthorized_access"]


def test_query_api_returns_clarification_instead_of_guessing() -> None:
    from fastapi.testclient import TestClient
    from backend.main import app

    with TestClient(app) as client:
        response = client.post("/api/query", json={"question": "Was the building safe?"})
    assert response.status_code == 200
    assert response.json()["needs_clarification"] is True
    assert response.json()["events"] == []
