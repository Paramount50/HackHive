from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.chat import HybridQueryEngine, ModelRoutingDecision, SemanticRanker
from backend.contracts import FixtureAnalysis, NaturalQueryRequest, QueryIntent
from backend.query import QueryClarification
from backend.storage import Store


NOW = int(datetime(2026, 10, 15, 4, 0, tzinfo=timezone.utc).timestamp() * 1000)


def test_count_is_not_truncated_by_list_retrieval_limit() -> None:
    store = Store()
    session = store.create_session(NOW - 100_000)
    for index in range(101):
        store.record_analysis(
            session, NOW - 90_000 + index, NOW - 80_000 + index,
            FixtureAnalysis(
                outcome="person_down", summary=f"Recorded candidate {index}.",
                observations=[{"predicate": "visible_person", "supported": True}],
            ),
            f"aggregate-{index}",
        )
    result = HybridQueryEngine(store).execute(QueryIntent(operation="count", limit=1))
    assert result["answer"] == "101 matching recorded incidents."


def test_structured_list_reports_exact_total_and_paginates() -> None:
    store = Store()
    session = store.create_session(NOW - 100_000)
    for index in range(105):
        store.record_analysis(
            session, NOW - 90_000 + index, NOW - 80_000 + index,
            FixtureAnalysis(
                outcome="person_down", summary=f"Recorded candidate {index}.",
                observations=[{"predicate": "visible_person", "supported": True}],
            ),
            f"list-total-{index}",
        )
    result = HybridQueryEngine(store).execute(NaturalQueryRequest(
        question="Show person down incidents", now_utc_ms=NOW
    ))
    assert result["result_count"] == 20
    assert result["matching_total"] == 105
    assert result["candidate_scan_count"] == 20
    assert result["candidate_scan_truncated"] is False
    assert result["answer"] == "Showing 20 of 105 matching recorded incidents."

    next_page = HybridQueryEngine(store).execute(NaturalQueryRequest(
        question="Show person down incidents", now_utc_ms=NOW, limit=20, offset=20
    ))
    assert next_page["result_count"] == 20
    assert next_page["matching_total"] == 105
    assert next_page["filters"]["offset"] == 20
    assert next_page["events"][0]["event_id"] != result["events"][0]["event_id"]

    grouped = HybridQueryEngine(store).execute(NaturalQueryRequest(
        question="Group person down incidents by category", now_utc_ms=NOW
    ))
    assert grouped["answer"] == {"person_down": 105}


def test_structured_query_lists_vlm_supported_events_before_other_results() -> None:
    store = Store()
    session = store.create_session(NOW - 100_000)
    event_ids = []
    for index, timestamp in enumerate((NOW - 10_000, NOW - 20_000, NOW - 30_000)):
        event_id = store.record_analysis(
            session, timestamp, timestamp + 1_000,
            FixtureAnalysis(
                outcome="person_down", summary=f"Person-down event {index}.",
                observations=[{"predicate": "visible_person", "supported": True}],
            ),
            f"verification-order-{index}",
        )
        event_ids.append(event_id)
        if index < 3:
            row = store.get_event_row(event_id)
            window_id = f"verification-order-vlm-{index}"
            verdict = "failed" if index == 0 else "supported"
            store.connection.execute(
                """INSERT INTO analysis_windows
                   (window_id,session_id,start_utc_ms,end_utc_ms,frame_count,reason,status,
                    model_outcome,model_id,model_revision,prompt_version,observations_json,timings_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    window_id, session, timestamp, timestamp + 1_000, 4,
                    "vlm_verification", "failed" if verdict == "failed" else "completed",
                    "person_down", "fixture://vlm", "v1", "test",
                    json.dumps({"verdict": None if verdict == "failed" else verdict}), "{}",
                ),
            )
            window_ids = json.loads(row["window_ids_json"])
            window_ids.append(window_id)
            store.connection.execute(
                "UPDATE events SET window_ids_json=? WHERE event_id=?",
                (json.dumps(window_ids), event_id),
            )
    store.connection.commit()

    engine = HybridQueryEngine(store)
    first_page = engine.execute(NaturalQueryRequest(
        question="Show person down incidents", now_utc_ms=NOW, limit=2,
    ))
    second_page = engine.execute(NaturalQueryRequest(
        question="Show person down incidents", now_utc_ms=NOW, limit=2, offset=2,
    ))

    assert [event["event_id"] for event in first_page["events"]] == [event_ids[1], event_ids[2]]
    assert [event["event_id"] for event in second_page["events"]] == [event_ids[0]]


class KeywordEmbedder:
    model_id = "fixture://keyword-embedder"
    model_revision = "v1"

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lowered = text.lower()
            vector = np.asarray([
                float("lying" in lowered or "prone" in lowered or "ground" in lowered),
                float("crowd" in lowered or "group" in lowered),
                float("entrance" in lowered or "doorway" in lowered),
                float("unusual" in lowered or "concerning" in lowered or "event id:" in lowered),
            ], dtype=np.float32)
            vectors.append(vector / np.linalg.norm(vector))
        return np.stack(vectors)


class FakeQwen:
    model_id = "fixture://qwen3-0.6b"
    model_revision = "v1"

    def __init__(self, *, malformed_route: bool = False, answer: str = "Grounded local answer."):
        self.malformed_route = malformed_route
        self.answer_text = answer
        self.answer_calls = 0

    def route(self, question: str, suggested_mode: str) -> ModelRoutingDecision:
        if self.malformed_route:
            raise ValueError("malformed JSON")
        return ModelRoutingDecision(retrieval_mode=suggested_mode, semantic_query=question)

    def answer(self, question: str, events: list[dict], deterministic_facts: str) -> str:
        self.answer_calls += 1
        return self.answer_text


def populated_store() -> tuple[Store, str, str]:
    store = Store()
    session = store.create_session(NOW - 60_000)
    person = store.record_analysis(
        session,
        NOW - 40_000,
        NOW - 35_000,
        FixtureAnalysis(
            outcome="person_down",
            summary="A person remained lying on the ground near the entrance.",
            observations=[{"predicate": "prone_posture", "supported": True}],
        ),
        "rag-person",
    )
    crowd = store.record_analysis(
        session,
        NOW - 20_000,
        NOW - 15_000,
        FixtureAnalysis(
            outcome="crowd_gathering",
            summary="A group remained together in the marked lobby zone.",
            observations=[{"predicate": "people_in_zone", "supported": True}],
        ),
        "rag-crowd",
    )
    return store, person, crowd


def engine(store: Store, model: FakeQwen | None = None) -> HybridQueryEngine:
    return HybridQueryEngine(store, text_model=model, ranker=SemanticRanker(KeywordEmbedder()))


def test_structured_count_is_from_sqlite_and_does_not_call_llm() -> None:
    store, person, _ = populated_store()
    model = FakeQwen(answer="There are 99 incidents.")
    result = engine(store, model).execute(NaturalQueryRequest(
        question="How many person-down incidents were recorded?", now_utc_ms=NOW
    ))
    assert result["answer"] == "1 matching recorded incidents."
    assert result["retrieval_mode"] == "structured"
    assert result["citations"] == [{"event_id": person}]
    assert model.answer_calls == 0


def test_semantic_query_finds_descriptive_event_without_trusting_model_ids() -> None:
    store, person, crowd = populated_store()
    model = FakeQwen(answer="The model mentioned made-up-event, but citations are application-owned.")
    result = engine(store, model).execute(NaturalQueryRequest(
        question="Did anyone appear to be lying on the ground?", now_utc_ms=NOW
    ))
    assert result["retrieval_mode"] == "hybrid"
    assert result["events"][0]["event_id"] == person
    assert crowd not in {item["event_id"] for item in result["citations"]}
    assert result["citations"] == [{"event_id": person}]


def test_hybrid_filters_are_applied_before_semantic_ranking() -> None:
    store, person, _ = populated_store()
    result = engine(store).execute(NaturalQueryRequest(
        question="What unusual things happened on camera 1 this morning?", now_utc_ms=NOW
    ))
    assert result["retrieval_mode"] == "hybrid"
    assert result["filters"]["camera_ids"] == ["cam-1"]
    assert result["filters"]["start_utc_ms"] is not None
    assert result["retrieval"]["structured_candidates"] == 2
    assert person in {item["event_id"] for item in result["events"]}


def test_deleted_event_and_stale_cache_cannot_be_returned() -> None:
    store, person, crowd = populated_store()
    query_engine = engine(store)
    request = NaturalQueryRequest(
        question="What concerning incidents happened near the entrance?", now_utc_ms=NOW
    )
    first = query_engine.execute(request)
    assert person in {item["event_id"] for item in first["events"]}
    store.connection.execute("DELETE FROM events WHERE event_id=?", (person,))
    store.connection.commit()
    second = query_engine.execute(request)
    assert person not in {item["event_id"] for item in second["events"]}
    assert all(item["event_id"] == crowd for item in second["events"])


def test_malformed_model_route_falls_back_to_deterministic_plan() -> None:
    store, person, _ = populated_store()
    result = engine(store, FakeQwen(malformed_route=True)).execute(NaturalQueryRequest(
        question="Did anyone appear to be lying on the ground?", now_utc_ms=NOW
    ))
    assert result["retrieval_mode"] == "hybrid"
    assert result["events"][0]["event_id"] == person


def test_empty_retrieval_is_honest_and_unsupported_query_clarifies() -> None:
    store = Store()
    store.seed_camera()
    result = engine(store).execute(NaturalQueryRequest(
        question="Did anyone appear to be lying on the ground?", now_utc_ms=NOW
    ))
    assert result["events"] == []
    assert "does not establish that nothing happened" in result["answer"]
    with pytest.raises(QueryClarification):
        engine(store).execute(NaturalQueryRequest(question="Was the building safe?", now_utc_ms=NOW))


def test_system_dismissed_events_keep_existing_default_exclusion() -> None:
    store, person, crowd = populated_store()
    store.connection.execute("UPDATE events SET decision_state='dismissed' WHERE event_id=?", (person,))
    store.connection.commit()
    result = engine(store).execute(NaturalQueryRequest(
        question="What concerning incidents involved a group of people near the entrance?", now_utc_ms=NOW
    ))
    assert person not in {item["event_id"] for item in result["events"]}
    assert crowd in {item["event_id"] for item in result["events"]}


def test_query_api_exposes_hybrid_provenance_and_real_citations(monkeypatch) -> None:
    from backend import main

    store, person, _ = populated_store()
    query_engine = engine(store, FakeQwen())
    monkeypatch.setattr(main, "get_query_engine", lambda: query_engine)
    with TestClient(main.app) as client:
        response = client.post("/api/query", json={
            "question": "Did anyone appear to be lying on the ground?",
            "timezone": "Asia/Kolkata",
            "now_utc_ms": NOW,
        })
    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval_mode"] == "hybrid"
    assert payload["model"]["id"] == "fixture://qwen3-0.6b"
    assert payload["retrieval"]["structured_candidates"] == 1
    assert payload["citations"] == [{"event_id": person}]
