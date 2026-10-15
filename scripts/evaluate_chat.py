"""Evaluate routing, filters, retrieval and citations against a labeled question set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.chat import HybridQueryEngine, MiniLMEmbedder, Qwen3LocalModel, SemanticRanker
from backend.contracts import NaturalQueryRequest
from backend.query import QueryClarification
from backend.storage import Store


def subset(expected: dict, actual: dict) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path)
    parser.add_argument("--database", type=Path, default=Path("data/hackhive.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("reports/chat-evaluation.json"))
    parser.add_argument("--with-models", action="store_true")
    parser.add_argument("--chat-dir", type=Path, default=Path("models/qwen3-0.6b"))
    parser.add_argument("--embedding-dir", type=Path, default=Path("models/all-minilm-l6-v2"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    cases = json.loads(args.questions.read_text("utf-8"))
    store = Store(args.database)
    text_model = Qwen3LocalModel(args.chat_dir, device=args.device) if args.with_models else None
    ranker = SemanticRanker(MiniLMEmbedder(args.embedding_dir)) if args.with_models else None
    engine = HybridQueryEngine(store, text_model=text_model, ranker=ranker)
    rows = []
    for case in cases:
        started = perf_counter()
        try:
            response = engine.execute(NaturalQueryRequest(
                question=case["question"],
                timezone=case.get("timezone", "Asia/Kolkata"),
                now_utc_ms=case.get("now_utc_ms"),
            ))
        except QueryClarification as exc:
            response = {"needs_clarification": True, "answer": str(exc), "events": [], "citations": []}
        returned_ids = {item["event_id"] for item in response.get("events", [])}
        citation_ids = {item["event_id"] for item in response.get("citations", [])}
        expected_ids = case.get("expected_event_ids")
        rows.append({
            "id": case["id"],
            "question": case["question"],
            "expected_mode": case.get("expected_mode"),
            "actual_mode": response.get("retrieval_mode"),
            "routing_correct": (
                response.get("retrieval_mode") == case["expected_mode"]
                if case.get("expected_mode") is not None else None
            ),
            "filters_correct": subset(case.get("expected_filters", {}), response.get("filters", {})),
            "retrieval_correct": (returned_ids == set(expected_ids) if expected_ids is not None else None),
            "citation_valid": citation_ids.issubset(returned_ids) and all(store.get_event(item) for item in citation_ids),
            "clarification_correct": response.get("needs_clarification", False) == case.get("expect_clarification", False),
            "returned_event_ids": sorted(returned_ids),
            "latency_ms": round((perf_counter() - started) * 1000, 2),
            "response": response,
        })
    measured = [row for row in rows if row["retrieval_correct"] is not None]
    summary = {
        "cases": len(rows),
        "labeled_retrieval_cases": len(measured),
        "routing_correct": sum(row["routing_correct"] is True for row in rows),
        "filters_correct": sum(row["filters_correct"] is True for row in rows),
        "retrieval_correct": sum(row["retrieval_correct"] is True for row in measured),
        "citations_valid": sum(row["citation_valid"] is True for row in rows),
        "clarifications_correct": sum(row["clarification_correct"] is True for row in rows),
        "models_enabled": args.with_models,
        "warning": "Unlabeled expected_event_ids are excluded; no accuracy is inferred from them.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "cases": rows}, indent=2), "utf-8")
    print(json.dumps(summary, indent=2))
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
