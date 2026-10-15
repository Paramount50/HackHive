"""Download pinned Qwen3/MiniLM weights and optionally run a local smoke test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.chat import (
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION,
    QWEN_MODEL_ID,
    QWEN_MODEL_REVISION,
    MiniLMEmbedder,
    Qwen3LocalModel,
)


def write_provenance(path: Path, filename: str, model_id: str, revision: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / filename).write_text(
        json.dumps({"model_id": model_id, "revision": revision}, indent=2),
        encoding="utf-8",
    )


def download(chat_dir: Path, embedding_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        QWEN_MODEL_ID,
        revision=QWEN_MODEL_REVISION,
        local_dir=chat_dir,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
        max_workers=2,
    )
    write_provenance(chat_dir, "hackhive-chat-model.json", QWEN_MODEL_ID, QWEN_MODEL_REVISION)
    snapshot_download(
        EMBEDDING_MODEL_ID,
        revision=EMBEDDING_MODEL_REVISION,
        local_dir=embedding_dir,
        allow_patterns=[
            "*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "1_Pooling/*"
        ],
        max_workers=2,
    )
    write_provenance(
        embedding_dir,
        "hackhive-embedding-model.json",
        EMBEDDING_MODEL_ID,
        EMBEDDING_MODEL_REVISION,
    )


def smoke(chat_dir: Path, embedding_dir: Path, device: str) -> dict:
    started = perf_counter()
    model = Qwen3LocalModel(chat_dir, device=device, max_new_tokens=96)
    route = model.route("What unusual incidents happened on camera 1 this morning?", "hybrid")
    routed = perf_counter()
    embedder = MiniLMEmbedder(embedding_dir)
    vectors = embedder.encode([
        "someone lying on the ground",
        "a person remained prone near the entrance",
    ])
    embedded = perf_counter()
    answer = model.answer(
        "What happened?",
        [{
            "event_id": "smoke-event",
            "camera_id": "cam-1",
            "category": "person_down",
            "start_utc_ms": 1,
            "end_utc_ms": 2,
            "summary": "A person remained prone near the entrance.",
            "predicates": [{"name": "prone_posture", "supported": True}],
            "limitations": ["Synthetic smoke record; not model accuracy evidence."],
            "authorization_state": "unknown",
            "review_status": "unreviewed",
        }],
        "Application retrieval returned 1 synthetic smoke record.",
    )
    finished = perf_counter()
    return {
        "status": "completed",
        "device": device,
        "chat_model": {"id": QWEN_MODEL_ID, "revision": QWEN_MODEL_REVISION},
        "embedding_model": {"id": EMBEDDING_MODEL_ID, "revision": EMBEDDING_MODEL_REVISION},
        "route": route.model_dump(),
        "embedding_shape": list(vectors.shape),
        "answer": answer,
        "timings_ms": {
            "route_and_chat_load": round((routed - started) * 1000, 2),
            "embedding_and_load": round((embedded - routed) * 1000, 2),
            "answer": round((finished - embedded) * 1000, 2),
            "total": round((finished - started) * 1000, 2),
        },
        "validation_scope": "runtime_only_not_retrieval_accuracy",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download both pinned model snapshots.")
    parser.add_argument("--smoke", action="store_true", help="Run local routing, embedding and answer generation.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--chat-dir", type=Path, default=Path("models/qwen3-0.6b"))
    parser.add_argument("--embedding-dir", type=Path, default=Path("models/all-minilm-l6-v2"))
    args = parser.parse_args()
    if not args.download and not args.smoke:
        parser.error("choose --download, --smoke, or both")
    if args.download:
        download(args.chat_dir, args.embedding_dir)
        print(f"Downloaded pinned chat model to {args.chat_dir}")
        print(f"Downloaded pinned embedding model to {args.embedding_dir}")
    if args.smoke:
        print(json.dumps(smoke(args.chat_dir, args.embedding_dir, args.device), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
