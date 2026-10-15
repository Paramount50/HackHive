from __future__ import annotations

import json

from backend.chat import (
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION,
    QWEN_MODEL_ID,
    QWEN_MODEL_REVISION,
)
from scripts.setup_chat_models import write_provenance


def test_chat_model_provenance_files_are_exact(tmp_path) -> None:
    write_provenance(tmp_path, "chat.json", QWEN_MODEL_ID, QWEN_MODEL_REVISION)
    write_provenance(tmp_path, "embedding.json", EMBEDDING_MODEL_ID, EMBEDDING_MODEL_REVISION)
    assert json.loads((tmp_path / "chat.json").read_text("utf-8")) == {
        "model_id": "Qwen/Qwen3-0.6B",
        "revision": QWEN_MODEL_REVISION,
    }
    assert json.loads((tmp_path / "embedding.json").read_text("utf-8")) == {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": EMBEDDING_MODEL_REVISION,
    }
