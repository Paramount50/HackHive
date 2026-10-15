"""Download the pinned local XCLIP action model for offline replay analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huggingface_hub import snapshot_download

from backend.perception.action import MODEL_ID, MODEL_REVISION


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models/xclip-base-patch32"))
    args = parser.parse_args()
    path = snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=args.model_dir,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
        max_workers=2,
    )
    provenance = {"model_id": MODEL_ID, "revision": MODEL_REVISION}
    (Path(path) / "hackhive-action-model.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"Downloaded pinned local action model to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
