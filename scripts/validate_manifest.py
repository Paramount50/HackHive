from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.dataset import ManifestEntry, validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a HackHive JSONL dataset manifest")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    entries = [
        ManifestEntry.model_validate_json(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(json.dumps(validate_manifest(entries), indent=2))


if __name__ == "__main__":
    main()
