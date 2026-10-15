from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.evaluation import LabeledEvent, PredictedEvent, score_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Score frozen HackHive event predictions")
    parser.add_argument("input", type=Path, help="JSON containing duration_ms, labels, and predictions")
    parser.add_argument("--output", type=Path, help="Optional metrics JSON path")
    parser.add_argument("--minimum-iou", type=float, default=0.1)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    metrics = score_events(
        [LabeledEvent(**item) for item in payload["labels"]],
        [PredictedEvent(**item) for item in payload["predictions"]],
        evaluated_duration_ms=payload["duration_ms"],
        minimum_iou=args.minimum_iou,
    )
    rendered = json.dumps(metrics, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
