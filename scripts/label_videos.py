"""Create source-separated labels for the supplied video clips."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.contracts import Category


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_duration_ms(path: Path) -> int:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("video labeling requires opencv-python-headless") from exc
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or fps <= 0 or count <= 0:
            raise ValueError(f"cannot read duration for {path.name}")
        return round(count * 1000 / fps)
    finally:
        capture.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=Path("..") / "sample_videos")
    parser.add_argument("--output", type=Path, default=Path("evaluation") / "sample_manifest.jsonl")
    parser.add_argument("--provenance", required=True, help="How these videos were obtained")
    parser.add_argument("--license", required=True, help="Rights/permission basis for using the videos")
    args = parser.parse_args()
    if not args.videos.is_dir():
        parser.error(f"video directory does not exist: {args.videos}")
    videos = sorted(path for path in args.videos.iterdir()
                    if path.suffix.lower() in {".avi", ".mp4", ".mkv", ".mov"})
    if not videos:
        parser.error(f"no supported videos found in {args.videos}")

    valid_categories = {category.value for category in Category
                        if category in {
                            Category.PERSON_DOWN,
                            Category.CROWD_GATHERING,
                            Category.SUSPECTED_TAILGATING,
                        }}
    entries = []
    for path in videos:
        duration_ms = video_duration_ms(path)
        print(f"\nReview this complete clip in a video player: {path.resolve()}")
        split = input("Split (train/validation/test; keep all windows from this source together): ").strip()
        while split not in {"train", "validation", "test"}:
            split = input("Enter train, validation, or test: ").strip()
        source_hash = file_sha256(path)
        source_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", path.stem)[:100]
        interval_count_text = input(
            "Number of clear labeled intervals (0 = entire clip is a confirmed negative, "
            "s = skip/ambiguous): "
        ).strip().lower()
        if interval_count_text == "s":
            continue
        try:
            interval_count = int(interval_count_text)
        except ValueError:
            parser.error("interval count must be a non-negative integer or s")
        if interval_count < 0:
            parser.error("interval count must not be negative")
        if interval_count == 0:
            intervals = [(0, duration_ms, [])]
        else:
            intervals = []
            for index in range(interval_count):
                print(f"Interval {index + 1}/{interval_count}; valid categories: {', '.join(sorted(valid_categories))}")
                categories_text = input("Comma-separated category label(s): ").strip()
                categories = [item.strip() for item in categories_text.split(",") if item.strip()]
                if not categories or any(item not in valid_categories for item in categories):
                    parser.error("each interval needs one or more supported category labels")
                try:
                    start_ms = round(float(input("Start time in seconds: ").strip()) * 1000)
                    end_ms = round(float(input("End time in seconds: ").strip()) * 1000)
                except ValueError:
                    parser.error("interval times must be numeric seconds")
                if start_ms < 0 or end_ms <= start_ms or end_ms > duration_ms:
                    parser.error(f"interval must satisfy 0 <= start < end <= {duration_ms} ms")
                intervals.append((start_ms, end_ms, sorted(set(categories))))

        notes = input("Annotation notes / ambiguity (optional): ").strip()
        for index, (start_ms, end_ms, categories) in enumerate(intervals):
            entries.append({
                "sample_id": f"{source_id}-{index + 1:03d}",
                "source_id": source_id,
                "source_sha256": source_hash,
                "split": split,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "categories": categories,
                "provenance": args.provenance,
                "license": args.license,
                "notes": notes,
            })

    if not entries:
        print("No labels were recorded; no manifest was written.")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries),
        encoding="utf-8",
    )
    print(f"Wrote {len(entries)} labeled interval(s) to {args.output}")
    print("Validate with: python scripts/validate_manifest.py " + str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
