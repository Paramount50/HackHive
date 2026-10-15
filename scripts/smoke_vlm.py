"""Download once, then measure local four-frame requests on unlabelled videos."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.inference.hardware import hardware_report
from backend.inference.smolvlm import MODEL_ID, MODEL_REVISION, SmolVLMWorker


def sample_window(path: Path, fraction: float):
    if not 0 <= fraction <= 1:
        raise ValueError("window fraction must be between zero and one")
    import cv2
    from PIL import Image
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or fps <= 0 or count < 4:
            raise ValueError(f"cannot sample four frames from {path.name}")
        span = min(count - 1, max(3, round(fps * 4)))
        start = round((count - span - 1) * fraction)
        indices = [start + round(span * index / 3) for index in range(4)]
        images, timestamps = [], []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"cannot decode frame {index} of {path.name}")
            images.append(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
            timestamps.append(round(index * 1000 / fps))
        return images, timestamps, indices
    finally:
        capture.release()


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Explicit network setup; no inference")
    parser.add_argument("--hardware-only", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=Path("models/smolvlm2-500m"))
    parser.add_argument("--videos", type=Path)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.hardware_only:
        print(json.dumps(hardware_report(), indent=2))
        return 0
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=args.model_dir,
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"], max_workers=2)
        (args.model_dir / "hackhive-model.json").write_text(
            json.dumps({"model_id": MODEL_ID, "revision": MODEL_REVISION}, indent=2), encoding="utf-8")
        print(f"Downloaded pinned model to {args.model_dir}")
        return 0
    if not args.videos or not args.videos.is_dir() or not 1 <= args.requests <= 100:
        parser.error("provide --videos directory and --requests between 1 and 100")
    videos = sorted(path for path in args.videos.iterdir() if path.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov"})
    if not videos:
        parser.error("no video files found")
    output = args.output or Path("reports") / datetime.now(timezone.utc).strftime("vlm-%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    manifest = []
    for path in videos:
        with path.open("rb") as source:
            manifest.append({"file": path.name, "sha256": hashlib.file_digest(source, "sha256").hexdigest()})
    (output / "sources.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "hardware.json").write_text(json.dumps(hardware_report(), indent=2), encoding="utf-8")
    worker = SmolVLMWorker(args.model_dir, max_new_tokens=args.max_new_tokens)
    try:
        metadata = worker.load()
    except Exception as exc:
        (output / "failure.json").write_text(json.dumps({"stage": "load", "error": str(exc)}, indent=2), encoding="utf-8")
        print(f"Model load failed: {exc}", file=sys.stderr)
        return 1
    (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "environment.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    results = []
    with (output / "requests.jsonl").open("w", encoding="utf-8") as stream:
        for index in range(args.requests + 1):
            warmup = index == 0
            position = max(0, index - 1)
            path = videos[position % len(videos)]
            rounds = max(1, math.ceil(args.requests / len(videos)))
            fraction = (position // len(videos) + 1) / (rounds + 1)
            try:
                images, pts, indices = sample_window(path, fraction)
                result = worker.analyze(images, pts)
                result["source_frame_indices"] = indices
            except Exception as exc:
                result = {"status": "failed", "error_code": "input_failed", "error": str(exc)}
            result.update(request_index=index, warmup=warmup, source=path.name, mode="offline_smoke")
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            if not warmup:
                results.append(result)
            print(f"{'warmup' if warmup else index}: {path.name}: {result['status']} {result.get('error_code') or ''}", flush=True)
            if result.get("error_code") == "gpu_out_of_memory":
                break
    valid = sum(item["status"] == "completed" for item in results)
    timings = [item["timings_ms"]["total"] for item in results if "timings_ms" in item]
    summary = {"mode": "offline_smoke", "requested": args.requests, "attempted": len(results),
               "valid_outputs": valid, "failed_outputs": len(results)-valid,
               "valid_output_rate": valid/len(results) if results else None,
               "request_ms_p50": percentile(timings, .5), "request_ms_p95": percentile(timings, .95),
               "peak_allocated_mib": max((item.get("peak_allocated_mib", 0) for item in results), default=0),
               "limitations": ["Unlabelled footage; no accuracy or recall claim.",
                               "Offline sampled windows; not real-time latency or a baseline/cascade comparison.",
                               "Warm-up excluded; malformed responses counted as failures.",
                               "PTS estimated from frame index and declared FPS; verify variable-frame-rate sources separately."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if valid == args.requests else 1


if __name__ == "__main__":
    raise SystemExit(main())
