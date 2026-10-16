"""Download once, then measure local four-frame requests on unlabelled videos."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.inference.hardware import hardware_report
from backend.inference.contracts import VisualObservation, verification_verdict
from backend.inference.smolvlm import SmolVLMWorker, model_config


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
    parser.add_argument("--model", choices=("500m", "2.2b"), default="500m")
    parser.add_argument("--candidate", choices=(
        "person_down", "crowd_gathering", "suspected_tailgating",
        "aggression_fight", "fire_smoke", "intrusion_unauthorized_access",
    ))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--videos", type=Path)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--window-manifest", type=Path,
                        help="JSON manifest of source/fraction windows shared across variant runs")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--unconstrained", action="store_true", help="Diagnostic prompt-only comparison")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = model_config(args.model)
    model_dir = args.model_dir or config.default_dir
    if args.hardware_only:
        print(json.dumps(hardware_report(), indent=2))
        return 0
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot_download(config.model_id, revision=config.revision, local_dir=model_dir,
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"], max_workers=2)
        (model_dir / "hackhive-model.json").write_text(
            json.dumps({"model_id": config.model_id, "revision": config.revision}, indent=2), encoding="utf-8")
        print(f"Downloaded pinned {args.model} model to {model_dir}")
        return 0
    if not args.videos or not args.videos.is_dir() or not 1 <= args.requests <= 100:
        parser.error("provide --videos directory and --requests between 1 and 100")
    videos = sorted(path for path in args.videos.iterdir() if path.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov"})
    if not videos:
        parser.error("no video files found")
    output = args.output or Path("reports") / datetime.now(timezone.utc).strftime("vlm-%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    rounds = max(1, math.ceil(args.requests / len(videos)))
    if args.window_manifest and args.window_manifest.is_file():
        window_specs = json.loads(args.window_manifest.read_text(encoding="utf-8"))
        if len(window_specs) != args.requests or any(
            item.get("source") not in {path.name for path in videos}
            or not 0 <= item.get("fraction", -1) <= 1 for item in window_specs
        ):
            raise ValueError("window manifest does not match this video directory and request count")
    else:
        window_specs = [
            {"source": videos[position % len(videos)].name,
             "fraction": (position // len(videos) + 1) / (rounds + 1)}
            for position in range(args.requests)
        ]
        if args.window_manifest:
            args.window_manifest.write_text(json.dumps(window_specs, indent=2), encoding="utf-8")
    manifest = []
    for path in videos:
        with path.open("rb") as source:
            manifest.append({"file": path.name, "sha256": hashlib.file_digest(source, "sha256").hexdigest()})
    (output / "sources.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "hardware.json").write_text(json.dumps(hardware_report(), indent=2), encoding="utf-8")
    worker = SmolVLMWorker(model_dir, variant=args.model, max_new_tokens=args.max_new_tokens,
                           constrained=not args.unconstrained)
    try:
        metadata = worker.load()
    except Exception as exc:
        (output / "failure.json").write_text(json.dumps({
            "stage": "load", "error": str(exc), "variant": args.model,
            "model_id": config.model_id, "revision": config.revision,
        }, indent=2), encoding="utf-8")
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
            spec = window_specs[position]
            path = args.videos / spec["source"]
            fraction = spec["fraction"]
            try:
                images, pts, indices = sample_window(path, fraction)
                if args.candidate:
                    result = worker.analyze_candidate(images, pts, args.candidate)
                else:
                    result = worker.analyze(images, pts)
                if result.get("observation") is not None and args.candidate:
                    observation = VisualObservation.model_validate(result["observation"])
                    result["verification_verdict"] = verification_verdict(
                        observation, args.candidate
                    )
                result["source_frame_indices"] = indices
            except Exception as exc:
                result = {"status": "failed", "error_code": "input_failed", "error": str(exc)}
            result.update(request_index=index, warmup=warmup, source=path.name,
                          mode="candidate_verification" if args.candidate else "offline_smoke",
                          model_variant=args.model, candidate_category=args.candidate)
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            if not warmup:
                results.append(result)
            print(f"{'warmup' if warmup else index}: {path.name}: {result['status']} {result.get('error_code') or ''}", flush=True)
            if result.get("error_code") == "gpu_out_of_memory":
                break
    valid = sum(item["status"] == "completed" for item in results)
    outcomes = Counter(item["observation"]["outcome"] for item in results
                       if item.get("observation") is not None)
    timings = [item["timings_ms"]["total"] for item in results if "timings_ms" in item]
    summary = {"mode": "candidate_verification" if args.candidate else "offline_smoke",
               "variant": args.model, "model_id": config.model_id,
               "revision": config.revision, "candidate_category": args.candidate,
               "requested": args.requests, "attempted": len(results),
               "valid_outputs": valid, "failed_outputs": len(results)-valid,
               "valid_output_rate": valid/len(results) if results else None,
               "outcome_counts": dict(outcomes),
               "unknown_outputs": outcomes["unknown"],
               "quality_validated": False,
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
