from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter, process_time
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.ingest.replay import VideoReplay


def benchmark(path: Path, sample_fps: float) -> dict:
    results = {}
    for strategy in ("sequential", "grab"):
        with TemporaryDirectory(prefix=f"hackhive-{strategy}-") as directory:
            replay = VideoReplay(
                path, "benchmark-camera", f"benchmark-{strategy}", 0, directory,
                sample_fps=sample_fps, pace=False, decode_strategy=strategy,
            )
            wall_start, cpu_start = perf_counter(), process_time()
            packets = list(replay.frames())
            results[strategy] = {
                "wall_seconds": perf_counter() - wall_start,
                "cpu_seconds": process_time() - cpu_start,
                "emitted_frames": len(packets),
                "source_pts_ms": [packet.source_pts_ms for packet in packets],
            }
    results["comparable_output"] = (
        results["sequential"]["emitted_frames"] == results["grab"]["emitted_frames"]
        and results["sequential"]["source_pts_ms"] == results["grab"]["source_pts_ms"]
    )
    results["note"] = (
        "grab/retrieve may avoid image conversion for discarded frames, but compressed codecs may still "
        "decode internally. Treat timing as specific to this file, codec, backend, and machine."
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare HackHive OpenCV sampling strategies")
    parser.add_argument("video", type=Path)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = benchmark(args.video, args.sample_fps)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
