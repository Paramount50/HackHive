import json
import sys

from scripts import label_videos


def test_labeling_tool_includes_aggression_and_skips_duplicate_sources(
    tmp_path, monkeypatch, capsys
) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "copy-a.avi").write_bytes(b"same-source")
    (videos / "copy-b.avi").write_bytes(b"same-source")
    (videos / "fight.avi").write_bytes(b"fight-source")
    output = tmp_path / "manifest.jsonl"
    answers = iter([
        "train", "0", "confirmed normal",
        "test", "1", "aggression_fight", "0.5", "2.0", "clear fight",
    ])

    monkeypatch.setattr(label_videos, "video_duration_ms", lambda _path: 3_000)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setattr(sys, "argv", [
        "label_videos.py",
        "--videos", str(videos),
        "--output", str(output),
        "--provenance", "team-reviewed sample",
        "--license", "approved for evaluation",
    ])

    assert label_videos.main() == 0

    entries = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [entry["categories"] for entry in entries] == [[], ["aggression_fight"]]
    assert entries[1]["start_ms"] == 500
    assert entries[1]["end_ms"] == 2_000
    assert "Skipping byte-identical duplicate copy-b.avi" in capsys.readouterr().out
