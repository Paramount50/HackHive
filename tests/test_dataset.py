import hashlib

import pytest

from backend.dataset import ManifestEntry, validate_manifest


def entry(
    sample_id: str,
    source_id: str,
    split: str,
    categories=None,
    source_sha256: str | None = None,
) -> ManifestEntry:
    return ManifestEntry(
        sample_id=sample_id,
        source_id=source_id,
        source_sha256=source_sha256 or hashlib.sha256(source_id.encode()).hexdigest(),
        split=split,
        start_ms=0,
        end_ms=10_000,
        categories=categories or [],
        provenance="team-recorded fixture",
        license="team-owned demo footage",
    )


def test_manifest_summary_preserves_negatives_and_test_split() -> None:
    report = validate_manifest([
        entry("normal-1", "source-normal", "test"),
        entry("down-1", "source-down", "validation", ["person_down"]),
    ])
    assert report["samples"] == 2
    assert report["sources"] == 2
    assert report["negative_samples"] == 1
    assert report["categories"] == {"person_down": 1}
    assert report["warnings"] == []


def test_manifest_rejects_source_leakage() -> None:
    with pytest.raises(ValueError, match="source leakage"):
        validate_manifest([
            entry("clip-1", "same-video", "validation"),
            entry("clip-2", "same-video", "test"),
        ])


def test_manifest_rejects_identical_content_under_different_source_ids() -> None:
    repeated_hash = "a" * 64
    with pytest.raises(ValueError, match="identical source content"):
        validate_manifest([
            entry("clip-a", "named-copy-a", "validation", source_sha256=repeated_hash),
            entry("clip-b", "named-copy-b", "test", source_sha256=repeated_hash),
        ])


def test_manifest_rejects_duplicate_sample_ids() -> None:
    with pytest.raises(ValueError, match="duplicate sample_id"):
        validate_manifest([
            entry("duplicate", "source-a", "test"),
            entry("duplicate", "source-b", "test"),
        ])
