from backend.perception.rules import CandidateCategory, CandidateRules, Detection


def person(pts: int, track: str, *, horizontal: bool = True, in_zone: bool = False) -> Detection:
    return Detection(
        frame_id=pts // 1000, source_pts_ms=pts, track_id=track, class_name="person",
        score=0.9, center_x=0.5, center_y=0.5, width=1.4 if horizontal else 0.5,
        height=0.5 if horizontal else 1.4, in_zone=in_zone,
    )


def test_person_down_requires_persistence() -> None:
    rules = CandidateRules()
    assert not rules.evaluate([[person(0, "p1")], [person(1_000, "p1")]])
    result = rules.evaluate([[person(0, "p1")], [person(2_000, "p1")]])
    assert result[0].category == CandidateCategory.PERSON_DOWN


def test_crowd_requires_count_and_duration() -> None:
    frames = [[person(0, f"p{i}", in_zone=True) for i in range(4)],
              [person(3_000, f"p{i}", in_zone=True) for i in range(4)]]
    result = CandidateRules().evaluate(frames)
    crowd = next(item for item in result if item.category == CandidateCategory.CROWD_GATHERING)
    assert len(crowd.track_ids) == 4


def test_tailgating_requires_distinct_close_entries() -> None:
    first = person(1_000, "p1")
    second = person(2_500, "p2")
    first = Detection(**{**first.__dict__, "direction": "entry"})
    second = Detection(**{**second.__dict__, "direction": "entry"})
    result = CandidateRules().evaluate([[first], [second]])
    assert result[0].category == CandidateCategory.SUSPECTED_TAILGATING
