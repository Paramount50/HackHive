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
    result = rules.evaluate([[person(pts, "p1")] for pts in (0, 1000, 2000)])
    assert result[0].category == CandidateCategory.PERSON_DOWN


def test_crowd_requires_count_and_duration() -> None:
    frames = [[person(pts, f"p{i}", in_zone=True) for i in range(4)] for pts in (0, 1000, 2000, 3000)]
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


def test_posture_persistence_tolerates_a_short_missing_detection():
    rules = CandidateRules()
    frames = [
        [person(0, "p1")], [person(1000, "p1")], [], [person(2000, "p1")],
    ]
    result = rules.evaluate(frames)
    person_down = next(item for item in result if item.category == CandidateCategory.PERSON_DOWN)
    assert (person_down.start_pts_ms, person_down.trigger_pts_ms) == (0, 2000)


def test_posture_persistence_resets_on_explicit_upright_observation_and_long_gap():
    rules = CandidateRules()
    assert not rules.evaluate([
        [person(0, "p1")],
        [person(1000, "p1", horizontal=False)],
        [person(2000, "p1")],
    ])
    assert not rules.evaluate([[person(0, "p1")], [person(5000, "p1")]])


def test_crowd_duration_resets_when_count_drops():
    frames = [[person(pts, f"p{i}", horizontal=False, in_zone=True) for i in range(count)]
              for pts, count in ((0, 4), (1000, 4), (2000, 1), (3000, 4), (4000, 4))]
    assert not CandidateRules().evaluate(frames)


def test_later_tailgating_pair_is_not_hidden_by_first_pair():
    frames = [[Detection(**{**person(pts, track).__dict__, "direction": "entry"})]
              for pts, track in ((0, "a"), (1000, "b"), (5000, "c"), (6000, "d"))]
    result = CandidateRules().evaluate(frames)
    assert [item.trigger_pts_ms for item in result] == [1000, 6000]
