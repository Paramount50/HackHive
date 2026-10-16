from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta, timezone

from .contracts import Category, NaturalQueryRequest, QueryIntent, utc_now_ms


class QueryClarification(ValueError):
    pass


CATEGORY_TERMS: dict[Category, tuple[str, ...]] = {
    Category.PERSON_DOWN: (
        "person down", "fallen person", "someone fell", "person fell",
        "lying on the floor", "on the ground", "prone person", "person lying",
    ),
    Category.CROWD_GATHERING: (
        "crowd", "gathering", "people gathering", "group of people",
        "too many people", "large group",
    ),
    Category.SUSPECTED_TAILGATING: (
        "tailgating", "tailgate", "close-following entry",
        "following through", "followed through the door", "multiple people entered after one",
    ),
    Category.AGGRESSION_FIGHT: (
        "aggression", "aggressive", "fight", "fighting", "assault",
        "physical confrontation", "people fighting",
    ),
    Category.FIRE_SMOKE: (
        "fire", "smoke", "flames", "burning", "smoke alarm",
    ),
    Category.INTRUSION_UNAUTHORIZED_ACCESS: (
        "intrusion", "intruder", "unauthorized access", "unauthorised access",
        "break in", "break-in", "trespassing", "unapproved entry",
    ),
}


def _time_range(text: str, request: NaturalQueryRequest) -> tuple[int | None, int | None]:
    if request.timezone == "Asia/Kolkata":
        zone = timezone(timedelta(hours=5, minutes=30), name="Asia/Kolkata")
    elif request.timezone == "UTC":
        zone = timezone.utc
    else:
        raise QueryClarification(f"Unsupported timezone: {request.timezone}")
    now_ms = request.now_utc_ms if request.now_utc_ms is not None else utc_now_ms()
    now = datetime.fromtimestamp(now_ms / 1000, timezone.utc)
    local_now = now.astimezone(zone)
    explicit_range = re.search(
        r"\b(?:between|from)\s+(\d{1,2}):(\d{2})\s+(?:and|to)\s+"
        r"(\d{1,2}):(\d{2})\b", text
    )
    explicit_date = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", text)
    if explicit_range:
        has_video = re.search(r"\b[\w.-]+\.(?:mp4|avi|mkv|mov)\b", text, re.IGNORECASE)
        if explicit_date is None and has_video is not None:
            return None, None
        if explicit_date is None:
            raise QueryClarification("Add a date to the clock range, for example 'between 09:00 and 10:00 on 2026-10-16'.")
        try:
            selected_date = date(*(int(value) for value in explicit_date.groups()))
            start_clock = time(int(explicit_range.group(1)), int(explicit_range.group(2)))
            end_clock = time(int(explicit_range.group(3)), int(explicit_range.group(4)))
        except ValueError as exc:
            raise QueryClarification("The date or clock range is invalid.") from exc
        start = datetime.combine(selected_date, start_clock, zone)
        end = datetime.combine(selected_date, end_clock, zone)
        if end <= start:
            raise QueryClarification("The end clock time must be later than the start clock time.")
    elif explicit_date and re.search(r"\b\d{1,2}:\d{2}\b", text):
        raise QueryClarification("Use a complete clock range, for example 'between 09:00 and 10:00 on 2026-10-16'.")
    elif explicit_date:
        try:
            selected_date = date(*(int(value) for value in explicit_date.groups()))
        except ValueError as exc:
            raise QueryClarification("The calendar date is invalid.") from exc
        start = datetime.combine(selected_date, time.min, zone)
        end = start + timedelta(days=1)
    elif re.search(r"\b(?:at|around)\s+\d{1,2}:\d{2}\b", text):
        raise QueryClarification("Clock-time filters need a date; use 'between HH:MM and HH:MM on YYYY-MM-DD'.")
    elif re.search(r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
                   r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
                   r"dec(?:ember)?)\s+\d{1,2}(?:,?\s+20\d{2})?\b", text):
        raise QueryClarification("Use an ISO date such as 2026-10-16 so the date filter is unambiguous.")
    elif "this morning" in text:
        start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        noon = start + timedelta(hours=12)
        end = min(local_now, noon)
        if end <= start:
            end = start + timedelta(minutes=1)
    elif "today" in text:
        start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
    elif "yesterday" in text:
        end = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=1)
    else:
        relative = re.search(r"(?:last|past)\s+(\d{1,3})\s*(minute|minutes|hour|hours)", text)
        if not relative:
            return None, None
        amount = int(relative.group(1))
        if amount < 1:
            raise QueryClarification("The relative time amount must be positive.")
        delta = timedelta(minutes=amount) if relative.group(2).startswith("minute") else timedelta(hours=amount)
        start, end = now - delta, now
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def _extract_source_offsets(text: str) -> tuple[str | None, int | None, int | None]:
    quoted_filename = re.search(
        r"""["']([^"']+\.(?:mp4|avi|mkv|mov))["']""", text, re.IGNORECASE
    )
    filename = quoted_filename or re.search(
        r"\b([\w.-]+\.(?:mp4|avi|mkv|mov))\b", text, re.IGNORECASE
    )
    offset = re.search(
        r"\b(?:from|between)\s+(\d{1,2}:\d{2}(?:\.\d{1,3})?)\s+"
        r"(?:to|and)\s+(\d{1,2}:\d{2}(?:\.\d{1,3})?)\b", text
    )
    mentions_offset = "offset" in text or "video time" in text or "recording time" in text
    if mentions_offset and (filename is None or offset is None):
        raise QueryClarification(
            "For a video-offset filter, include a recording filename and a range, "
            "for example 'entrance.mp4 from 00:42.400 to 00:49.200'."
        )
    if filename is None and offset is None:
        return None, None, None
    if filename is None and offset is not None:
        if mentions_offset or any("." in value for value in offset.groups()):
            raise QueryClarification("A recording-offset range must include a video filename.")
        return None, None, None
    if filename is None or offset is None:
        raise QueryClarification("A recording-offset range must include both a video filename and start/end offsets.")

    def to_ms(value: str) -> int:
        minute, seconds = value.split(":")
        return int(minute) * 60_000 + round(float(seconds) * 1000)

    start_ms, end_ms = (to_ms(value) for value in offset.groups())
    if end_ms <= start_ms:
        raise QueryClarification("The video-offset end must be later than its start.")
    filename_value = filename.group(1) if quoted_filename else filename.group(1)
    return filename_value.strip(), start_ms, end_ms


def parse_natural_query(request: NaturalQueryRequest) -> QueryIntent:
    original = request.question.lower()
    text = " ".join(original.replace("-", " ").split())
    if re.search(
        r"\b(?:vlm|model)\s+(?:supported|contradicted|inconclusive|verified|unverified)\b|"
        r"\b(?:supported|contradicted|inconclusive)\s+by\s+(?:the\s+)?(?:vlm|model)\b",
        text,
    ):
        raise QueryClarification(
            "VLM verdict filters are not supported yet. Human review status is separate; "
            "you can filter for human-confirmed, human-rejected, or unreviewed incidents."
        )
    if re.search(r"\b(?:not confirmed|unconfirmed|confirmed by the model)\b", text):
        raise QueryClarification(
            "Do you mean human review confirmation or VLM support? Those are separate statuses; "
            "try 'human-confirmed' or 'VLM-supported'."
        )
    categories = [category for category, terms in CATEGORY_TERMS.items() if any(term in text for term in terms)]
    unsupported = [term for term in ("weapon", "gun", "knife") if term in text]
    if unsupported:
        raise QueryClarification(
            f"Unsupported category '{unsupported[0]}'. Choose one of the six supported categories: "
            "person down, crowd gathering, suspected tailgating, aggression/fight, "
            "fire/smoke, or intrusion/unauthorized access."
        )
    if not categories and not re.search(r"\b(incidents?|events?|recorded|camera|footage|alerts?)\b", text):
        raise QueryClarification(
            "I can search recorded incidents, but I need a category, camera, or incident term."
        )

    if re.search(r"\b(how many|count|number of)\b", text):
        operation = "count"
        group_by = None
    elif re.search(r"\b(by|per)\s+camera\b", text):
        operation, group_by = "group", "camera"
    elif re.search(r"\b(by|per)\s+categor(?:y|ies)\b", text):
        operation, group_by = "group", "category"
    elif re.search(r"\b(show|list|find|which|what|any|are there|did|were|was|events?|incidents?)\b", text):
        operation, group_by = "list", None
    else:
        raise QueryClarification("Ask to show, list, count, or group recorded incidents.")

    camera_ids = list(dict.fromkeys(
        f"cam-{int(number)}"
        for number in re.findall(r"\b(?:camera|cam)[-\s_]*(\d+)\b", text)
    ))
    review_statuses = []
    if "confirmed by the model" in text or "vlm confirmed" in text or "model confirmed" in text:
        raise QueryClarification(
            "VLM support and human review confirmation are separate. Ask for 'VLM-supported' "
            "or 'human-confirmed' incidents."
        )
    if re.search(r"\b(?:human|operator|manually)\s+confirmed\b|\breviewed\s+and\s+confirmed\b", text):
        review_statuses.append("confirmed")
    elif re.search(r"\bconfirmed\b", text):
        review_statuses.append("confirmed")
    if re.search(r"\b(?:human|operator|manually)\s+rejected\b|\breviewed\s+and\s+rejected\b", text):
        review_statuses.append("rejected")
    if "unreviewed" in text or "not reviewed" in text:
        review_statuses.append("unreviewed")
    if re.search(r"\b(?:with|has|having)\s+(?:playable\s+)?(?:evidence|video|clip)\b", text):
        has_ready_media = True
    elif re.search(r"\b(?:without|no)\s+(?:playable\s+)?(?:evidence|video|clip)\b", text):
        has_ready_media = False
    else:
        has_ready_media = None
    start, end = _time_range(original, request)
    source_name, start_pts_ms, end_pts_ms = _extract_source_offsets(original)
    return QueryIntent(
        operation=operation,
        categories=categories,
        camera_ids=camera_ids,
        review_statuses=review_statuses,
        has_ready_media=has_ready_media,
        source_name=source_name,
        start_pts_ms=start_pts_ms,
        end_pts_ms=end_pts_ms,
        start_utc_ms=start,
        end_utc_ms=end,
        group_by=group_by,
    )
