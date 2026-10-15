from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from .contracts import Category, NaturalQueryRequest, QueryIntent, utc_now_ms


class QueryClarification(ValueError):
    pass


CATEGORY_TERMS: dict[Category, tuple[str, ...]] = {
    Category.PERSON_DOWN: ("person down", "fallen person", "lying on the floor", "prone person"),
    Category.CROWD_GATHERING: ("crowd", "gathering", "too many people"),
    Category.SUSPECTED_TAILGATING: ("tailgating", "tailgate", "close-following entry"),
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
    if "today" in text:
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


def parse_natural_query(request: NaturalQueryRequest) -> QueryIntent:
    text = " ".join(request.question.lower().split())
    categories = [category for category, terms in CATEGORY_TERMS.items() if any(term in text for term in terms)]
    unsupported = [term for term in ("weapon", "gun", "knife", "fight", "fire", "smoke") if term in text]
    if unsupported and not categories:
        raise QueryClarification(
            f"Unsupported category '{unsupported[0]}'. Choose person down, crowd gathering, or suspected tailgating."
        )

    if re.search(r"\b(how many|count|number of)\b", text):
        operation = "count"
        group_by = None
    elif re.search(r"\b(by|per)\s+camera\b", text):
        operation, group_by = "group", "camera"
    elif re.search(r"\b(by|per)\s+categor(?:y|ies)\b", text):
        operation, group_by = "group", "category"
    elif re.search(r"\b(show|list|find|which|what|events?|incidents?)\b", text):
        operation, group_by = "list", None
    else:
        raise QueryClarification("Ask to show, list, count, or group recorded incidents.")

    camera_ids: list[str] = []
    camera = re.search(r"\b(?:camera|cam)[-\s_]*(\d+)\b", text)
    if camera:
        camera_ids = [f"cam-{int(camera.group(1))}"]
    start, end = _time_range(text, request)
    return QueryIntent(
        operation=operation,
        categories=categories,
        camera_ids=camera_ids,
        start_utc_ms=start,
        end_utc_ms=end,
        group_by=group_by,
    )
