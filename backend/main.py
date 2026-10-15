from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from .contracts import FixtureAnalysis, QueryIntent, ReviewRequest, utc_now_ms
from .perception.rules import CandidateRules, Detection
from .storage import Store


app = FastAPI(title="HackHive Local Event Desk", version="0.1.0")
store = Store(Path("data") / "hackhive.sqlite")
store.seed_camera()


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>HackHive Event Desk</title>
<style>body{background:#0b1220;color:#f1f5f9;font:16px system-ui;margin:32px}
main{max-width:1100px;margin:auto}.panel{background:#111c2e;padding:20px;margin:16px 0;border-radius:10px}
button{padding:10px 14px;background:#60a5fa;border:0;border-radius:6px;cursor:pointer}
.muted{color:#b8c5d6}.event{border-top:1px solid #334155;padding:14px 0}</style></head>
<body><main><h1>HackHive local event desk</h1>
<div class="panel"><strong>Replay fixture</strong><p class="muted">CPU fixture mode · simulated access grants · Asia/Kolkata</p>
<button onclick="seed()">Run deterministic fixture</button> <button onclick="load()">Refresh events</button></div>
<div id="message" class="panel">No events loaded.</div><div id="events" class="panel"></div>
<script>
async function seed(){let r=await fetch('/api/fixtures/run',{method:'POST'});let x=await r.json();
document.getElementById('message').textContent=x.message;load()}
async function load(){let x=await (await fetch('/api/events')).json();document.getElementById('events').innerHTML=x.events.map(e=>
`<div class="event"><b>${e.category}</b> · ${e.review_status}<br>${e.summary}<br>
<span class="muted">System decision: ${e.decision_state} · Evidence score ${Math.round(e.confidence_score*4)}/4 · uncalibrated</span>
<button onclick="review('${e.event_id}','confirm')">Confirm</button>
<button onclick="review('${e.event_id}','reject')">Reject</button></div>`).join('')||'No matching recorded incidents.'}
async function review(id,action){await fetch('/api/events/'+id+'/reviews',{method:'POST',headers:{'content-type':'application/json'},
body:JSON.stringify({action,actor_label:'operator',note:'Reviewed in local dashboard',request_key:id+'-'+action})});load()}
load()
</script></main></body></html>"""


@app.get("/api/status")
def status() -> dict[str, Any]:
    return {
        "source": {"kind": "file", "mode": "fixture", "health": "ready", "fixture": True},
        "analysis": {"mode": "deterministic_fixture", "gpu_required": False, "degraded": True,
                     "scheduler": "bounded"},
        "policy": "Simulated access grants",
        "timezone": "Asia/Kolkata",
    }


@app.get("/api/cameras")
def cameras() -> list[dict[str, Any]]:
    row = store.connection.execute(
        "SELECT camera_id,name,source_kind,display_timezone,policy_version FROM cameras"
    ).fetchall()
    return [dict(item) for item in row]


@app.get("/api/events")
def events(
    category: str | None = Query(default=None),
    start_utc_ms: int | None = None,
    end_utc_ms: int | None = None,
) -> dict[str, Any]:
    categories = [category] if category else []
    intent = QueryIntent(categories=categories, start_utc_ms=start_utc_ms, end_utc_ms=end_utc_ms)
    result = store.list_events(intent)
    return {"events": result, "count": len(result), "timezone": "Asia/Kolkata"}


@app.get("/api/events/{event_id}")
def event(event_id: str) -> dict[str, Any]:
    result = store.list_events(QueryIntent(limit=100))
    for item in result:
        if item["event_id"] == event_id:
            item["media"] = [
                dict(row) for row in store.connection.execute(
                    "SELECT * FROM media WHERE event_id=?", (event_id,)
                ).fetchall()
            ]
            return item
    raise HTTPException(status_code=404, detail="event not found")


@app.post("/api/events/{event_id}/reviews")
def review(event_id: str, request: ReviewRequest) -> dict[str, Any]:
    if not store.connection.execute("SELECT 1 FROM events WHERE event_id=?", (event_id,)).fetchone():
        raise HTTPException(status_code=404, detail="event not found")
    return store.review(event_id, request)


@app.post("/api/fixtures/run")
def run_fixture() -> dict[str, Any]:
    started = utc_now_ms() - 10_000
    session_id = store.create_session(started)
    analysis = FixtureAnalysis(
        outcome="suspected_tailgating",
        summary="Two people crossed the gate in the same direction after one simulated grant.",
        observations=[
            {"predicate": "distinct_same_direction_passages", "supported": True, "frame_indices": [1, 2]},
            {"predicate": "temporal_proximity", "supported": True, "frame_indices": [2, 3]},
            {"predicate": "gate_visibility", "supported": True, "frame_indices": [0, 1, 2, 3]},
        ],
        limitations=["Visual crossings do not establish authorization; grant evidence is simulated."],
        frame_indices=[0, 1, 2, 3],
    )
    event_id = store.record_analysis(session_id, started + 4_000, started + 6_000, analysis, "fixture-gate-tailgate-1")
    if event_id and not store.connection.execute("SELECT 1 FROM media WHERE event_id=?", (event_id,)).fetchone():
        store.add_media(event_id)
    return {"session_id": session_id, "event_id": event_id, "fixture": True,
            "message": "Deterministic fixture recorded; this is not real inference."}


@app.post("/api/fixtures/perception-run")
def run_perception_fixture() -> dict[str, Any]:
    """Run detections through the CPU rules before persisting the event."""
    started = utc_now_ms() - 10_000
    session_id = store.create_session(started)
    frames = [
        [
            Detection(
                frame_id=index,
                source_pts_ms=started + index * 1_000,
                track_id=track,
                class_name="person",
                score=0.9,
                center_x=0.5,
                center_y=0.5,
                width=0.5,
                height=1.2,
                direction="entry",
            )
            for track in (("track-a",) if index == 0 else ("track-a", "track-b"))
        ]
        for index in range(3)
    ]
    candidates = CandidateRules(tailgating_window_ms=2_000).evaluate(frames)
    candidate = next(
        (item for item in candidates if item.category.value == "suspected_tailgating"), None
    )
    if candidate is None:
        raise HTTPException(status_code=422, detail="fixture did not produce a candidate")
    analysis = FixtureAnalysis(
        outcome=candidate.category.value,
        summary="Two people crossed the gate in the same direction within the configured window.",
        observations=[
            {"predicate": "distinct_same_direction_passages", "supported": True, "frame_indices": [1, 2]},
            {"predicate": "temporal_proximity", "supported": True, "frame_indices": [1, 2]},
        ],
        limitations=["Candidate was generated by deterministic CPU fixture detections."],
        frame_indices=[0, 1, 2],
    )
    event_id = store.record_analysis(
        session_id, candidate.start_pts_ms, candidate.trigger_pts_ms, analysis,
        f"perception-fixture:{session_id}",
    )
    return {
        "session_id": session_id,
        "event_id": event_id,
        "candidate": candidate.reason,
        "fixture": True,
        "message": "CPU detections produced a candidate and stored event; this is not real inference.",
    }


@app.post("/api/query")
def query(intent: QueryIntent) -> dict[str, Any]:
    result = store.list_events(intent)
    if intent.operation == "count":
        answer = f"{len(result)} matching recorded incidents."
    elif intent.operation == "group":
        grouped: dict[str, int] = {}
        for item in result:
            key = item["camera_name"] if intent.group_by == "camera" else item["category"]
            grouped[key] = grouped.get(key, 0) + 1
        answer = grouped
    else:
        answer = "No matching recorded incidents." if not result else f"{len(result)} matching recorded incidents."
    return {
        "answer": answer,
        "filters": intent.model_dump(mode="json"),
        "timezone": "Asia/Kolkata",
        "events": result,
        "citations": [{"event_id": item["event_id"]} for item in result],
        "coverage": {"fixture": True, "warning": "Coverage is limited to recorded fixture sessions."},
    }
