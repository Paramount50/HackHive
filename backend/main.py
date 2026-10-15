from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .chat import HybridQueryEngine, MiniLMEmbedder, Qwen3LocalModel, SemanticRanker
from .contracts import Category, FixtureAnalysis, NaturalQueryRequest, QueryIntent, ReplayStartRequest, ReviewRequest, ReviewStatus, utc_now_ms
from .ingest.manager import ReplayManager
from .inference.service import SemanticInferenceService, SemanticJob
from .inference.smolvlm import SmolVLMWorker
from .media import MediaExporter
from .orchestrator import EventOrchestrator
from .perception.action import XClipActionClassifier
from .perception.pipeline import PerceptionPipeline, ScenePolicy, YoloDetector
from .perception.rules import CandidateRules, Detection
from .query import QueryClarification
from .storage import Store
from .uploads import UploadValidationError, save_recording_upload


app = FastAPI(title="HackHive Local Event Desk", version="0.1.0")
database_path = Path("data") / "hackhive.sqlite"
store = Store(database_path)
store.seed_camera()
recordings_root = Path(os.environ.get("HACKHIVE_RECORDINGS_DIR", "recordings"))
max_upload_bytes = int(os.environ.get("HACKHIVE_MAX_UPLOAD_MIB", "250")) * 1024 * 1024
replay_manager = ReplayManager(recordings_root, Path("data") / "replay_frames")
media_root = (Path("data") / "media").resolve()
detector_weights = Path(os.environ.get("HACKHIVE_DETECTOR_WEIGHTS", "models/yolo11n.pt")).resolve()
action_model_dir = Path(os.environ.get("HACKHIVE_ACTION_MODEL_DIR", "models/xclip-base-patch32")).resolve()
action_classifier: XClipActionClassifier | None = None
action_classifier_lock = Lock()
vlm_model_dir = Path(os.environ.get("HACKHIVE_VLM_MODEL_DIR", "models/smolvlm2-500m")).resolve()
vlm_sampling_strategy = os.environ.get("HACKHIVE_VLM_SAMPLING", "swift").strip().lower()
if vlm_sampling_strategy not in {"swift", "uniform"}:
    raise ValueError("HACKHIVE_VLM_SAMPLING must be swift or uniform")
vlm_service: SemanticInferenceService | None = None
vlm_service_lock = Lock()
chat_enabled = os.environ.get("HACKHIVE_CHAT_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
chat_model_dir = Path(os.environ.get("HACKHIVE_CHAT_MODEL_DIR", "models/qwen3-0.6b")).resolve()
embedding_model_dir = Path(os.environ.get("HACKHIVE_EMBEDDING_MODEL_DIR", "models/all-minilm-l6-v2")).resolve()
chat_device = os.environ.get("HACKHIVE_CHAT_DEVICE", "cpu").strip().lower()
if chat_device not in {"cpu", "cuda"}:
    raise ValueError("HACKHIVE_CHAT_DEVICE must be cpu or cuda")
chat_model: Qwen3LocalModel | None = None
semantic_ranker: SemanticRanker | None = None
query_models_lock = Lock()


def get_action_classifier() -> XClipActionClassifier | None:
    global action_classifier
    provenance_path = action_model_dir / "hackhive-action-model.json"
    if not provenance_path.is_file():
        return None
    with action_classifier_lock:
        if action_classifier is None:
            try:
                import torch
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                action_classifier = XClipActionClassifier(action_model_dir, device=device)
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                raise HTTPException(status_code=503, detail=f"could not initialize local action model: {exc}") from exc
        return action_classifier


def _persist_vlm_result(job: SemanticJob, result: dict[str, Any]) -> None:
    connection = Store(database_path)
    try:
        connection.record_vlm_result(job, result)
    finally:
        connection.close()


def get_vlm_service() -> SemanticInferenceService | None:
    global vlm_service
    if not (vlm_model_dir / "hackhive-model.json").is_file():
        return None
    with vlm_service_lock:
        if vlm_service is None:
            vlm_service = SemanticInferenceService(
                SmolVLMWorker(vlm_model_dir), _persist_vlm_result, max_pending=3,
                selection_strategy=vlm_sampling_strategy,
            )
            vlm_service.start()
        return vlm_service


def get_query_engine() -> HybridQueryEngine:
    """Create an engine over the current authoritative Store with lazy local models."""
    global chat_model, semantic_ranker
    if not chat_enabled:
        return HybridQueryEngine(store)
    with query_models_lock:
        if chat_model is None and (chat_model_dir / "hackhive-chat-model.json").is_file():
            chat_model = Qwen3LocalModel(chat_model_dir, device=chat_device)
        if semantic_ranker is None and (
            embedding_model_dir / "hackhive-embedding-model.json"
        ).is_file():
            semantic_ranker = SemanticRanker(MiniLMEmbedder(embedding_model_dir))
    return HybridQueryEngine(store, text_model=chat_model, ranker=semantic_ranker)


def public_replay_status(result: dict[str, Any]) -> dict[str, Any]:
    latest = result.get("latest_frame")
    if latest:
        latest = dict(latest)
        latest.pop("asset_id", None)
        latest["frame_url"] = f"/api/replays/{result['session_id']}/frame"
        result = {**result, "latest_frame": latest}
    return result


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
    detector_ready = detector_weights.is_file() and importlib.util.find_spec("ultralytics") is not None
    action_model_ready = (
        (action_model_dir / "hackhive-action-model.json").is_file()
        and (action_model_dir / "config.json").is_file()
        and (action_model_dir / "model.safetensors").is_file()
    )
    semantic_status = vlm_service.snapshot() if vlm_service is not None else {
        "state": "available" if (vlm_model_dir / "hackhive-model.json").is_file() else "not_configured",
        "error": None,
        "pending": 0,
        "counters": {},
        "model_id": None,
        "model_revision": None,
        "selection_strategy": vlm_sampling_strategy,
    }
    query_status = {
        "enabled": chat_enabled,
        "state": (
            "disabled" if not chat_enabled
            else "ready" if chat_model is not None and semantic_ranker is not None
            else "available" if (
                (chat_model_dir / "hackhive-chat-model.json").is_file()
                and (embedding_model_dir / "hackhive-embedding-model.json").is_file()
            ) else "not_configured"
        ),
        "model_id": "Qwen/Qwen3-0.6B" if (chat_model_dir / "hackhive-chat-model.json").is_file() else None,
        "embedding_model_id": (
            "sentence-transformers/all-MiniLM-L6-v2"
            if (embedding_model_dir / "hackhive-embedding-model.json").is_file() else None
        ),
        "device": chat_device if chat_enabled else None,
        "local_only": True,
    }
    if not detector_ready:
        analysis_message = "Install ultralytics and add local YOLO weights to enable replay event analysis."
    elif action_model_ready:
        analysis_message = (
            "Local YOLO rules and XCLIP action ranking are available; action scores are uncalibrated "
            "and incidents require operator review."
        )
    else:
        analysis_message = (
            "YOLO candidate rules are available; the optional local XCLIP action model is not installed."
        )
    return {
        "source": {"kind": "file", "mode": "local", "health": "ready", "fixture": False},
        "analysis": {
            "mode": (
                "cpu_detector+xclip_action" if detector_ready and action_model_ready
                else "cpu_detector" if detector_ready
                else "decode_only"
            ),
            "detector_ready": detector_ready,
            "detector_weights": detector_weights.name if detector_weights.is_file() else None,
            "action_model_ready": action_model_ready,
            "action_model_id": "microsoft/xclip-base-patch32" if action_model_ready else None,
            "degraded": not detector_ready,
            "message": analysis_message,
            "vlm_connected": semantic_status["state"] in {"loading", "ready"},
            "vlm": semantic_status,
        },
        "policy": "Visual evidence only; no access feed",
        "timezone": "Asia/Kolkata",
        "query": query_status,
    }


@app.get("/api/cameras")
def cameras() -> list[dict[str, Any]]:
    row = store.connection.execute(
        "SELECT camera_id,name,source_kind,display_timezone,policy_version FROM cameras"
    ).fetchall()
    return [dict(item) for item in row]


@app.post("/api/uploads", status_code=201)
async def upload_recording(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        metadata = await save_recording_upload(file, recordings_root, max_bytes=max_upload_bytes)
    except UploadValidationError as exc:
        status_code = 413 if "upload limit" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    try:
        store.add_recording(metadata)
    except Exception:
        (recordings_root.resolve() / metadata["stored_name"]).unlink(missing_ok=True)
        raise
    return {**metadata, "source_name": metadata["stored_name"]}


@app.get("/api/uploads")
def recordings(limit: int = Query(default=50, ge=1, le=100)) -> dict[str, Any]:
    items = store.list_recordings(limit)
    return {"recordings": items, "count": len(items), "max_upload_bytes": max_upload_bytes}


@app.post("/api/replays", status_code=202)
def start_replay(request: ReplayStartRequest) -> dict[str, Any]:
    started = utc_now_ms()
    try:
        replay_manager.resolve_source(request.source_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    on_frame = None
    on_finish = None
    action_model_enabled = False
    if detector_weights.is_file():
        if importlib.util.find_spec("ultralytics") is None:
            raise HTTPException(
                status_code=503,
                detail="YOLO weights exist but ultralytics is missing; install requirements.txt.",
            )
        try:
            detector = YoloDetector(detector_weights, device="cpu")
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=f"could not initialize local detector: {exc}") from exc
        policy_version = "replay-policy-v1"
        policy = {"gate_line_y": request.gate_line_y, "zone": request.zone}
        classifier = get_action_classifier()
        semantic_service = get_vlm_service()
        session_id = store.create_session(
            started, mode="replay", policy_version=policy_version, policy_json=policy
        )
        action_model_enabled = classifier is not None
        orchestrator = EventOrchestrator(
            session_id=session_id,
            session_started_utc_ms=started,
            pipeline=PerceptionPipeline(
                detector,
                policy=ScenePolicy(gate_line_y=request.gate_line_y, zone=request.zone),
            ),
            store=store,
            media=MediaExporter(media_root),
            action_classifier=classifier,
            semantic_submit=semantic_service.submit if semantic_service is not None else None,
        )
        on_frame, on_finish = orchestrator.process, orchestrator.finish
    else:
        session_id = store.create_session(started, mode="replay")
    if on_frame is None:
        analysis_message = (
            "Video is decoding, but no event analysis ran because local YOLO weights "
            "were not found."
        )
    elif request.gate_line_y is None and action_model_enabled:
        analysis_message = (
            "YOLO temporal rules and local action model are active. Tailgating is disabled "
            "until a gate line is configured; predictions remain unreviewed."
        )
    elif request.gate_line_y is None:
        analysis_message = (
            "YOLO candidate rules are active; the local action model is not installed. Tailgating is "
            "disabled until a gate line is configured."
        )
    elif action_model_enabled:
        analysis_message = "YOLO temporal rules and local action model are active; predictions remain unreviewed."
    else:
        analysis_message = "Configured person-down, crowd, and gate-crossing rules are active; predictions require review."
    try:
        result = replay_manager.start(
            session_id=session_id,
            camera_id="cam-1",
            source_name=request.source_name,
            started_utc_ms=started,
            sample_fps=request.sample_fps,
            pace=request.pace,
            decode_strategy=request.decode_strategy,
            max_frames=request.max_frames,
            on_frame=on_frame,
            on_finish=on_finish,
            analysis_message=analysis_message,
        )
        return public_replay_status(result)
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


@app.get("/api/replays/{session_id}")
def replay_status(session_id: str) -> dict[str, Any]:
    try:
        return public_replay_status(replay_manager.status(session_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="replay session not found") from exc


@app.post("/api/replays/{session_id}/stop", status_code=202)
def stop_replay(session_id: str) -> dict[str, Any]:
    try:
        return public_replay_status(replay_manager.stop(session_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="replay session not found") from exc


@app.get("/api/replays/{session_id}/frame")
def replay_frame(session_id: str) -> FileResponse:
    try:
        path = replay_manager.latest_asset(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="replay session not found") from exc
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="replay has no sampled frame yet")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/events")
def events(
    category: list[Category] = Query(default=[]),
    camera_id: list[str] = Query(default=[]),
    review_status: list[ReviewStatus] = Query(default=[]),
    has_evidence: bool | None = None,
    start_utc_ms: int | None = None,
    end_utc_ms: int | None = None,
) -> dict[str, Any]:
    intent = QueryIntent(
        categories=category,
        camera_ids=camera_id,
        review_statuses=review_status,
        has_ready_media=has_evidence,
        start_utc_ms=start_utc_ms,
        end_utc_ms=end_utc_ms,
    )
    result = store.list_events(intent)
    return {"events": result, "count": len(result), "timezone": "Asia/Kolkata"}


@app.get("/api/events/{event_id}")
def event(event_id: str) -> dict[str, Any]:
    result = store.get_event(event_id, include_media=True)
    if result is None:
        raise HTTPException(status_code=404, detail="event not found")
    return result


@app.get("/api/media/{media_id}")
def media(media_id: str) -> FileResponse:
    item = store.get_media(media_id)
    if item is None:
        raise HTTPException(status_code=404, detail="media not found")
    if item["status"] != "ready":
        raise HTTPException(status_code=409, detail=f"media is {item['status']}")
    path = (media_root / item["relative_path"]).resolve()
    if not path.is_relative_to(media_root) or not path.is_file():
        raise HTTPException(status_code=404, detail="media file is unavailable")
    return FileResponse(path, media_type=item["mime_type"])


@app.post("/api/events/{event_id}/reviews")
def review(event_id: str, request: ReviewRequest) -> dict[str, Any]:
    if not store.connection.execute("SELECT 1 FROM events WHERE event_id=?", (event_id,)).fetchone():
        raise HTTPException(status_code=404, detail="event not found")
    return store.review(event_id, request)


@app.post("/api/fixtures/run")
def run_fixture() -> dict[str, Any]:
    started = utc_now_ms() - 10_000
    session_id = store.create_session(started, mode="offline_eval")
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
    session_id = store.create_session(started, mode="offline_eval")
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
def query(request: QueryIntent | NaturalQueryRequest) -> dict[str, Any]:
    suggestions = [
        "Show suspected tailgating today",
        "How many crowd gatherings were recorded?",
        "Did anyone appear to be lying on the ground?",
        "What unusual incidents happened on camera 1 this morning?",
    ]
    engine = get_query_engine()
    try:
        response = engine.execute(request)
    except QueryClarification as exc:
        return {
            "answer": str(exc), "needs_clarification": True, "events": [],
            "citations": [],
            "retrieval_mode": None,
            "timezone": request.timezone if isinstance(request, NaturalQueryRequest) else "Asia/Kolkata",
            "suggestions": suggestions,
        }
    fixture_sessions = engine.store.connection.execute(
        "SELECT COUNT(*) FROM stream_sessions WHERE mode='offline_eval'"
    ).fetchone()[0]
    replay_sessions = engine.store.connection.execute(
        "SELECT COUNT(*) FROM stream_sessions WHERE mode='replay'"
    ).fetchone()[0]
    return {
        **response,
        "coverage": {
            "fixture_sessions": fixture_sessions,
            "replay_sessions": replay_sessions,
            "warning": (
                "Results include deterministic fixture sessions; verify event provenance."
                if fixture_sessions else "Results include locally replayed footage only."
            ),
        },
        "suggestions": suggestions,
    }
