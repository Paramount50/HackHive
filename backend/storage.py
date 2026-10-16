from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from .contracts import (
    Category,
    DecisionState,
    FixtureAnalysis,
    MediaStatus,
    QueryIntent,
    ReviewRequest,
    ReviewStatus,
    utc_now_ms,
)
from .inference.contracts import PREDICATES, VisualObservation, verification_verdict
from .inference.service import SemanticJob


VERIFIED_SUMMARY_PREFIXES = {
    "person_down": "Possible person-down incident.",
    "crowd_gathering": "Possible sustained crowd gathering.",
    "suspected_tailgating": "Possible close same-direction gate passages.",
    "aggression_fight": "Possible physical altercation.",
    "fire_smoke": "Possible fire or smoke.",
    "intrusion_unauthorized_access": "Possible restricted-area intrusion.",
}
PREDICATE_LABELS = {
    "visible_person": "a visible person",
    "prone_posture": "prone posture",
    "persistence": "posture persistence",
    "usable_context": "usable scene context",
    "people_in_zone": "people in the monitored zone",
    "threshold_reached": "the configured crowd threshold",
    "duration_reached": "the configured crowd duration",
    "scene_visibility": "scene visibility",
    "distinct_people": "distinct people",
    "same_direction_passages": "same-direction passages",
    "temporal_proximity": "temporal proximity",
    "gate_visibility": "gate visibility",
    "multiple_people_visible": "multiple people",
    "physical_contact": "physical contact",
    "aggressive_motion": "aggressive motion",
    "temporal_persistence": "temporal persistence",
    "flame_like_region": "a flame-like region",
    "smoke_like_region": "a smoke-like region",
    "person_in_restricted_zone": "a person in the restricted zone",
    "restricted_boundary_crossing": "restricted-boundary crossing",
}


def verified_event_summary(category: str, observation: VisualObservation) -> str:
    """Build event-facing text from controlled evidence, never model prose."""
    supported = [
        PREDICATE_LABELS[item.predicate]
        for item in observation.observations
        if item.supported is True and item.predicate in PREDICATE_LABELS
    ]
    if not supported:
        return VERIFIED_SUMMARY_PREFIXES[category]
    if len(supported) == 1:
        evidence = supported[0]
    else:
        evidence = f"{', '.join(supported[:-1])} and {supported[-1]}"
    suffix = " Authorization remains unknown without policy/access evidence." if (
        category == "intrusion_unauthorized_access"
    ) else ""
    return f"{VERIFIED_SUMMARY_PREFIXES[category]} Visual verification supports {evidence}.{suffix}"


def evidence_completeness(category: str, predicates: list[dict[str, Any]]) -> float:
    """Return answered controlled predicates / expected controlled predicates."""
    expected = PREDICATES.get(category, set()) - {"access_mismatch"}
    if not expected:
        return 0.0
    latest = {item.get("name"): item.get("supported") for item in predicates}
    answered = sum(type(latest.get(predicate)) is bool for predicate in expected)
    return answered / len(expected)


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS cameras (
 camera_id TEXT PRIMARY KEY, name TEXT NOT NULL, source_kind TEXT NOT NULL,
 source_ref TEXT NOT NULL, display_timezone TEXT NOT NULL DEFAULT 'Asia/Kolkata',
 policy_version TEXT NOT NULL, policy_json TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS stream_sessions (
 session_id TEXT PRIMARY KEY, camera_id TEXT NOT NULL REFERENCES cameras(camera_id),
 mode TEXT NOT NULL, started_utc_ms INTEGER NOT NULL, ended_utc_ms INTEGER,
 policy_version TEXT NOT NULL, policy_json TEXT NOT NULL, source_sha256 TEXT
);
CREATE TABLE IF NOT EXISTS analysis_windows (
 window_id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES stream_sessions(session_id),
 start_utc_ms INTEGER NOT NULL, end_utc_ms INTEGER NOT NULL, frame_count INTEGER NOT NULL,
 reason TEXT NOT NULL, status TEXT NOT NULL, model_outcome TEXT, model_id TEXT,
 model_revision TEXT, prompt_version TEXT, observations_json TEXT, timings_json TEXT, error_code TEXT
);
CREATE TABLE IF NOT EXISTS access_grants (
 grant_id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES stream_sessions(session_id),
 gate_id TEXT NOT NULL, valid_from_utc_ms INTEGER NOT NULL, valid_until_utc_ms INTEGER NOT NULL,
 allowed_passages INTEGER NOT NULL, consumed_passages INTEGER NOT NULL DEFAULT 0,
 evidence_source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS passages (
 passage_id TEXT PRIMARY KEY, dedupe_key TEXT NOT NULL UNIQUE,
 session_id TEXT NOT NULL REFERENCES stream_sessions(session_id), gate_id TEXT NOT NULL,
 track_id TEXT NOT NULL, crossing_utc_ms INTEGER NOT NULL, crossing_pts_ms INTEGER NOT NULL,
 direction TEXT NOT NULL, grant_id TEXT REFERENCES access_grants(grant_id)
);
CREATE TABLE IF NOT EXISTS events (
 event_id TEXT PRIMARY KEY, candidate_key TEXT NOT NULL UNIQUE,
 session_id TEXT NOT NULL REFERENCES stream_sessions(session_id), category TEXT NOT NULL,
 zone_id TEXT, start_utc_ms INTEGER NOT NULL, end_utc_ms INTEGER, trigger_utc_ms INTEGER NOT NULL,
 last_observed_utc_ms INTEGER NOT NULL, start_pts_ms INTEGER NOT NULL, end_pts_ms INTEGER,
 first_alert_utc_ms INTEGER, decision_state TEXT NOT NULL, lifecycle_state TEXT NOT NULL,
 summary TEXT NOT NULL, confidence_score REAL NOT NULL, confidence_kind TEXT NOT NULL,
 predicates_json TEXT NOT NULL, track_ids_json TEXT NOT NULL, window_ids_json TEXT NOT NULL,
 policy_version TEXT NOT NULL, policy_json TEXT NOT NULL, authorization_state TEXT NOT NULL,
 access_evidence_source TEXT, grant_ids_json TEXT NOT NULL, passage_ids_json TEXT NOT NULL,
 limitations_json TEXT NOT NULL, created_utc_ms INTEGER NOT NULL, updated_utc_ms INTEGER NOT NULL,
 metadata_expires_utc_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS media (
 media_id TEXT PRIMARY KEY, event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
 kind TEXT NOT NULL, relative_path TEXT NOT NULL UNIQUE, mime_type TEXT NOT NULL,
 status TEXT NOT NULL, coverage_start_utc_ms INTEGER NOT NULL, coverage_end_utc_ms INTEGER NOT NULL,
 requested_start_utc_ms INTEGER NOT NULL, requested_end_utc_ms INTEGER NOT NULL,
 byte_size INTEGER, sha256 TEXT, expires_utc_ms INTEGER NOT NULL, failure_reason TEXT
);
CREATE TABLE IF NOT EXISTS reviews (
 review_seq INTEGER PRIMARY KEY AUTOINCREMENT, review_id TEXT NOT NULL UNIQUE,
 request_key TEXT NOT NULL UNIQUE, event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
 action TEXT NOT NULL, actor_label TEXT NOT NULL, note TEXT NOT NULL, created_utc_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS recordings (
 recording_id TEXT PRIMARY KEY, original_name TEXT NOT NULL, stored_name TEXT NOT NULL UNIQUE,
 mime_type TEXT NOT NULL, byte_size INTEGER NOT NULL, sha256 TEXT NOT NULL,
 duration_ms INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
 fps REAL NOT NULL, frame_count INTEGER NOT NULL, created_utc_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_time ON events(start_utc_ms, last_observed_utc_ms);
CREATE INDEX IF NOT EXISTS idx_recordings_created ON recordings(created_utc_ms DESC);
"""


class Store:
    def __init__(self, path: str | Path = ":memory:") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)
        self.connection.execute("PRAGMA journal_mode=WAL")

    def close(self) -> None:
        self.connection.close()

    def seed_camera(self) -> None:
        policy = {
            "version": "cam-1-v1",
            "zones": [{"id": "door", "polygon_normalized": [[0.1, 0.2], [0.8, 0.9]]}],
            "entry": {"gate_id": "door", "following_window_s": 2, "grant_log_complete": False},
        }
        self.connection.execute(
            """INSERT OR IGNORE INTO cameras
            (camera_id,name,source_kind,source_ref,policy_version,policy_json)
            VALUES ('cam-1','Fixture Gate','file','fixture://gate','cam-1-v1',?)""",
            (json.dumps(policy),),
        )
        self.connection.commit()

    def create_session(
        self,
        started_utc_ms: int,
        mode: str = "replay",
        *,
        policy_version: str | None = None,
        policy_json: dict[str, Any] | None = None,
    ) -> str:
        self.seed_camera()
        session_id = str(uuid.uuid4())
        policy = self.connection.execute(
            "SELECT policy_version, policy_json FROM cameras WHERE camera_id='cam-1'"
        ).fetchone()
        self.connection.execute(
            """INSERT INTO stream_sessions
            (session_id,camera_id,mode,started_utc_ms,policy_version,policy_json)
            VALUES (?,?,?,?,?,?)""",
            (
                session_id,
                "cam-1",
                mode,
                started_utc_ms,
                policy_version or policy["policy_version"],
                json.dumps(policy_json) if policy_json is not None else policy["policy_json"],
            ),
        )
        self.connection.commit()
        return session_id

    @staticmethod
    def _predicates(analysis: FixtureAnalysis) -> list[dict[str, Any]]:
        supported = len(analysis.observations)
        return [
            {
                "name": item.get("predicate", "unclassified"),
                "supported": item.get("supported"),
                "evidence_window_ids": [],
                "frame_indices": item.get("frame_indices", analysis.frame_indices),
                **({"details": item["details"]} if isinstance(item.get("details"), dict) else {}),
            }
            for item in analysis.observations
        ] + [{"name": "evidence_completeness", "supported": supported >= 1}]

    def record_analysis(
        self,
        session_id: str,
        start_utc_ms: int,
        end_utc_ms: int,
        analysis: FixtureAnalysis,
        candidate_key: str,
        *,
        start_pts_ms: int | None = None,
    ) -> str | None:
        now = utc_now_ms()
        window_id = str(uuid.uuid4())
        outcome = str(analysis.outcome)
        self.connection.execute(
            """INSERT INTO analysis_windows
            (window_id,session_id,start_utc_ms,end_utc_ms,frame_count,reason,status,model_outcome,
             model_id,model_revision,prompt_version,observations_json,timings_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (window_id, session_id, start_utc_ms, end_utc_ms, len(analysis.frame_indices),
             "candidate", "completed", outcome, analysis.model_id, analysis.model_revision,
             analysis.prompt_version, json.dumps(analysis.observations), json.dumps({"fixture": True})),
        )
        if outcome in {"none", "unknown"}:
            self.connection.commit()
            return None
        existing = self.connection.execute(
            "SELECT event_id,lifecycle_state,window_ids_json FROM events WHERE candidate_key=?",
            (candidate_key,),
        ).fetchone()
        if existing and existing["lifecycle_state"] == "ended":
            candidate_key = f"{candidate_key}:{start_utc_ms}"
            existing = self.connection.execute(
                "SELECT event_id,lifecycle_state,window_ids_json FROM events WHERE candidate_key=?",
                (candidate_key,),
            ).fetchone()
        if existing:
            window_ids = json.loads(existing["window_ids_json"])
            if window_id not in window_ids:
                window_ids.append(window_id)
            self.connection.execute(
                """UPDATE events SET last_observed_utc_ms=?, end_utc_ms=NULL, end_pts_ms=NULL,
                lifecycle_state='active', updated_utc_ms=?,
                window_ids_json=? WHERE event_id=?""",
                (end_utc_ms, now, json.dumps(window_ids), existing["event_id"]),
            )
            self.connection.commit()
            return existing["event_id"]
        predicates = self._predicates(analysis)
        score = evidence_completeness(outcome, predicates)
        event_id = str(uuid.uuid4())
        policy = self.connection.execute(
            "SELECT policy_version,policy_json FROM stream_sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        candidate_only = analysis.model_id.startswith(("candidate://", "xclip://"))
        decision = DecisionState.NEEDS_REVIEW if outcome == Category.SUSPECTED_TAILGATING or candidate_only else DecisionState.ACCEPTED
        self.connection.execute(
            """INSERT INTO events
            (event_id,candidate_key,session_id,category,start_utc_ms,trigger_utc_ms,last_observed_utc_ms,
             start_pts_ms,first_alert_utc_ms,decision_state,lifecycle_state,summary,confidence_score,
             confidence_kind,predicates_json,track_ids_json,window_ids_json,policy_version,policy_json,
             authorization_state,access_evidence_source,grant_ids_json,passage_ids_json,limitations_json,
             created_utc_ms,updated_utc_ms,metadata_expires_utc_ms)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (event_id, candidate_key, session_id, outcome, start_utc_ms, end_utc_ms, end_utc_ms,
             start_utc_ms if start_pts_ms is None else start_pts_ms,
             now, decision, "active", analysis.summary, score,
             "evidence_completeness_v2", json.dumps(predicates), "[]", json.dumps([window_id]),
             policy["policy_version"], policy["policy_json"],
             (
                 "unknown"
                 if outcome in {
                     Category.SUSPECTED_TAILGATING,
                     Category.INTRUSION_UNAUTHORIZED_ACCESS,
                 }
                 else "not_applicable"
             ),
             (
                 "simulated"
                 if outcome == Category.SUSPECTED_TAILGATING and analysis.model_id.startswith("fixture://")
                 else None
             ),
             "[]", "[]",
             json.dumps(analysis.limitations), now, now, now + 90 * 24 * 3600 * 1000),
        )
        self.connection.commit()
        return event_id

    def record_vlm_result(self, job: SemanticJob, result: dict[str, Any]) -> str:
        """Attach one model attempt to an existing candidate without auto-accepting it."""
        event = self.connection.execute(
            """SELECT e.*, s.started_utc_ms AS session_started_utc_ms
               FROM events e JOIN stream_sessions s ON s.session_id=e.session_id
               WHERE e.event_id=?""",
            (job.event_id,),
        ).fetchone()
        if event is None:
            raise KeyError(job.event_id)
        observation: VisualObservation | None = None
        status = str(result.get("status", "failed"))
        if status not in {"pending", "running", "completed", "rejected", "failed", "expired", "not_configured"}:
            status = "failed"
        verdict: str | None = None
        error_code = result.get("error_code")
        if status == "completed" and result.get("observation") is not None:
            try:
                observation = VisualObservation.model_validate(result["observation"])
                verdict = verification_verdict(observation, job.expected_outcome)
            except (TypeError, ValueError) as exc:
                error_code = "invalid_model_output"
                status = "failed"
                result = {**result, "error": str(exc)[:1000]}
        elif status == "completed":
            status = "failed"
            error_code = error_code or "missing_observation"
        window_id = job.verification_id or str(uuid.uuid4())
        start_utc_ms = (
            event["session_started_utc_ms"] + job.source_pts_ms[0]
            if job.source_pts_ms else event["start_utc_ms"]
        )
        end_utc_ms = (
            event["session_started_utc_ms"] + job.source_pts_ms[-1]
            if job.source_pts_ms else event["last_observed_utc_ms"]
        )
        payload = {
            "status": status,
            "verdict": verdict,
            "expected_outcome": job.expected_outcome,
            "observation": observation.model_dump() if observation else None,
            "raw_output": str(result.get("raw_output", ""))[:8_000],
            "error": result.get("error"),
            "error_code": error_code,
            "validation_stage": result.get("validation_stage"),
            "validation_error": result.get("validation_error"),
            "hit_max_new_tokens": result.get("hit_max_new_tokens"),
            "eos_reached": result.get("eos_reached"),
            "candidate_category": result.get("candidate_category", job.expected_outcome),
            "selection": result.get("selection"),
        }
        with self.connection:
            self.connection.execute(
                """INSERT INTO analysis_windows
                (window_id,session_id,start_utc_ms,end_utc_ms,frame_count,reason,status,model_outcome,
                 model_id,model_revision,prompt_version,observations_json,timings_json,error_code)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(window_id) DO UPDATE SET
                 start_utc_ms=excluded.start_utc_ms,end_utc_ms=excluded.end_utc_ms,
                 frame_count=excluded.frame_count,status=excluded.status,
                 model_outcome=excluded.model_outcome,model_id=excluded.model_id,
                 model_revision=excluded.model_revision,prompt_version=excluded.prompt_version,
                 observations_json=excluded.observations_json,timings_json=excluded.timings_json,
                 error_code=excluded.error_code""",
                (
                    window_id, event["session_id"], start_utc_ms, end_utc_ms, len(job.image_paths),
                    "vlm_verification", status,
                    observation.outcome if observation else None,
                    result.get("model_id"), result.get("model_revision"), result.get("prompt_version"),
                    json.dumps(payload), json.dumps(result.get("timings_ms", {})), error_code,
                ),
            )
            current = self.connection.execute(
                "SELECT window_ids_json,limitations_json,predicates_json FROM events WHERE event_id=?",
                (job.event_id,),
            ).fetchone()
            window_ids = json.loads(current["window_ids_json"])
            window_ids.append(window_id)
            window_ids = list(dict.fromkeys(window_ids))
            limitations = json.loads(current["limitations_json"])
            if verdict is not None:
                limitation = f"VLM verification verdict: {verdict}."
                if limitation not in limitations:
                    limitations.append(limitation)
            elif status in {"rejected", "failed", "expired", "not_configured"}:
                limitation = f"VLM verification status: {status} ({error_code or 'unspecified'})."
                if limitation not in limitations:
                    limitations.append(limitation)
            updates: dict[str, Any] = {
                "window_ids_json": json.dumps(window_ids),
                "limitations_json": json.dumps(limitations[-8:]),
                "updated_utc_ms": utc_now_ms(),
                "decision_state": str(DecisionState.NEEDS_REVIEW),
            }
            if verdict == "supported" and observation is not None:
                predicates = json.loads(current["predicates_json"])
                predicates = [
                    item for item in predicates
                    if window_id not in item.get("evidence_window_ids", [])
                ]
                predicates.extend({
                    "name": item.predicate,
                    "supported": item.supported,
                    "evidence_window_ids": [window_id],
                    "frame_indices": item.frame_indices,
                    "source": "vlm_verification",
                } for item in observation.observations)
                updates["summary"] = verified_event_summary(job.expected_outcome, observation)
                updates["predicates_json"] = json.dumps(predicates)
                updates["confidence_score"] = evidence_completeness(
                    job.expected_outcome, predicates
                )
                updates["confidence_kind"] = "evidence_completeness_v2"
            assignments = ",".join(f"{column}=?" for column in updates)
            self.connection.execute(
                f"UPDATE events SET {assignments} WHERE event_id=?",
                [*updates.values(), job.event_id],
            )
        return verdict or status

    def close_inactive_events(
        self, observed_utc_ms: int, merge_gap_ms: int = 5_000, session_id: str | None = None
    ) -> list[str]:
        """Close active incidents whose last evidence is older than the merge gap."""
        if merge_gap_ms < 0:
            raise ValueError("merge_gap_ms must not be negative")
        cutoff = observed_utc_ms - merge_gap_ms
        sql = """SELECT event_id,start_utc_ms,start_pts_ms,last_observed_utc_ms FROM events
                 WHERE lifecycle_state='active' AND last_observed_utc_ms < ?"""
        params: list[Any] = [cutoff]
        if session_id is not None:
            sql += " AND session_id=?"
            params.append(session_id)
        rows = self.connection.execute(
            sql, params,
        ).fetchall()
        now = utc_now_ms()
        with self.connection:
            for row in rows:
                duration_ms = max(0, row["last_observed_utc_ms"] - row["start_utc_ms"])
                self.connection.execute(
                    """UPDATE events SET lifecycle_state='ended', end_utc_ms=?, end_pts_ms=?,
                       updated_utc_ms=? WHERE event_id=? AND lifecycle_state='active'""",
                    (
                        row["last_observed_utc_ms"],
                        row["start_pts_ms"] + duration_ms,
                        now,
                        row["event_id"],
                    ),
                )
        return [row["event_id"] for row in rows]

    def get_event_row(self, event_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
        return dict(row) if row else None

    def get_event(self, event_id: str, *, include_media: bool = False) -> dict[str, Any] | None:
        row = self.connection.execute(
            """SELECT e.*, c.camera_id, c.name AS camera_name, s.mode AS session_mode
               FROM events e JOIN stream_sessions s ON s.session_id=e.session_id
               JOIN cameras c ON c.camera_id=s.camera_id WHERE e.event_id=?""",
            (event_id,),
        ).fetchone()
        if row is None:
            return None
        review, reviews = self._review_projection(event_id)
        item = dict(row)
        item["review_status"] = review
        item["reviews"] = reviews
        item["predicates"] = json.loads(row["predicates_json"])
        item["limitations"] = json.loads(row["limitations_json"])
        item["verification"] = self._verification_projection(event_id)
        item["candidate"] = self._candidate_projection(event_id)
        if include_media:
            item["media"] = [
                dict(media) for media in self.connection.execute(
                    "SELECT * FROM media WHERE event_id=? ORDER BY coverage_start_utc_ms,media_id",
                    (event_id,),
                ).fetchall()
            ]
        return item

    def _verification_projection(self, event_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            """SELECT status,model_id,model_revision,prompt_version,observations_json,timings_json,error_code
               FROM analysis_windows
               WHERE reason='vlm_verification' AND window_id IN (
                 SELECT value FROM json_each((SELECT window_ids_json FROM events WHERE event_id=?))
               )
               ORDER BY rowid DESC LIMIT 1""",
            (event_id,),
        ).fetchone()
        if row is None:
            return {"status": "not_run", "verdict": None}
        payload = json.loads(row["observations_json"] or "{}")
        return {
            "status": row["status"],
            "verdict": payload.get("verdict"),
            "model_id": row["model_id"],
            "model_revision": row["model_revision"],
            "prompt_version": row["prompt_version"],
            "observation": payload.get("observation"),
            "selection": payload.get("selection"),
            "timings_ms": json.loads(row["timings_json"] or "{}"),
            "error_code": row["error_code"],
            "error": payload.get("error"),
        }

    def _candidate_projection(self, event_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            """SELECT model_id,model_revision,prompt_version
               FROM analysis_windows
               WHERE reason='candidate' AND window_id IN (
                 SELECT value FROM json_each((SELECT window_ids_json FROM events WHERE event_id=?))
               )
               ORDER BY rowid DESC LIMIT 1""",
            (event_id,),
        ).fetchone()
        if row is None:
            return {"source": "Unknown candidate source", "model_id": None}
        model_id = row["model_id"] or ""
        if model_id.startswith("xclip://"):
            source = "XCLIP action ranking"
        elif model_id == "candidate://cpu-rules":
            source = "YOLO + temporal rule"
        elif model_id.startswith("fixture://"):
            source = "Deterministic fixture"
        else:
            source = model_id or "Unknown candidate source"
        return {
            "source": source,
            "model_id": row["model_id"],
            "model_revision": row["model_revision"],
            "prompt_version": row["prompt_version"],
        }

    def add_media(
        self,
        event_id: str,
        status: MediaStatus = MediaStatus.MISSING,
        *,
        kind: str = "keyframe",
    ) -> str:
        media_id = str(uuid.uuid4())
        now = utc_now_ms()
        self.connection.execute(
            """INSERT INTO media
            (media_id,event_id,kind,relative_path,mime_type,status,coverage_start_utc_ms,coverage_end_utc_ms,
             requested_start_utc_ms,requested_end_utc_ms,expires_utc_ms)
            VALUES (?,?,?,?,?,?, ?,?,?,?,?)""",
            (media_id, event_id, kind, f"fixtures/{event_id}-{kind}",
             "image/jpeg" if kind == "keyframe" else "video/mp4", status,
             now, now, now, now, now + 90 * 24 * 3600 * 1000),
        )
        self.connection.commit()
        return media_id

    def add_media_artifact(self, event_id: str, artifact: Any) -> str:
        media_id = str(uuid.uuid4())
        now = utc_now_ms()
        self.connection.execute(
            """INSERT INTO media
            (media_id,event_id,kind,relative_path,mime_type,status,coverage_start_utc_ms,coverage_end_utc_ms,
             requested_start_utc_ms,requested_end_utc_ms,byte_size,sha256,expires_utc_ms)
            VALUES (?,?,?,?,?,'ready',?,?,?,?,?,?,?)""",
            (
                media_id, event_id, artifact.kind, artifact.relative_path, artifact.mime_type,
                artifact.coverage_start_utc_ms, artifact.coverage_end_utc_ms,
                artifact.requested_start_utc_ms, artifact.requested_end_utc_ms,
                artifact.byte_size, artifact.sha256, now + 30 * 24 * 3600 * 1000,
            ),
        )
        self.connection.commit()
        return media_id

    def get_media(self, media_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM media WHERE media_id=?", (media_id,)).fetchone()
        return dict(row) if row else None

    def add_recording(self, metadata: dict[str, Any]) -> dict[str, Any]:
        self.connection.execute(
            """INSERT INTO recordings
            (recording_id,original_name,stored_name,mime_type,byte_size,sha256,duration_ms,
             width,height,fps,frame_count,created_utc_ms)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                metadata["recording_id"], metadata["original_name"], metadata["stored_name"],
                metadata["mime_type"], metadata["byte_size"], metadata["sha256"],
                metadata["duration_ms"], metadata["width"], metadata["height"],
                metadata["fps"], metadata["frame_count"], metadata["created_utc_ms"],
            ),
        )
        self.connection.commit()
        return metadata

    def get_recording(self, recording_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM recordings WHERE recording_id=?", (recording_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_recordings(self, limit: int = 50) -> list[dict[str, Any]]:
        return [
            dict(row) for row in self.connection.execute(
                "SELECT * FROM recordings ORDER BY created_utc_ms DESC LIMIT ?", (limit,)
            ).fetchall()
        ]

    def add_grant(
        self,
        session_id: str,
        grant_id: str,
        gate_id: str,
        valid_from_utc_ms: int,
        valid_until_utc_ms: int,
        allowed_passages: int = 1,
    ) -> None:
        if valid_until_utc_ms <= valid_from_utc_ms or allowed_passages < 1:
            raise ValueError("invalid grant window")
        self.connection.execute(
            """INSERT INTO access_grants
            (grant_id,session_id,gate_id,valid_from_utc_ms,valid_until_utc_ms,allowed_passages,evidence_source)
            VALUES (?,?,?,?,?,?,?)""",
            (grant_id, session_id, gate_id, valid_from_utc_ms, valid_until_utc_ms, allowed_passages, "simulated"),
        )
        self.connection.commit()

    def record_passage(
        self,
        session_id: str,
        gate_id: str,
        track_id: str,
        crossing_utc_ms: int,
        crossing_pts_ms: int,
        direction: str = "entry",
        dedupe_key: str | None = None,
    ) -> dict[str, Any]:
        if direction not in {"entry", "exit"}:
            raise ValueError("direction must be entry or exit")
        key = dedupe_key or f"{session_id}:{gate_id}:{track_id}:{crossing_pts_ms}:{direction}"
        existing = self.connection.execute(
            "SELECT * FROM passages WHERE dedupe_key=?", (key,)
        ).fetchone()
        if existing:
            return dict(existing)
        passage_id = str(uuid.uuid4())
        grant = self.connection.execute(
            """SELECT * FROM access_grants
            WHERE session_id=? AND gate_id=? AND valid_from_utc_ms<=? AND valid_until_utc_ms>?
            AND consumed_passages < allowed_passages ORDER BY valid_from_utc_ms LIMIT 1""",
            (session_id, gate_id, crossing_utc_ms, crossing_utc_ms),
        ).fetchone()
        with self.connection:
            self.connection.execute(
                """INSERT INTO passages
                (passage_id,dedupe_key,session_id,gate_id,track_id,crossing_utc_ms,crossing_pts_ms,direction,grant_id)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (passage_id, key, session_id, gate_id, track_id, crossing_utc_ms, crossing_pts_ms, direction,
                 grant["grant_id"] if grant else None),
            )
            if grant:
                self.connection.execute(
                    "UPDATE access_grants SET consumed_passages=consumed_passages+1 WHERE grant_id=?",
                    (grant["grant_id"],),
                )
        return dict(self.connection.execute("SELECT * FROM passages WHERE passage_id=?", (passage_id,)).fetchone())

    def _review_projection(self, event_id: str) -> tuple[ReviewStatus, list[dict[str, Any]]]:
        rows = self.connection.execute(
            "SELECT * FROM reviews WHERE event_id=? ORDER BY review_seq", (event_id,)
        ).fetchall()
        if not rows:
            return ReviewStatus.UNREVIEWED, []
        action = rows[-1]["action"]
        status = ReviewStatus.CONFIRMED if action == "confirm" else ReviewStatus.REJECTED
        return status, [dict(row) for row in rows]

    def list_events(self, intent: QueryIntent) -> list[dict[str, Any]]:
        sql = """SELECT e.*, c.camera_id, c.name AS camera_name, s.mode AS session_mode,
                 (SELECT m.media_id FROM media m
                  WHERE m.event_id=e.event_id AND m.kind='keyframe' AND m.status='ready'
                  ORDER BY m.coverage_start_utc_ms,m.media_id LIMIT 1) AS thumbnail_media_id,
                 EXISTS(SELECT 1 FROM media em WHERE em.event_id=e.event_id AND em.status='ready')
                  AS has_ready_media,
                 EXISTS(SELECT 1 FROM media km WHERE km.event_id=e.event_id
                        AND km.kind='keyframe' AND km.status='ready') AS has_keyframe,
                 EXISTS(SELECT 1 FROM media cm WHERE cm.event_id=e.event_id
                        AND cm.kind='clip' AND cm.status='ready') AS has_clip
                 FROM events e JOIN stream_sessions s ON s.session_id=e.session_id
                 JOIN cameras c ON c.camera_id=s.camera_id WHERE 1=1"""
        params: list[Any] = []
        if intent.categories:
            sql += f" AND e.category IN ({','.join('?' for _ in intent.categories)})"
            params.extend(intent.categories)
        if intent.camera_ids:
            sql += f" AND c.camera_id IN ({','.join('?' for _ in intent.camera_ids)})"
            params.extend(intent.camera_ids)
        if intent.decision_states:
            sql += f" AND e.decision_state IN ({','.join('?' for _ in intent.decision_states)})"
            params.extend(intent.decision_states)
        if intent.review_statuses:
            latest_review = """(SELECT r.action FROM reviews r WHERE r.event_id=e.event_id
                                ORDER BY r.review_seq DESC LIMIT 1)"""
            review_clauses = []
            if ReviewStatus.UNREVIEWED in intent.review_statuses:
                review_clauses.append(f"{latest_review} IS NULL")
            actions = []
            if ReviewStatus.CONFIRMED in intent.review_statuses:
                actions.append("confirm")
            if ReviewStatus.REJECTED in intent.review_statuses:
                actions.append("reject")
            if actions:
                review_clauses.append(f"{latest_review} IN ({','.join('?' for _ in actions)})")
                params.extend(actions)
            sql += f" AND ({' OR '.join(review_clauses)})"
        if intent.has_ready_media is not None:
            predicate = "EXISTS" if intent.has_ready_media else "NOT EXISTS"
            sql += f" AND {predicate}(SELECT 1 FROM media fm WHERE fm.event_id=e.event_id AND fm.status='ready')"
        if intent.start_utc_ms is not None:
            sql += " AND e.last_observed_utc_ms >= ?"
            params.append(intent.start_utc_ms)
        if intent.end_utc_ms is not None:
            sql += " AND e.start_utc_ms < ?"
            params.append(intent.end_utc_ms)
        sql += " ORDER BY e.start_utc_ms DESC LIMIT ?"
        params.append(intent.limit if intent.operation == "list" else 10000)
        result = []
        for row in self.connection.execute(sql, params).fetchall():
            review, reviews = self._review_projection(row["event_id"])
            item = dict(row)
            item["review_status"] = review
            item["reviews"] = reviews
            item["predicates"] = json.loads(row["predicates_json"])
            item["limitations"] = json.loads(row["limitations_json"])
            item["verification"] = self._verification_projection(row["event_id"])
            item["candidate"] = self._candidate_projection(row["event_id"])
            result.append(item)
        return result

    def review(self, event_id: str, request: ReviewRequest) -> dict[str, Any]:
        existing = self.connection.execute(
            "SELECT review_id FROM reviews WHERE request_key=?", (request.request_key,)
        ).fetchone()
        if not existing:
            self.connection.execute(
                """INSERT INTO reviews
                (review_id,request_key,event_id,action,actor_label,note,created_utc_ms)
                VALUES (?,?,?,?,?,?,?)""",
                (str(uuid.uuid4()), request.request_key, event_id, request.action,
                 request.actor_label, request.note, utc_now_ms()),
            )
            self.connection.commit()
        result = self.get_event(event_id)
        if result is None:
            raise KeyError(event_id)
        return result
