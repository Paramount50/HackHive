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
CREATE INDEX IF NOT EXISTS idx_events_time ON events(start_utc_ms, last_observed_utc_ms);
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

    def create_session(self, started_utc_ms: int, mode: str = "replay") -> str:
        self.seed_camera()
        session_id = str(uuid.uuid4())
        policy = self.connection.execute(
            "SELECT policy_version, policy_json FROM cameras WHERE camera_id='cam-1'"
        ).fetchone()
        self.connection.execute(
            """INSERT INTO stream_sessions
            (session_id,camera_id,mode,started_utc_ms,policy_version,policy_json)
            VALUES (?,?,?,?,?,?)""",
            (session_id, "cam-1", mode, started_utc_ms, policy["policy_version"], policy["policy_json"]),
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
            "SELECT event_id FROM events WHERE candidate_key=?", (candidate_key,)
        ).fetchone()
        if existing:
            self.connection.execute(
                """UPDATE events SET last_observed_utc_ms=?, end_utc_ms=?, updated_utc_ms=?,
                window_ids_json=? WHERE event_id=?""",
                (end_utc_ms, end_utc_ms, now, json.dumps([window_id]), existing["event_id"]),
            )
            self.connection.commit()
            return existing["event_id"]
        predicates = self._predicates(analysis)
        score = sum(p.get("supported") is True for p in predicates) / 4
        event_id = str(uuid.uuid4())
        policy = self.connection.execute(
            "SELECT policy_version,policy_json FROM stream_sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        decision = DecisionState.NEEDS_REVIEW if outcome == Category.SUSPECTED_TAILGATING else DecisionState.ACCEPTED
        self.connection.execute(
            """INSERT INTO events
            (event_id,candidate_key,session_id,category,start_utc_ms,trigger_utc_ms,last_observed_utc_ms,
             start_pts_ms,first_alert_utc_ms,decision_state,lifecycle_state,summary,confidence_score,
             confidence_kind,predicates_json,track_ids_json,window_ids_json,policy_version,policy_json,
             authorization_state,access_evidence_source,grant_ids_json,passage_ids_json,limitations_json,
             created_utc_ms,updated_utc_ms,metadata_expires_utc_ms)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (event_id, candidate_key, session_id, outcome, start_utc_ms, start_utc_ms, end_utc_ms,
             start_utc_ms, now, decision, "active", analysis.summary, score,
             "evidence_completeness_v1", json.dumps(predicates), "[]", json.dumps([window_id]),
             policy["policy_version"], policy["policy_json"],
             "unknown" if outcome == Category.SUSPECTED_TAILGATING else "not_applicable",
             "simulated" if outcome == Category.SUSPECTED_TAILGATING else None, "[]", "[]",
             json.dumps(analysis.limitations), now, now, now + 90 * 24 * 3600 * 1000),
        )
        self.connection.commit()
        return event_id

    def add_media(self, event_id: str, status: MediaStatus = MediaStatus.READY) -> str:
        media_id = str(uuid.uuid4())
        now = utc_now_ms()
        self.connection.execute(
            """INSERT INTO media
            (media_id,event_id,kind,relative_path,mime_type,status,coverage_start_utc_ms,coverage_end_utc_ms,
             requested_start_utc_ms,requested_end_utc_ms,expires_utc_ms)
            VALUES (?,?,'keyframe',?,'image/jpeg',?,?,?,?,?,?)""",
            (media_id, event_id, f"fixtures/{event_id}.jpg", status, now, now, now, now, now + 90 * 24 * 3600 * 1000),
        )
        self.connection.commit()
        return media_id

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
        sql = """SELECT e.*, c.camera_id, c.name AS camera_name
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
        return next(item for item in self.list_events(QueryIntent(limit=100)) if item["event_id"] == event_id)
