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
        score = sum(p.get("supported") is True for p in predicates) / 4
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
             "evidence_completeness_v1", json.dumps(predicates), "[]", json.dumps([window_id]),
             policy["policy_version"], policy["policy_json"],
             "unknown" if outcome == Category.SUSPECTED_TAILGATING else "not_applicable",
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
        if include_media:
            item["media"] = [
                dict(media) for media in self.connection.execute(
                    "SELECT * FROM media WHERE event_id=? ORDER BY coverage_start_utc_ms,media_id",
                    (event_id,),
                ).fetchall()
            ]
        return item

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
        sql = """SELECT e.*, c.camera_id, c.name AS camera_name, s.mode AS session_mode
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
        result = self.get_event(event_id)
        if result is None:
            raise KeyError(event_id)
        return result
