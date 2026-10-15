# Build tracker

## Person A staged work — 2026-10-15

Stage 1: added `backend/inference/contracts.py`, `hardware.py` and contract tests.
Validation rejects malformed JSON, unsupported P0 outcomes, invalid frame references,
non-boolean evidence and visual claims of access authorization. Unknown remains distinct
from normal. Three unittest cases (including seven invalid-output subcases) pass using
the bundled Python 3.12 runtime. No model execution is claimed by these tests.

Hardware at 17:05 UTC: RTX 3060 Laptop, 6144 MiB dedicated memory, 4838 MiB free,
driver 591.74. Provisional budget after 1024 MiB reserve: 3814 MiB. Available system
RAM: 6331/16236 MiB after the user closed applications. These are snapshots, not
minimum-free-memory measurements over a soak. Six user-provided AVI files exist in
the sibling `sample_videos` directory; labels and footage provenance remain unverified.

Stage 2: added the local-only serialized SmolVLM2 adapter and `scripts/smoke_vlm.py`.
The runner samples real video, pins the model revision, saves source hashes and raw
failed/successful responses, excludes warm-up, and records runtime/memory/timings.
All 42 tests pass (one upstream Starlette/AnyIO deprecation warning). Six supplied
AVI files decode into four ordered frames each. GPU execution remains pending
dependency/model downloads at this stage. No detection accuracy, model quality or
real-time performance is established. Replay integration, ByteTrack, scheduler audit
fairness, and deterministic policy application remain future stages.

Updated: 2026-10-15. This tracker records evidence, not intentions. The first CPU-only backend slice is implemented; real model and performance work remain unrun.

Status values: `TODO`, `IN_PROGRESS`, `BLOCKED`, `DONE`, `CUT`. An item becomes DONE only when its acceptance evidence is linked. Owners A/B/C are placeholders defined in [ImplementationPlan.md](ImplementationPlan.md).

Only A has GPU access. See [TeamAssignments.md](TeamAssignments.md) for CPU-only development and handoffs, and [MeetingDecisions.md](MeetingDecisions.md) for the organizer-reference decisions. All model/runtime experiments remain unrun.

## Planning deliverables

| Item | Status | Evidence |
|---|---|---|
| Scope and requirements | DONE | [prd.md](prd.md) |
| Research review and decisions | DONE | [Research.md](Research.md) |
| Technical/data contracts | DONE | [techspec.md](techspec.md), [Schema.md](Schema.md) |
| User flow and UI | DONE | [Appflow.md](Appflow.md), [design.md](design.md) |
| Build/evaluation plan and working rules | DONE | [ImplementationPlan.md](ImplementationPlan.md), [Rules.md](Rules.md), [AGENTS.md](AGENTS.md) |

Documentation verification after the meeting update: all twelve Markdown files exist, local Markdown cross-links and code fences passed checks, all four JSON examples parsed, and the revised DDL successfully created eight tables in an in-memory SQLite database with foreign keys enabled. This verifies syntax and document consistency, not application behavior or model quality.

## Implementation backlog

| ID | Owner | Priority | Task | Depends on | Status | Acceptance evidence to attach |
|---|---|---|---|---|---|---|
| T01 | A | P0 | Hardware budget and local four-frame smoke test | — | TODO | Environment/config, 20-request log, peak memory |
| T02 | C | P0 | Safe fixtures, labels, split manifest | — | IN_PROGRESS | JSONL manifest contract and validator enforce source hashes, provenance/license, supported labels, interval validity, unique samples and source-level split isolation; actual footage and labels remain to collect |
| T03 | B | P0 | SQLite/API skeleton and validated contracts | — | DONE | `backend/storage.py`, `backend/contracts.py`, `backend/main.py`; `pytest -q` passes 5 tests covering unknown outcomes, merge deduplication, idempotent review, time filtering, fixture API and citations |
| T04 | B | P0 | Paced replay/webcam, timestamps, ring buffer | T03 | IN_PROGRESS | Real replay runs in a bounded background worker with safe local path resolution and start/status/stop APIs; webcam capture remains pending |
| T05 | A | P0 | CPU detection/tracking and temporal candidates | T04 | IN_PROGRESS | Added a lazy YOLO adapter, deterministic CPU IoU tracker, normalized zone/gate state, one-shot line crossings, and candidate integration in `backend/perception/pipeline.py`; real weights and ByteTrack comparison remain a GPU-owner handoff |
| T06 | A | P0 | VLM worker, bounded scheduler and audits | T01,T05 | IN_PROGRESS | `backend/inference/scheduler.py` adds bounded single-worker admission, visible queue overflow, expiry, and handler failure states; scheduler tests pass; real VLM worker/audit loop pending |
| T07 | B | P0 | Passages and simulated grant reconciliation | T03,T05 | IN_PROGRESS | SQLite passage/grant operations use stable dedupe keys and transactional grant consumption; retry test passes, while normal unmatched/two-grant cases remain pending |
| T08 | B | P0 | Event merging, keyframes, event/earlier-context clip export | T03,T04,T06 | IN_PROGRESS | CPU orchestrator now turns current-frame candidates into needs-review incidents, merges repeats, saves a keyframe, closes after quiet gap, and exports an honestly bounded clip. Compressed 60-second context pinning and VLM confirmation remain |
| T09 | C | P0 | Dashboard, evidence drawer, visible failure states | T03,T08 | IN_PROGRESS | Dashboard includes replay controls, worker state, grounded query panel, full incident fetch, ready keyframe/clip playback, and explicit missing/pending/failed evidence states; browser walkthrough with real detector output remains |
| T10 | B | P0 | Query intent, list/count/group and citations | T03,T06 | IN_PROGRESS | Analyst chat now keeps conversation history, offers grounded quick prompts, maps natural language into six controlled categories, and returns clickable citations/clarification suggestions; query tests cover the expanded taxonomy. Frozen retrieval evaluation remains |
| T11 | B | P0 | Confirm/reject reviews, retention and recovery | T08 | TODO | Retry/restart/expiry test results |
| T12 | A+C | P0 | Frozen B0/C1 comparison and error analysis | T02,T06,T08 | IN_PROGRESS | Added deterministic one-to-one event scorer and CLI for raw TP/FP/FN, per-class precision/recall, false alerts/hour, and p50/p95 latency. Frozen labeled footage and real B0/C1 predictions remain |
| T13 | All | P0 | Soak, offline rehearsal and demo backup | T09,T10,T11,T12 | TODO | 30-minute log, disconnected run, labeled recording |
| T14 | All | P1 | Second camera, RTSP, zone editor, refinement | T13 | TODO | Admit only with spare time and repeated resource test |
| T15 | A | P1 | Optional vLLM/speculative-decoding feasibility trial | T12 | TODO | Profile first; CUT if unsupported, no headroom or no end-to-end gain |

## Measurements

| Measurement | Current value |
|---|---|
| Actual free VRAM / app budget | Not measured |
| Selected model revision/runtime/quantization | Not selected; initial candidate in techspec |
| Source/session counts in validation and test | Not collected |
| Per-class event precision and recall | Not measured |
| Normal footage hours and false alerts | Not measured |
| B0/C1 calls per minute and recall difference | Not measured |
| Alert p50/p95 and missed-event count | Not measured |
| Peak memory and missed audit deadlines | Not measured |
| Chat exact-match and citation validity | Not measured |

## Decision/change log

| Date | Decision | Reason |
|---|---|---|
| 2026-10-15 | Single-camera first; laptop remains the hard cap | User specified 48 hours, three people, shared RTX 3060 |
| 2026-10-15 | Defer detector switching and paper reproduction | Integration/VRAM cost; no local benefit yet demonstrated |
| 2026-10-15 | Three observable categories and explicit access context | Demonstrable semantics and useful hard negatives |
| 2026-10-15 | One DB, separate media files, bounded retention | Keep retrieval and implementation simple |
| 2026-10-15 | A owns GPU runs; B owns CPU capture/backend; C owns UI/data/scoring | Only A has a GPU; independent development must remain possible |
| 2026-10-15 | Treat screenshot/meeting technologies as suggestions | Explicit user clarification; model and runtime remain measured choices |
| 2026-10-15 | Preserve bounded earlier context, beyond default event clip | User requested inspection of where an incident actually started |

When changing scope, append the reason and update affected contracts. Record blockers with their owner and next concrete action; do not mark blocked work complete to satisfy the schedule.
