# Build tracker

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
| T02 | C | P0 | Safe fixtures, labels, split manifest | — | TODO | Source hashes, labels, non-overlapping source groups |
| T03 | B | P0 | SQLite/API skeleton and validated contracts | — | DONE | `backend/storage.py`, `backend/contracts.py`, `backend/main.py`; `pytest -q` passes 5 tests covering unknown outcomes, merge deduplication, idempotent review, time filtering, fixture API and citations |
| T04 | B | P0 | Paced replay/webcam, timestamps, ring buffer | T03 | IN_PROGRESS | `backend/ingest/replay.py` adds ordered fixture packets, a latest-frame slot and bounded context ring; replay tests pending |
| T07 | B | P0 | Passages and simulated grant reconciliation | T03,T05 | IN_PROGRESS | SQLite passage/grant operations are implemented with stable dedupe keys and transactional consumption; focused retry tests pending |
| T05 | A | P0 | CPU detection/tracking and temporal candidates | T04 | IN_PROGRESS | `backend/perception/rules.py` provides typed detections and CPU candidate rules for person-down, crowd-gathering, and suspected-tailgating; focused rule tests pass; detector adapter remains pending |
| T06 | A | P0 | VLM worker, bounded scheduler and audits | T01,T05 | TODO | Real output, timeout/overload behavior, missed-audit log |
| T07 | B | P0 | Passages and simulated grant reconciliation | T03,T05 | TODO | Retry and normal-crossing tests |
| T08 | B | P0 | Event merging, keyframes, event/earlier-context clip export | T03,T04,T06 | TODO | One incident, one ID, seekable context, partial state |
| T09 | C | P0 | Dashboard, evidence drawer, visible failure states | T03,T08 | IN_PROGRESS | React/Vite dashboard added under `frontend/`; `npm run build` succeeds and browser smoke test verified fixture list, category filter, evidence drawer, and review controls; real media walkthrough pending |
| T10 | B | P0 | Query intent, list/count/group and citations | T03,T06 | TODO | Frozen question results and SQL-filter tests |
| T11 | B | P0 | Confirm/reject reviews, retention and recovery | T08 | TODO | Retry/restart/expiry test results |
| T12 | A+C | P0 | Frozen B0/C1 comparison and error analysis | T02,T06,T08 | TODO | Configs, predictions, timing logs, raw metrics |
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
