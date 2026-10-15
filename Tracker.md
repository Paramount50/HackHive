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

Stage 3: candidate rules now reset sustained-posture/count evidence after an observed
break or a gap longer than 1500 ms; crowd counting uses unique tracks. Posture uses
pixel aspect ratio on non-square sources. Later gate-crossing pairs are no longer
hidden behind the first pair in history. All 46 tests pass, including four added
regressions. This checks deterministic behavior, not detector accuracy; tracking
fragmentation, ByteTrack integration, and policy-specific tuning remain unvalidated.

Stage 4: scheduler now coalesces candidate/audit updates, limits each camera to two
pending candidates, prioritizes urgent candidates/aged audits, serves a waiting audit
after four candidates, and counts expired/overloaded work by reason. Queue mutation
and worker execution are separately serialized; explicit handler failures are preserved.
All 50 tests pass. Counters are process-local; B still needs durable coverage logging,
and the application still needs the periodic audit producer and model-worker wiring.

Stage 5: CUDA PyTorch 2.9.1+cu126 and Transformers 4.57.6 are installed in `.venv`;
CUDA reports available and identifies the RTX 3060 Laptop. Model classes import,
`pip check` reports no broken requirements, and all 52 tests pass. The model weights
and CUDA wheel were downloaded with SHA256-verified resumable chunks after full-file
transfers stalled. `requirements-gpu.txt` records the runtime/import-tested versions.
The first pilot in `reports/vlm-pilot-20261015/` stopped at RAM admission before
loading weights (2860 MiB available, 3072 MiB required). This is not a model inference
failure or a successful benchmark; the raw failure, source hashes and hardware are saved.

Stage 6: added matching torchvision 0.24.1+cu126 after the processor exposed that
dependency. Real GPU generation now works. Prompt-only pilots (`r3`, `v2`) returned
zero valid outputs out of two measured requests each. Local constrained decoding
with lm-format-enforcer 0.11.3 produced 2/2 valid outputs in the `json` pilot, but both
were `unknown` and repeated the example wording. This demonstrates runtime/format
success, not useful recognition. Peak allocated GPU memory was 1106.82 MiB; measured
pilot request times were approximately 5.02 and 6.36 seconds. All 54 tests pass.
GPU admission now takes the lower of NVIDIA and PyTorch free-memory reports because
they differed on this Windows host. The 20-request smoke is being recorded separately.

Stage 7: installed pinned Ultralytics 8.3.203 and local YOLO11n weights. ReplayManager now passes real frames to a per-session detector/event/media pipeline and finalizes active incidents at replay end. The React dashboard exposes detector/replay mode and optional gate-line configuration. A real 33-frame `v001` replay created crowd candidates and ready keyframes/clips; this revealed duplicate crowd events from changing track sets, so crowd identity was changed to a stable per-session zone key. A 49-frame `v026` replay completed without an event, showing that the person-down path is not yet validated. After integration changes, 57 tests passed and the React production build passed. A human-labeling CLI was added; the six sample videos still have no confirmed labels or rights/provenance record. The VLM remains unconnected to replay events.

Post-fix rerun: the same 33-frame `v001` replay now creates one merged `crowd_gathering` event, finalizes it as `ended`/`needs_review`, exports a ready keyframe and event clip, and returns one query citation. The replay policy (zone and optional gate line) is stored with the session. Current verification is 61 passing tests plus a successful frontend production build. These are integration checks, not accuracy results.

Stage 8: integrated the pinned local XCLIP action ranker into replay as overlapping
two-second/eight-frame windows. `aggression_fight` is an explicit review-only
category; raw and normalized scores, latency, model revision, prompt version, and
abstentions are retained. The six supplied clips all replayed to completion: 29
action windows ranked 13 `aggression_fight`, 12 `person_down`, one
`crowd_gathering`, and three `unknown`. Eight total events (including two YOLO
crowd-rule candidates) remained `needs_review`; all had ready keyframes and clips.
The recorded pilot is in [action-pilot-20261016.json](reports/action-pilot-20261016.json).
This is an unlabelled smoke/integration result, not verified accuracy or evidence
of generalization.

Audit correctness stage — 2026-10-16: event detail and review lookup now query the
requested event ID directly, so events outside the newest 100 can still be opened
and reviewed. Pillow 12.3.0 is declared in the base requirements because the CPU
video sampling tests import it. Verification: 62 passed, 4 GPU-dependent skipped;
the remaining warning is the upstream Starlette/AnyIO deprecation.

Upload stage — 2026-10-16: the dashboard now uploads MP4/MOV/AVI/MKV bytes instead
of sending an assumed server filename. `POST /api/uploads` streams with a configurable
250 MiB default cap, writes through a temporary file, validates OpenCV decoding,
generates the stored name, hashes the file, records dimensions/FPS/duration in SQLite,
and cleans up rejected uploads. Upload and replay remain separate so a recording can
be analyzed repeatedly. Verification: 64 passed, 4 GPU-dependent skipped, plus the
React production build; no detector-accuracy claim follows from upload validation.

Evidence discovery stage — 2026-10-16: incident-list queries expose a ready keyframe
media ID and playable-evidence flag without leaking paths. Timeline cards render the
thumbnail or an explicit unavailable state before the evidence drawer is opened. The
React category vocabulary now matches all six backend incident categories.

Sampling experiment stage — 2026-10-16: sequential `read()` remains the default
correctness baseline; replay can opt into OpenCV `grab/retrieve`, and
`scripts/benchmark_sampling.py` compares wall time, CPU time, emitted count, and PTS
sequence on the same file. No general speedup is claimed because codec backends may
still decode compressed frames internally. One uncommitted console trial on
`v001_converted 3.avi` emitted the same 33 PTS values: sequential used 0.766 s wall /
0.797 s CPU and grab used 0.281 s wall / 0.453 s CPU. This single warm-cache run is
directional only; repeat trials across the frozen videos before changing the default.

Live-state stage — 2026-10-16: the dashboard policy banner and footer now derive from
`GET /api/status` instead of claiming fixture mode during real replay. While a replay
is active, status and incidents refresh every second so newly persisted events become
visible before completion. Polling is bounded to the active job lifetime; SSE remains
a later transport optimization.

Discovery-filter stage — 2026-10-16: `GET /api/events` now accepts repeated category
and camera filters plus review status and playable-evidence availability. SQLite
applies review projection from the latest review and evidence existence before the
result limit. The dashboard exposes multi-category, review, and evidence controls.

End-to-end acceptance stage — 2026-10-16: a TestClient journey now uploads real AVI
bytes, registers the recording, starts background replay, crosses an injected detector
boundary, sustains a person-down rule for three sampled seconds, creates one
needs-review incident, and retrieves its ready keyframe and event clip. The substitute
detector proves application wiring only; it does not validate YOLO accuracy.

VLM connection stage — 2026-10-16: P0 replay candidates can now submit four ordered
frames to the pinned SmolVLM worker through the bounded single-worker scheduler without
blocking replay. Category-specific evidence requirements produce `supported`,
`insufficient_evidence`, `contradicted`, `category_conflict`, or `inconclusive`; every
outcome stays `needs_review`. A separate SQLite connection records the model attempt,
timing, provenance, raw structured output, and verdict on the existing incident.
Verification: 70 passed, 4 GPU-dependent skipped. CPU tests use an injected worker;
real replay-to-VLM execution and quality remain a GPU-owner validation gate.

Swift Sampling stage — 2026-10-16: candidate verification now supplies a bounded
causal pool of up to 16 frames. SmolVLM extracts globally pooled attention-key features
from vision layer 0, applies the authors' third-order relative Taylor residual and
local-maximum selection, and forwards four chronological frames to generation. The
selection strategy, pool size, indices, source PTS, Taylor order, and latency are stored
with the VLM analysis window. NumPy tests cover residual selection and service tests
cover selected-frame handoff. Verification: 73 passed, 4 GPU-dependent skipped.
This `16 → 4` short-window adaptation has not been run on the GPU or shown to improve
event recall; the paper evaluates longer pools and different VLM backbones.

Sampler-ablation stage — 2026-10-16: `HACKHIVE_VLM_SAMPLING=swift|uniform`
selects the candidate-frame policy at service startup, and `/api/status` reports the
active choice. Both paths retain the same four-frame generation budget and store their
selection provenance, enabling a controlled GPU comparison without code changes.
Verification after adding the uniform baseline: 74 passed, 4 GPU-dependent skipped.

Local query-model stage — 2026-10-16: the existing `/api/query` and Analyst Chat now
support validated `structured`, `semantic`, and `hybrid` retrieval. Parameterized
SQLite filtering runs before semantic ranking; counts and groups never come from the
model. Pinned Qwen3-0.6B may route descriptive questions and phrase answers from a
bounded event context, while pinned all-MiniLM-L6-v2 ranks event documents on CPU.
Application code owns citations and resolves ranked IDs back through SQLite; deleted
events and changed event text cannot silently survive the in-memory embedding cache.
Model absence, malformed routing and load failure retain deterministic structured
queries. The React chat exposes retrieval mode, source count, model, fallbacks and
semantic score without changing evidence-drawer behavior. Added `backend/chat.py`,
`scripts/setup_chat_models.py`, `scripts/evaluate_chat.py`, the frozen question-set
template and deterministic fake-model tests. Current code validation passes without
model downloads: 83 tests passed, four GPU-dependent tests skipped, the React
production build passed, both scripts compiled, and a model-free evaluator dry run
completed. Real Qwen/MiniLM smoke, latency and RAM remain unverified on this
checkout and must not be reported as measured results.

Status values: `TODO`, `IN_PROGRESS`, `BLOCKED`, `DONE`, `CUT`. An item becomes DONE only when its acceptance evidence is linked. Owners A/B/C are placeholders defined in [ImplementationPlan.md](ImplementationPlan.md).

Only A has GPU access. See [TeamAssignments.md](TeamAssignments.md) for CPU-only development and handoffs, and [MeetingDecisions.md](MeetingDecisions.md) for organizer-reference decisions. GPU smoke and local CPU detector runs exist, but no measured accuracy claim is supported.

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
| T01 | A | P0 | Hardware budget and local four-frame smoke test | — | IN_PROGRESS | GPU is available; 20-request smoke returned 18 valid JSON outputs, all `unknown`; no quality result |
| T02 | C | P0 | Safe fixtures, labels, split manifest | — | IN_PROGRESS | Manifest validator and labeling CLI added; actual labels, source rights/provenance, and splits still to collect |
| T03 | B | P0 | SQLite/API skeleton and validated contracts | — | DONE | `backend/storage.py`, `backend/contracts.py`, `backend/main.py`; `pytest -q` passes 5 tests covering unknown outcomes, merge deduplication, idempotent review, time filtering, fixture API and citations |
| T04 | B | P0 | Paced replay/webcam, timestamps, ring buffer | T03 | IN_PROGRESS | Real replay runs in a bounded background worker; frames now reach the detector/event/media callback; webcam remains pending |
| T05 | A | P0 | CPU detection/tracking and temporal candidates | T04 | IN_PROGRESS | YOLO11n weights and Ultralytics are present; detection runs on sampled frames, but per-class event quality and ByteTrack comparison remain unvalidated |
| T06 | A | P0 | VLM worker, bounded scheduler and audits | T01,T05 | IN_PROGRESS | `backend/inference/scheduler.py` adds bounded single-worker admission, visible queue overflow, expiry, and handler failure states; scheduler tests pass; real VLM worker/audit loop pending |
| T07 | B | P0 | Passages and simulated grant reconciliation | T03,T05 | IN_PROGRESS | SQLite passage/grant operations use stable dedupe keys and transactional grant consumption; retry test passes, while normal unmatched/two-grant cases remain pending |
| T08 | B | P0 | Event merging, keyframes, event/earlier-context clip export | T03,T04,T06 | IN_PROGRESS | Real keyframe/clip path exercised; repeated crowd candidates now share stable per-session zone identity and must be rerun; earlier context and VLM confirmation remain |
| T09 | C | P0 | Dashboard, evidence drawer, visible failure states | T03,T08 | IN_PROGRESS | Replay mode and detector state are exposed; production build passes; real-event browser walkthrough remains |mains |
| T10 | B | P0 | Query intent, list/count/group and citations | T03,T06 | IN_PROGRESS | Deterministic SQLite filters/counts, Qwen3-0.6B routing/answer adapter, CPU MiniLM hybrid ranking, real-event citations and frozen evaluation runner are implemented; pinned real-model smoke and labeled expected event IDs remain |
| T11 | B | P0 | Confirm/reject reviews, retention and recovery | T08 | TODO | Retry/restart/expiry test results |
| T12 | A+C | P0 | Frozen B0/C1 comparison and error analysis | T02,T06,T08 | IN_PROGRESS | Added deterministic one-to-one event scorer and CLI for raw TP/FP/FN, per-class precision/recall, false alerts/hour, and p50/p95 latency. Frozen labeled footage and real B0/C1 predictions remain |
| T13 | All | P0 | Soak, offline rehearsal and demo backup | T09,T10,T11,T12 | TODO | 30-minute log, disconnected run, labeled recording |
| T14 | All | P1 | Second camera, RTSP, zone editor, refinement | T13 | TODO | Admit only with spare time and repeated resource test |
| T15 | A | P1 | Optional vLLM/speculative-decoding feasibility trial | T12 | TODO | Profile first; CUT if unsupported, no headroom or no end-to-end gain |
| T16 | A | P1 | User-requested action classification pilot | T02,T04,T08 | IN_PROGRESS | Six integrated replays/29 windows and deterministic tests recorded in `reports/action-pilot-20261016.json`; independently verified labels, false-positive evaluation, held-out-camera and normal-footage results are still required |

## Measurements

| Measurement | Current value |
|---|---|
| Actual free VRAM / app budget | 4838 MiB free in a 6144 MiB RTX 3060 Laptop snapshot; provisional 3814 MiB after reserve; no soak measurement |
| Selected model revision/runtime/quantization | SmolVLM2-500M pinned revision and local runtime tested; not quality-validated or production-selected |
| Source/session counts in validation and test | Six AVI sources; no confirmed labels or source-separated frozen splits |
| Per-class event precision and recall | Not measured |
| Normal footage hours and false alerts | Not measured |
| B0/C1 calls per minute and recall difference | Not measured |
| Alert p50/p95 and missed-event count | Not measured |
| Peak memory and missed audit deadlines | 1106.82 MiB PyTorch allocated in one smoke; audit deadlines not integrated/measured |
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
