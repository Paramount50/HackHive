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
abstentions are retained. In the six-source pilot, 29 windows ranked 13
`aggression_fight`, 12 `person_down`, one `crowd_gathering`, and abstained three
times. This remains an unlabelled smoke result, not verified accuracy.

Stage 9: local duplicate citations in an otherwise valid candidate-verification
response are deduplicated with an audit note; malformed or out-of-range references
remain invalid. The integrated v001 replay now completes crowd verification
(`supported`) instead of failing on duplicate frame references. Eight AVI files
were present in `sample_videos`, but SHA256 identified two as byte-identical copies
of named clips, leaving six unique sources. All eight files replayed (279 sampled
frames, 11 events); all 11 VLM attempts completed with no errors: eight verdicts
`supported`, three `contradicted`. All 22 keyframe/clip artifacts were ready.
The report records candidate and verification data, exact latencies, source hashes,
and runtime resource snapshots in
[integrated-sample-dir-replay-20261016.json](reports/integrated-sample-dir-replay-20261016.json).
Median VLM total latency was 13.67 s (range 12.83-18.94 s). At the end-of-run
snapshot, GPU use was 3716/6144 MiB (PyTorch peak allocated 1933.6 MiB) and
available system RAM was 3237/16236 MiB. Results are not accuracy measurements:
the data has no independently verified labels or normal-footage/held-out-camera
evaluation.

The YOLO adapter now creates `data/ultralytics` and sets `YOLO_CONFIG_DIR` before
importing Ultralytics, unless a different directory is configured. The store marks
ready media as missing when its file is absent, and event projections use the
actual `event_clip` media kind. Targeted citation/config/media tests pass; full
suite passes (148 passed); the frontend production build passes. The remaining
warnings are upstream Starlette/AnyIO deprecation and Windows pytest-cache
permission warnings.

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

SmolVLM smoke audit: local report artifacts showed 4/6 responses failing after successful
generation because duplicate `visible_person` predicates were emitted; two outputs passed
validation but were semantically incorrect for the unlabelled footage. Outputs were below
the 256-token cap (207-213 tokens in the reproduced failures), so truncation is not
supported by this evidence. Updated the prompt/schema to request and constrain distinct
observation objects, and now record JSON/validation stage plus EOS/capacity diagnostics.
Verification: 84 tests passed, 5 GPU-dependent tests skipped. This is a structural
reliability fix only; no accuracy claim was added.

Kaggle audit follow-up: verification now distinguishes a required predicate explicitly
reported false (`contradicted`), explicitly unknown (`inconclusive`), and omitted
(`insufficient_evidence`). A schema-valid `person_down` response with false
`persistence`/`visible_person` evidence or no observations therefore cannot be treated
as semantically supported. Raw model output remains stored; the model output is not
rewritten. The supplied fight clip remains an unsupported accuracy benchmark until
the expanded contract is rerun on the same frozen windows. The unverified gun
statement remains an unconfirmed model claim because source frames were not supplied
here.

Generation-constraint audit: `generation_schema()` previously declared JSON Schema
`uniqueItems`, but the Kaggle LMFE 0.11.3 parser accepted exact duplicate observation
objects. The local development environment does not install LMFE, so its corresponding
test is skipped locally and cannot establish a version comparison. The unenforced
keyword was removed from the LMFE-facing schema. Duplicate predicates remain a
downstream `invalid_model_output` failure through Pydantic; raw output is preserved and
never silently deduplicated. Verification: the duplicate-parser regression now asserts
this boundary explicitly.

VLM variant upgrade: added selectable pinned configurations for the existing 500M
checkpoint and SmolVLM2-2.2B-Instruct revision
`482adb537c021c86670beed01cd58990d01e72e4`. The application and smoke CLI default to
the 500M configuration until a successful Kaggle/T4 runtime smoke promotes 2.2B,
while `--model 2.2b` remains available for explicit testing and same-window
comparison. Provenance, model directory, variant, prompt, contract, timings, and
hardware are recorded per run. The 2.2B admission floor is 6144 MiB and is not a
claim that it fits the 6 GiB laptop; Kaggle T4 execution remains GPU-unverified.
The official config exposes the expected SmolVLM vision path, and the adapter now
records an explicit Swift Sampling compatibility check with uniform fallback through
the existing service. No accuracy claim is supported without labelled windows.

Visual taxonomy expansion: the SmolVLM contract now supports the six controlled
incident categories plus `none` and `unknown`. Added category-specific observable
predicates and required-evidence checks for physical altercation, flame/smoke, and
restricted-boundary crossing. Authorization remains external policy evidence;
`intrusion_unauthorized_access` is presented in the UI as possible restricted-area
intrusion. Verification: 15 focused contract/semantic tests passed; full-suite and
frontend verification remain to be run. No accuracy improvement has been measured;
the same fight-video windows still require a GPU rerun.

Candidate-conditioned verification: integrated VLM verification now uses a
category-specific prompt and single-category generation schema derived from the
candidate already produced by rules/XCLIP/policy. The VLM cannot replace an
aggression candidate with intrusion or another incident during verification.
The generic multiclass path remains available for explicit low-rate audit smoke.
Candidate smoke runs are supported with `--candidate`, and persisted verification
payloads retain candidate category, raw output, validation diagnostics, EOS/token
diagnostics, observations, and deterministic verdict. Fire/smoke and restricted
intrusion still require documented upstream candidate/policy coverage; the same
fight windows require a fresh candidate-mode GPU rerun.

Candidate evidence completeness follow-up — 2026-10-16: candidate mode now
requires explicit true/false/null evidence and frame citations for every
category predicate. For aggression this is
`multiple_people_visible`, `physical_contact`, `aggressive_motion`, and
`temporal_persistence`; the deterministic `supported` verdict now requires all
four to be true. Person-down requires visible person, prone posture and
persistence. Fire/smoke requires scene visibility, temporal persistence and at
least one of flame-like or smoke-like evidence. Crowd count/duration remain
tracker facts; a VLM disagreement blocks support without replacing those facts.
Authorization remains outside the visual contract. Regression tests cover
true/false/null decision evidence across all six schemas. Generic audit mode
retains the optional observations list. Out-of-range frame references remain
invalid and are never repaired. Verification: `python -m pytest -q` completed
with 134 passed and one Starlette deprecation warning. GPU behavior was not
exercised; the newest Kaggle candidate run remains pending.

Candidate frame-reference follow-up — 2026-10-16: candidate prompts no longer
expose source PTS values to SmolVLM; they label images only as local frame
indices `0..3`. Source PTS values remain in the result and selection
provenance. Candidate summaries now request one concise sentence and use a
shorter schema bound. The schema declares frame bounds, but downstream
validation remains authoritative because LM Format Enforcer numeric-bound
support is not treated as proven. Truncated JSON remains an invalid output;
token count, EOS, max-token, and raw-output diagnostics remain recorded. The
newer Kaggle run still needs to be repeated to separate its exact truncation
cause from the source-PTS confusion.

Aggression replay integration — 2026-10-16: XCLIP
`aggression_fight` candidates now enter the same bounded, category-conditioned
SmolVLM verification service used by the original candidate categories. A
deterministic integration test exercises XCLIP candidate creation, event
persistence, `SemanticJob` submission with the expected aggression outcome,
the semantic worker callback, and storage of a `supported` verdict on the same
event. Verification: `python -m pytest -q` completed with 135 passed and one
Starlette deprecation warning. This is orchestration evidence only; the newest
2.2B Kaggle smoke and RTX 3060 co-residency measurements remain pending.

Person-down gap handling — 2026-10-16: posture persistence now tolerates a
short missing detection while retaining the last timestamped horizontal
observation. An explicit upright observation resets the run immediately, and a
later timestamp beyond `max_observation_gap_ms` expires it. Regression tests
cover both the tolerated-miss and upright-reset paths. Verification:
`python -m pytest -q` completed with 136 passed and one Starlette deprecation
warning. This is deterministic rule coverage, not measured person-down recall.

Intrusion authorization semantics — 2026-10-16: the compatible API category
identifier remains `intrusion_unauthorized_access`, while its UI label remains
“Possible restricted-area intrusion.” Visual-only intrusion events now persist
authorization as `unknown`, and the evidence drawer reports that no
policy/access evidence exists instead of implying visual authorization.
Verification: `python -m pytest -q` completed with 137 passed and one warning;
the Vite production build completed successfully. No intrusion candidate
generator or authorization integration is claimed.

Authoritative event summaries — 2026-10-16: a supported VLM result no longer
replaces the event summary with free-form model prose. Storage generates the
event-facing summary from the fixed candidate category and controlled true
predicates. The original model summary, raw output, model revision and prompt
provenance remain in the immutable analysis-window payload for audit.
Verification: `python -m pytest -q` completed with 137 passed and one warning.

Visible VLM lifecycle — 2026-10-16: each semantic attempt now has a stable ID
and persists `pending`, `running`, `completed`, `rejected`, `failed`, `expired`
or `not_configured` in one upserted analysis window. Event API projections
expose the latest state, verdict, evidence, model/prompt provenance, sampling,
timings and errors. Queue coalescing removes the superseded semantic job instead
of leaking it; deadline expiry and missing configuration remain distinct from a
contradicted candidate. Tests cover pending/running/completed, coalescing,
expiry, not-configured and failed persistence. Verification:
`python -m pytest -q` completed with 140 passed and one warning.

Evidence completeness — 2026-10-16: the compatibility
`confidence_score` field now uses `evidence_completeness_v2`, derived from the
current answered controlled predicates for the event category and recalculated
when supported VLM evidence is attached. The dashboard no longer presents this
internal completeness value as an operator-facing score or probability.
Verification: `python -m pytest -q` completed with 140 passed and one warning;
the Vite production build completed successfully.

Evidence-first incident desk — 2026-10-16: event API projections now expose
candidate source and media availability alongside the persisted VLM lifecycle.
Timeline cards show candidate source, review state, VLM verdict, evidence
availability, camera/time and category maturity. The evidence drawer shows
controlled predicates with local frame citations, sampling strategy/pool/
selection latency, model revision/inference latency, authorization separately,
human review controls and real limitations. Runtime health distinguishes YOLO,
XCLIP, SmolVLM state/queue/model variant, sampling and chat availability; it
does not imply accuracy. Chat citations continue to open the same drawer and
raw filters are no longer prominent. Sampled JPEG clips are labeled “Sampled
evidence clip.” Selected VLM images are not displayed because accessible frame
copies are not currently persisted. Verification: `python -m pytest -q`
completed with 140 passed and two dependency/deprecation warnings; the Vite
production build completed successfully.

Local candidate-output hardening — 2026-10-16: a fresh RTX 3060 500M
aggression smoke showed that hiding source PTS alone was insufficient: the
initial run produced 1/3 valid measured outputs, with invalid large/negative
frame references and two 256-token truncations. Frame indices are now generated
from the explicit enum `0,1,2,3`, the bounded ceiling is 384 tokens, and candidate
schemas correctly remove the unused `observations` property. An intermediate
run reached 2/3 and exposed that stray-property schema defect. The final identical
three-request run produced 3/3 structurally valid outputs, all four predicates
per request, local frame indices only, no max-token hits, 12.250 s median,
15.483 s p95 and 1106.52 MiB peak PyTorch allocation. All three deterministic
verdicts were `contradicted`; sources are unlabelled, so this is not an accuracy
result. Raw artifacts are in `reports/vlm-20261016T011647Z`,
`reports/vlm-500m-aggression-20261016-enum384` and
`reports/vlm-500m-aggression-20261016-schemafix`. The required 2.2B Kaggle
rerun and RTX 3060 2.2B+XCLIP co-residency measurement remain pending because
the local checkout contains only the 500M model.

Audit regression follow-up — 2026-10-16: application-owned count/group queries
no longer rewrite to a 100-row list, and aggregate storage reads are not capped;
a 101-event regression test verifies the exact count. XCLIP sliding windows now
reuse the active follow-up occurrence after the original event ends instead of
creating one event per stride. XCLIP runtime exceptions persist a failed
analysis window while CPU candidates continue through the same replay.
Verification: `python -m pytest -q` completed with 143 passed and two warnings.

RTX 3060 integrated replay smoke — 2026-10-16: `v001_converted 3.avi` ran
through the real replay endpoint at 5 sampled FPS with pacing disabled. The
6.43-second source completed with 33 sampled frames, two events and no replay
error. XCLIP produced an `aggression_fight` candidate (normalized score 0.593,
uncalibrated) and a `crowd_gathering` candidate (normalized score 0.403,
uncalibrated). SmolVLM2-500M completed aggression verification in 13.675 seconds
with a deterministic `contradicted` verdict. Crowd verification ran for 13.113
seconds but was persisted as `failed` because the model returned duplicate
`evidence_frames`, violating the distinct-local-index contract. This is a
runtime smoke result, not an accuracy result. After both XCLIP and the 500M VLM
were resident, the system snapshot showed 3536 MiB total GPU memory in use,
2459 MiB free and 2.92 GiB available system RAM; these are whole-system
snapshots, not isolated peak allocations. Ultralytics also required
`YOLO_CONFIG_DIR` to point inside the writable project on this managed Windows
runtime.

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

## Next implementation order

The next two milestones are now explicitly ordered:

1. Crowd counting with deterministic temporal fixtures, persisted count/threshold
   evidence, API exposure, and dashboard display.
2. Visual bounding boxes in displayed video, using the same timestamped
   detection/track records and a validated source-pixel overlay contract.

Bounding-box work depends on the crowd-counting contract and fixture coverage.
Neither milestone may claim detector accuracy without labeled footage.

| ID | Owner | Priority | Task | Depends on | Status | Acceptance evidence to attach |
|---|---|---|---|---|---|---|
| T01 | A | P0 | Hardware budget and local four-frame smoke test | — | IN_PROGRESS | GPU is available; 20-request smoke returned 18 valid JSON outputs, all `unknown`; no quality result |
| T02 | C | P0 | Safe fixtures, labels, split manifest | — | IN_PROGRESS | Labeling CLI covers integrated categories including aggression, collects provenance/license per unique source, and skips byte-identical files; manifest validation rejects duplicate content under different source IDs and prevents provisional labels from being frozen; judge-provided sample draft exists, but needs human label confirmation, normal footage, and a held-out test source |
| T03 | B | P0 | SQLite/API skeleton and validated contracts | — | DONE | `backend/storage.py`, `backend/contracts.py`, `backend/main.py`; `pytest -q` passes 5 tests covering unknown outcomes, merge deduplication, idempotent review, time filtering, fixture API and citations |
| T04 | B | P0 | Paced replay/webcam, timestamps, ring buffer | T03 | IN_PROGRESS | Real replay runs in a bounded background worker; frames now reach the detector/event/media callback; webcam remains pending |
| T05 | A | P0 | CPU detection/tracking and temporal candidates | T04 | IN_PROGRESS | YOLO11n weights and Ultralytics are present; detection runs on sampled frames, but per-class event quality and ByteTrack comparison remain unvalidated |
| T06 | A | P0 | VLM worker, bounded scheduler and audits | T01,T05 | IN_PROGRESS | Candidate-conditioned SmolVLM runs through replay; 11/11 attempts completed in the integrated report; quiet-window audit producer and durable coverage logs remain |
| T07 | B | P0 | Passages and simulated grant reconciliation | T03,T05 | IN_PROGRESS | SQLite passage/grant operations use stable dedupe keys and transactional grant consumption; retry test passes, while normal unmatched/two-grant cases remain pending |
| T08 | B | P0 | Event merging, keyframes, event/earlier-context clip export | T03,T04,T06 | IN_PROGRESS | Six unique sources replayed with one event per occurrence in the recorded run; 22/22 event artifacts ready; earlier-context retention remains |
| T09 | C | P0 | Dashboard, evidence drawer, visible failure states | T03,T08 | IN_PROGRESS | Verification state, predicates, citations, model provenance, media availability and review actions are exposed; production build passes; real browser walkthrough remains |
| T10 | B | P0 | Query intent, list/count/group and citations | T03,T06 | IN_PROGRESS | Deterministic SQLite filters/counts, Qwen3-0.6B routing/answer adapter, CPU MiniLM hybrid ranking, real-event citations and frozen evaluation runner are implemented; pinned real-model smoke and labeled expected event IDs remain |
| T11 | B | P0 | Confirm/reject reviews, retention and recovery | T08 | TODO | Retry/restart/expiry test results |
| T12 | A+C | P0 | Frozen B0/C1 comparison and error analysis | T02,T06,T08 | IN_PROGRESS | Added deterministic one-to-one event scorer and CLI for raw TP/FP/FN, per-class precision/recall, false alerts/hour, and p50/p95 latency. Frozen labeled footage and real B0/C1 predictions remain |
| T13 | All | P0 | Soak, offline rehearsal and demo backup | T09,T10,T11,T12 | TODO | 30-minute log, disconnected run, labeled recording |
| T14 | All | P1 | Second camera, RTSP, zone editor, refinement | T13 | TODO | Admit only with spare time and repeated resource test |
| T15 | A | P1 | Optional vLLM/speculative-decoding feasibility trial | T12 | TODO | Profile first; CUT if unsupported, no headroom or no end-to-end gain |
| T16 | A | P1 | User-requested action classification pilot | T02,T04,T08 | IN_PROGRESS | Six unique source hashes/29 XCLIP windows and integrated verification report recorded; independently verified labels, false-positive evaluation, held-out-camera and normal-footage results are still required |

## Measurements

| Measurement | Current value |
|---|---|
| Integrated XCLIP + SmolVLM resource snapshot | End of six-source run: 3716/6144 MiB GPU used, 2279 MiB free, PyTorch peak allocation 1933.6 MiB; 3237/16236 MiB system RAM available (not a soak measurement) |
| Selected model revision/runtime/quantization | SmolVLM2-500M pinned revision and local runtime tested; not quality-validated or production-selected |
| Source/session counts in validation and test | Six unique AVI sources (eight files including two byte-identical upload copies); no confirmed labels or source-separated frozen splits |
| Per-class event precision and recall | Not measured |
| Normal footage hours and false alerts | Not measured |
| B0/C1 calls per minute and recall difference | Not measured |
| Alert p50/p95 and missed-event count | Not measured |
| Peak memory and missed audit deadlines | Latest integrated XCLIP + 500M replay: 3716 MiB total GPU memory used, 2279 MiB free, PyTorch peak allocation 1933.6 MiB; audit deadlines not measured |
| Chat exact-match and citation validity | Not measured |

Dataset review — 2026-10-16: inventory of `sample_videos` found ten files
representing six byte-distinct sources, totaling 38.799 seconds. The user confirmed
the clips were supplied by hackathon judges and permitted for project evaluation,
and confirmed aggression is present in all six. The v026 aggression interval
(00:00.000–00:02.000) is human-confirmed; the other five interval boundaries remain
provisional visual estimates. The user notes v026 has poor quality and a misleading
angle: two people, including one dragging another, are visible in the first two
seconds; only one person is visible afterward, with a person on the floor in the
final seconds. All sources are assigned to `validation` because
they were already used in system smoke/replay; they are not held-out test footage.
Manifest validation reports six sources, five provisional intervals, one
human-confirmed interval, zero confirmed negatives, and `ready_to_freeze: false`;
normal footage and held-out test sources are still missing. No accuracy or
false-alert claim is supported.

Kaggle 2.2B candidate smoke follow-up — 2026-10-16: the user-shared
`aggression_fight` run loaded SmolVLM2-2.2B but produced 0/3 structurally valid
measured responses. The shared output reports normal EOS completion, about
11.1 seconds/request, and 4712.56 MiB peak PyTorch allocation; one supplied raw
response cited frame `10` four times, outside the allowed local indices `0..3`.
The shared validation message says duplicate citations, whereas the current parser
checks bounds before duplicates; its generated property order also differs from
the current constrained schema. The exact Kaggle revision and whether
`--unconstrained` was used have not been established, so this is a code/runtime
provenance discrepancy, not evidence to weaken local citation validation. Smoke
reports now save `code_provenance.json`, constrained-decoding mode, Git revision
when available, and hashes of inference sources to make the next rerun attributable.
This 2.2B candidate smoke remains failed pending a rerun on confirmed current code.

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
