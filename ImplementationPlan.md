# 48-hour implementation plan

Status: ready to build; implementation has not started. Team: three people. Hardware ceiling: one shared laptop RTX 3060, with other GPU users present. Hours below are elapsed project time, not a recommendation to work without rest.

Read [prd.md](prd.md) for scope, [techspec.md](techspec.md) for implementation contracts, [Schema.md](Schema.md) for data, and [Tracker.md](Tracker.md) for current status. Research decisions and sources are in [Research.md](Research.md).

## Current implementation order

The next implementation sequence is intentionally narrower than the original
48-hour plan:

1. **Crowd counting**
2. **Visual bounding boxes in displayed video**

Do not begin overlay work until crowd counting has a deterministic fixture path,
stable event behavior, and a visible count in the API/UI. Do not use the count
as a claim about detector accuracy until labeled footage is available.

### Milestone N1 — Crowd counting

**Goal:** turn person detections/tracks into a stable, reviewable crowd count
for each replay window and incident candidate.

#### Work order

1. Confirm the count contract:
   - count unique active person tracks, not raw detections;
   - record the count timestamp/window and camera/session;
   - preserve the configured crowd threshold and duration;
   - distinguish `observed_count`, `threshold_reached`, and
     `crowd_gathering` event state;
   - retain fixture versus real-detector provenance.
2. Validate the existing temporal rule with deterministic fixtures:
   - people entering and leaving the zone;
   - tracker-ID changes without double-counting;
   - short gaps and sustained gaps;
   - count exactly at, below, and above threshold;
   - no count when the frame is missing or invalid.
3. Persist the count and evidence metadata with the event/window without
   changing the existing review/lifecycle semantics.
4. Expose the count through the backend event/list/detail response.
5. Display the current count, configured threshold, and uncertainty/fixture
   state in the dashboard.

#### N1 acceptance criteria

- A CPU fixture produces the same count and event result on repeated runs.
- A track-ID change does not create a second person in the same active window.
- Threshold and duration behavior are tested at boundary values.
- API responses identify the count window, observed count, threshold, and
  whether the result came from a fixture or a real detector.
- The UI shows count and threshold without implying verified accuracy.
- Existing event merging, review, media, and query tests remain passing.

### Milestone N2 — Bounding boxes in displayed video

**Dependency:** N1 must be complete enough that displayed boxes can be tied to
the same timestamped detections/tracks used by the count.

**Goal:** show visual detection evidence on replay/display frames while keeping
the original video and raw detections available.

#### Work order

1. Define a versioned overlay contract:
   - source PTS/frame identifier;
   - image/video dimensions used for coordinates;
   - `x1`, `y1`, `x2`, `y2` in source-pixel coordinates;
   - class name, confidence, track ID, and fixture provenance;
   - optional candidate/category association.
2. Add deterministic validation:
   - coordinates must be finite and within source dimensions;
   - `x1 < x2` and `y1 < y2`;
   - boxes from another frame cannot be silently reused;
   - missing detections stay missing;
   - malformed detector output remains an explicit failure.
3. Implement overlay generation on the backend using the existing frame/PTS
   path, preserving the clean media asset.
4. Expose overlay data or an annotated media variant through the existing API
   without leaking filesystem paths.
5. Render boxes in the dashboard display/evidence view with:
   - correct scaling for the displayed aspect ratio;
   - frame/timestamp synchronization;
   - class/track labels only where available;
   - an explicit unavailable state when detections are absent.
6. Test the same frame and coordinate mapping in CPU fixtures before any GPU
   detector run.

#### N2 acceptance criteria

- A fixture frame with known source dimensions renders a box at the expected
  displayed location and scale.
- The overlay follows the selected frame timestamp and does not drift to a
  neighboring frame.
- Clean video remains available; overlay failure does not destroy media.
- Invalid coordinates, missing frames, and unavailable detections are visible
  failures rather than silently repaired output.
- API, frontend, and persistence tests cover the overlay contract.
- A real replay walkthrough shows count, frame, box, timestamp, and evidence
  state together.

### Scope guard

This sequence does not add new incident categories, fine-tuning, hosted
inference, a vector database, multi-camera support, or a new video model.
Detector quality, count accuracy, and incident recall remain unvalidated until
the team creates a labeled evaluation split. GPU inference is not required for
N1/N2 CPU fixtures; any real-model measurements must be recorded separately.

## 1. What to build to stand out

Build a complete incident investigation workflow: **observe a short sequence, explain the visible evidence and camera policy, preserve a clip, then retrieve the incident with a cited answer**. Make three things convincing:

1. **Compute allocation:** compare periodic VLM inference with candidate-triggered inference plus periodic audits. Show both saved calls and missed incidents.
2. **Context:** demonstrate two people entering after one simulated grant versus two people entering after two grants. Explain the policy difference without claiming to identify an intruder.
3. **Evidence:** show an incident's timestamps, full-scene frame, playable clip, uncertainty, and database-backed answer. Include a hard negative and a failure case.

This is an engineering contribution, not a claim to have invented a new anomaly-detection algorithm. A reliable measured prototype is the priority. Do not add detector switching, fine-tuning, a vector store, separate hot/cold databases, or paper reproduction to the critical path.

## 2. Owners and integration boundaries

Use A/B/C until actual names are assigned. **Only A has a GPU.** B and C develop on CPU using portable fixtures; the final integrated stack runs on A's laptop. Detailed handoffs and module ownership: [TeamAssignments.md](TeamAssignments.md). Decisions on the meeting questions and screenshots: [MeetingDecisions.md](MeetingDecisions.md).

| Owner | Primary responsibility | Deliverable/interface |
|---|---|---|
| A: perception/inference | CPU detector/tracker adapter, candidate rules, VLM worker, scheduler, all real GPU runs | Timestamped analysis results, real fixtures and measured model configuration |
| B: backend/video/data | Capture/replay, timestamps/ring, SQLite, passage/grant matching, event merging, media export, query filters | Frame packets for A; event/media/query/status API for C |
| C: UI/evaluation | Dashboard, fixtures/labels, test runner/report, demo | Evidence-first UI, frozen evaluation manifest, reproducible comparison |

At H0 agree enums and JSON examples. B supplies fake development fixtures so C can build before inference works; visibly label them as fixtures and exclude them from measured results. A and B integrate a real event by H10. All three review a small common label set so annotation rules agree.

## 3. Schedule and checkpoints

| Hours | A | B | C | Exit condition |
|---|---|---|---|---|
| 0–2 | Measure available VRAM/CPU; run four-frame SmolVLM2 smoke test | Create service skeleton, schema migration, local paths and configuration | Select/record safe staged scenarios; define labels and simple dashboard | One local inference returns usable output; hardware budget recorded |
| 2–6 | Detection/tracking; model adapter; recorded output fixtures | Paced replay/webcam, timestamped ring; event/media APIs, SQLite writer and grant fixtures | Player, event list, evidence drawer; annotate tuning fixtures | Frames flow at measured cadence; synthetic API fixtures render |
| 6–10 | One category through temporal rule → VLM → validated observation | Real event persistence; keyframe and pre/post-roll clip | Connect real SSE/API; verify timestamps and media | **G1: one real video → event → playable evidence** |
| 10–16 | Add remaining two categories and low-rate audits | Dedup/state machine, grant consumption, query DSL and deterministic SQL | Label independent test sessions; filter/chat view; error states | Three categories exercised; query lists the actual event |
| 16–22 | Profile frames/tokens; shared-worker query scheduling; optional one-model upgrade trial | Query adapter, counts/grouping, review endpoint and earlier-context export | Hard-negative tests, summary rubric, baseline runner | **G2: complete single-camera P0 flow; model/config frozen provisionally** |
| 22–28 | Tune rules only on validation split; compare periodic vs cascade | Reconnect/crash/media-state/retention fixes | Freeze test manifest; reproduce integration failures | Config, prompts and splits frozen; rest/handoff rotation |
| 28–34 | Run controlled baseline and cascade evaluations | Check event/query results; gather timing/resource logs | Score predictions, review summaries, generate report | **G3: actual measurements with raw counts and limitations** |
| 34–40 | Fix blocking reliability problems; retest only affected areas | Persistence recovery and query regression checks | UI polish, failure examples, presentation evidence | All P0 paths pass or reduced scope is declared |
| 40–44 | Run 30-minute paced soak with normal competing GPU use | Test disconnected operation and restart | Rehearse three-minute demo; record backup run | **G4: demo works with network off; backup clearly labeled** |
| 44–48 | Freeze models/dependencies; resolve only critical bugs | Package launch instructions/config and evidence | Final report, demo script, provenance, honest limitations | Reproducible handoff and working presentation |

Schedule includes buffers and rotation; avoid treating 144 theoretical person-hours as guaranteed productive capacity. Stop adding features after H34.

## 4. First two hours: resolve the real risks

1. Record GPU model, total VRAM, free VRAM under expected background applications, OS, CPU, RAM, driver, Python, PyTorch/CUDA, and model/runtime revisions.
2. Establish the memory admission budget from techspec. Use one resident VLM. Start with SmolVLM2-500M and four bounded images; no giant video input or long reasoning output.
3. Run at least 20 short representative requests, recording full-response latency, output validity, memory, and visible quality. Include a normal scene. Separate initial loading/warm-up from steady state.
4. If output is poor, permit one bounded trial of Qwen3-VL-2B 4-bit. Spend at most 90 minutes on that integration. Choose based on validation observations and available memory, not model popularity.
5. If neither configuration meets the budget, reduce image count/resolution and supported source cadence, document the quality tradeoff, and use the working smaller model. Never silently switch to a hosted API.
6. Obtain one real camera frame and a paced replay frame with trustworthy timestamps. Start downloads early; cache all assets needed for an offline demo.

Do not promise a fixed FPS or alert latency before this checkpoint. Model card and paper measurements are not laptop measurements.

## 5. Scope cuts, in order

P0: paced file replay and webcam, one active camera, three categories, saved zone/gate presets, evidence, structured list/count/group queries, confirm/reject review, evaluation.

P1 only after G3: second simulated camera, interactive zone drawing, RTSP-specific reconnect support, richer category correction UI, selective crop refinement. vLLM/speculative decoding are optional profiling-driven experiments with the cutoffs in MeetingDecisions, not required technologies. Configure zones in a JSON file before building a drawing editor.

If behind schedule, cut in this order: P1 features → animation/polish → free-form answer generation → optional model comparison → advanced frame selection. Keep the local natural-language intent parser and deterministic answers, even if the supported question vocabulary is narrow. If a category fails its validation gate, mark it experimental or remove it from the delivered claim; do not replace failures with handpicked predictions. The brief permits team-defined categories, but scope changes must be reflected in all documents and the demo.

At H10 without a real end-to-end event, stop all model exploration. At H22 without stable chat parsing, restrict intents and use explicit clarification/filter controls. At H34 without meaningful results, prioritize the baseline/cascade comparison over any second camera.

## 6. Evaluation data and labels

Minimum feasible target: **60 short clips from independent source recordings/sessions**, about 12 positive examples per category and 24 hard negatives, plus at least 30 minutes of continuous normal footage. This is a collection target, not a claim that the data already exists. If fewer are available, report the smaller denominator honestly.

Use team-recorded footage as the core benchmark. Use approximately half for tuning and half for final testing: target six test positives per category and twelve test negatives. Keep entire original videos, recording sessions, and near-duplicate takes in a single split; never random-split adjacent frames. Staged scenarios should be safe: a volunteer lies down rather than performing a dangerous fall. Public footage, if allowed, is an optional separately reported supplement. Record source, terms, hash, and source grouping. Use a different session/view where feasible to expose overfitting. Creating our own evaluation dataset does not imply training a model from scratch.

Hard negatives: bending to pick something up, sitting/crouching, exercise, a person already reclining, an orderly queue, people briefly passing a zone, opposite-direction crossings, two valid grants, missed/partial access logs, blur, and temporary occlusion. Use explicit ambiguity labels where the video cannot establish the answer.

Each label records:

```json
{
  "clip_id": "test-entry-03",
  "source_group": "recording-session-07",
  "split": "test",
  "source_sha256": "TO_BE_FILLED_FROM_FILE",
  "category": "suspected_tailgating",
  "start_pts_ms": 6000,
  "end_pts_ms": 11000,
  "earliest_observable_pts_ms": 7500,
  "policy_version": "gate-v1",
  "access_log_source": "simulated",
  "access_log_complete": true,
  "expected_observations": ["two distinct same-direction passages", "one grant"],
  "ambiguity": false
}
```

`earliest_observable` is when the required temporal evidence first exists, distinct from event onset. Review all disagreements and a sample of agreed labels with a second teammate. Human review performed after inference does not overwrite frozen test labels without a documented annotation correction and rerun.

## 7. Experiments

| Run | Configuration | Purpose |
|---|---|---|
| B0 | Same VLM, four uniformly spaced causal frames, periodic request every 2 seconds if capacity allows | Simple reference; log any missed deadlines/overload |
| C1 | Same model/frame format, cheap candidates plus audits every 10 seconds initially | Isolate scheduling/call reduction |
| C2, optional | C1 plus evidence-guided temporal selection at the same frame/request budget | Isolate frame-selection effect |

The frame budget/model are controlled; timestamps and number of requests differ by the scheduling treatment. Keep detector, event-merging, output schema, test footage, and scoring rules fixed between B0 and C1. Use untrimmed replay where available so the cascade is not handed ground-truth incident boundaries. Scheduling a VLM every two seconds does not mean it can finish that quickly. If B0 overloads, report its actual missed deadlines and additionally compare at the fastest sustainable fixed cadence.

Measure both real-time operation under the same constraints and, if useful, offline classification with full processing. Never mix offline recall with real-time latency into one supposedly live result. Run B0 and C1 sequentially on the same laptop, not simultaneously. Record competing GPU use and thermal state. Reverse run order once if time permits; do not run dozens of redundant trials.

## 8. Metrics and scoring rules

- **Event precision/recall per class:** match predictions to ground truth one-to-one by camera, category, and temporal IoU at least 0.3; report 0.5 as a sensitivity check if feasible. Extra duplicate alerts are false positives. Freeze the matching rule before test evaluation.
- **Alert versus review:** score accepted predictions separately, then show accepted-plus-needs-review recall and review burden. Abstention must not vanish from the denominator. Report ambiguity cases separately.
- **Gate recall:** fraction of labeled events that generated a candidate window intersecting them. Separately count audit-only recoveries and events neither route inspected. Inspect low detector recall as a failure, not only VLM errors.
- **False alerts:** accepted false positives divided by hours of continuous normal footage. Report raw numerator and exposure, e.g. `0 alerts / 0.5 h observed`, not a claim of a zero false-alarm rate in deployment.
- **Summaries:** human-check visible action, temporal consistency, camera/zone accuracy, and unsupported claims. Report factual errors and omissions separately on all test predictions or a prespecified sample.
- **Latency:** record onset-to-first-alert and earliest-observable-to-first-alert; separately measure queue, preprocess, inference-to-complete-output, validation, database write, and clip readiness. Show p50/p95, sample count, and missed-event count. Missed events have no successful latency and remain visible in recall.
- **Cost/coverage:** VLM calls/minute, input frames/tokens, peak process VRAM, total GPU use, detector cadence, oldest queue age, skipped frames/windows, missed audits, and evidence availability. A low call count from a broken pipeline is not an optimization.
- **Chat:** freeze 24 questions spanning list, counts, group-by-camera/category, relative time, explicit time, empty results, ambiguity, and unsupported requests. Compare expected event-ID sets/counts and parsed filters. Every returned citation must resolve to an existing event. Include expired clips and coverage gaps.

Target claim gate, chosen before testing: at least 40% fewer VLM calls than the sustainable periodic baseline, with no more than one additional missed test incident overall and no class losing more than one. Also report exact per-class differences: the tiny set makes percentage changes unstable. These are decision thresholds, not achieved results or statistical proof. If the cascade fails, ship the better measured configuration and explain the tradeoff.

Provisional operational targets: no OOM during a 30-minute soak, no unbounded queue, p95 first-alert delay within 8 seconds of earliest observable evidence on the selected demo workload, and 100% resolvable event citations. Change a target only with an explicit reason, before interpreting the final test. Missed targets go in the report.

## 9. Meaningful verification

Implement small deterministic tests for timestamp/interval filtering (including midnight), one-to-one event merge behavior, grant consumption retry idempotency, review projections, and media state transitions. Use fake model outputs for these tests; test real models separately on footage.

Integration probes: disconnect/reconnect, seek/restart replay, model timeout, malformed JSON, full candidate queue, failed media write, expired clip, repeated review request, repeated crossing callback, and restart with pending evidence. Assert a failure cannot produce a false normal result or an invented playable clip. Test normal crossings that consume grants without creating incidents.

Suggested reproducibility artifacts to create during implementation:

- `configs/demo.json`, model/runtime lockfile, and source hashes.
- `evaluation/manifest.jsonl`, frozen labels, questions, and split manifest.
- `evaluation/runs/<run_id>/config.json`, predictions, timing logs, metrics, and error examples.
- `reports/evaluation.md` with actual counts, not placeholder performance claims.
- A launch README and a labeled recording of the working demo.

## 10. Presentation sequence

Use the three-minute sequence in [Appflow.md](Appflow.md). Lead with a real event and evidence, then the policy contrast, then a cited question. Finish with the measured B0/C1 table and one honest limitation. Bring a prior-run recording in case the laptop or camera fails, and label it as recorded. Do not imply organizer affiliation beyond participation or imply production safety certification.
