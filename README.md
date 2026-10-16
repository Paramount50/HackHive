# HackHive local event desk

## Local Qwen3 analyst chat and hybrid retrieval

`POST /api/query` keeps validated `QueryIntent` filters and parameterized SQLite
queries as the source of truth. Exact counts, grouping, categories, cameras, time
ranges and review state are calculated by application code. Descriptive questions
can additionally rank the filtered real events with CPU MiniLM embeddings; Qwen3-0.6B
selects a validated retrieval mode. Answers and totals are generated deterministically
from the parsed filters and database results; the language model does not generate
incident facts or citations. Application code creates citations from retrieved event
IDs. Raw video and frames are never embedded.

The parser accepts ISO dates, complete clock ranges, camera and human-review filters,
and source-offset ranges such as `entrance.mp4 from 00:42.400 to 00:49.200`.
Ambiguous or unsupported constraints ask for clarification rather than being silently
discarded; VLM verdicts are not currently query filters, while human-review state is.
Count and category/camera grouping use exact SQL aggregates. Event lists
report the exact number matching structured filters; semantic ranking still searches
at most the newest 100 structured candidates and reports that scan cap separately.

The optional local semantic-retrieval models are pinned to exact revisions:

- `Qwen/Qwen3-0.6B` for non-thinking retrieval-mode routing.
- `sentence-transformers/all-MiniLM-L6-v2` for small CPU sentence embeddings using
  the existing Transformers/PyTorch runtime directly.

Download once while online, then run an explicit local smoke test:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
.\.venv\Scripts\python.exe scripts\setup_chat_models.py --download
.\.venv\Scripts\python.exe scripts\setup_chat_models.py --smoke --device cpu
```

Weights and provenance are stored under the Git-ignored
`models/qwen3-0.6b/` and `models/all-minilm-l6-v2/` directories. Chat defaults to
CPU so it does not compete with SmolVLM/XCLIP for the laptop GPU. Set
`HACKHIVE_CHAT_DEVICE=cuda` only for a measured isolated trial. Set
`HACKHIVE_CHAT_ENABLED=0` to disable both local chat models; supported structured
queries continue using the deterministic parser and SQLite. Override locations with
`HACKHIVE_CHAT_MODEL_DIR` and `HACKHIVE_EMBEDDING_MODEL_DIR`.

`GET /api/status` reports query-service `state=ready` when structured query and
database retrieval are available; `semantic_models_state` separately reports
`not_configured`, `available`, `ready`, or `disabled`. `available` means the pinned
provenance files exist; only `ready` means both adapters actually loaded. A load or
malformed-routing failure falls back to deterministic behavior when the question has
sufficient structured filters. No hosted inference, model-generated SQL, executable
output, LangChain, or vector database is used.

Run the reproducible query evaluation after replacing each `null`
`expected_event_ids` entry with labels from the database snapshot being tested:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_chat.py evaluation\chat_questions.json `
  --database data\hackhive.sqlite --with-models `
  --output reports\chat-evaluation.json
```

The evaluator records routing, parsed filters, returned event IDs, citation validity,
clarification behavior and latency. Unlabeled retrieval cases are excluded from its
retrieval-accuracy count. Do not report chatbot accuracy until the event IDs are
frozen and labeled for the chosen database snapshot.

## Person A: local GPU smoke test

The standalone worker is `backend/inference/smolvlm.py`. It serializes requests,
loads only local safetensors/configuration, and returns observations rather than
automatically accepting incidents. Replay frames can run through the local YOLO
detector, temporal candidate rules, event storage, and evidence exporter. The VLM
worker is connected as an optional candidate-verification path through one bounded
background queue. When the pinned local model is installed, a new P0 candidate submits
four chronological frames; its structured result is attached to that incident and
remains review-only. Replay verification is submitted after the clip ends so its
four selected frames can be drawn from the complete incident interval and context.
XCLIP action scores are
also aggregated across temporal windows, allowing a brief altercation to take
precedence over crowd-only frames while keeping the result review-only. Model
absence or failure leaves the CPU candidate path available.
The CPU rules detect candidate `person_down`, `crowd_gathering`, and
`suspected_tailgating` events. An optional local XCLIP zero-shot action ranker adds
reviewable `aggression_fight`, `person_down`, and `crowd_gathering` candidates.
Its scores are uncalibrated, not verified labels, and do not establish production
accuracy or generalization. Tailgating remains geometry/policy based.

From the HackHive directory, using a Python 3.12 virtual environment. The 2.2B
candidate is fully wired, but the operational default remains 500M until a
successful Kaggle/T4 runtime smoke promotes it; both variants remain selectable:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_vlm.py --hardware-only
# One-time network setup for either pinned variant:
.\.venv\Scripts\python.exe scripts\smoke_vlm.py --model 500m --download
.\.venv\Scripts\python.exe scripts\smoke_vlm.py --model 2.2b --download
# Local-only inference after setup; keep models and videos on disk:
.\.venv\Scripts\python.exe scripts\smoke_vlm.py --model 2.2b --videos ..\sample_videos --requests 20
.\.venv\Scripts\python.exe scripts\smoke_vlm.py --model 500m --videos ..\sample_videos --requests 20
```

Candidate-conditioned verification uses a category-specific prompt and schema:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_vlm.py `
  --model 2.2b --candidate aggression_fight `
  --videos ..\sample_videos --requests 3
```

Without `--candidate`, the script remains an independent low-rate audit smoke
using the controlled multiclass taxonomy. Candidate runs record the candidate,
category-specific observations, deterministic verification verdict, raw output,
timestamps, latency, and peak VRAM. This is not an accuracy evaluation.
Candidate mode requires a fixed `evidence` and `evidence_frames` entry for every
predicate in the selected category. The application still normalizes that
response into its existing observations contract and retains the raw JSON.
The model sees only local frame labels `0..3`; source PTS values remain internal
metadata and are persisted in the request provenance. Frame indices are
constrained to `0..3`; repeated references are collapsed only when the complete
citation list is otherwise valid and local, with a normalization note and the raw
response retained. Out-of-range, malformed, empty, or oversized references are
rejected, never clamped or removed; constrained decoding does not replace downstream validation. Candidate
summaries are deliberately limited to one concise sentence rather than relying
on a larger generation budget.
Each report records whether constrained JSON decoding was enabled, the Git
revision when available, and SHA256 fingerprints for the smoke script and
inference sources. Check `run_config.json` and `code_provenance.json` before
interpreting a remote run, especially when notebook files were copied manually.

For a fair structural/performance comparison, create and reuse one deterministic
window manifest:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_vlm.py --model 500m --videos ..\sample_videos --requests 20 --window-manifest reports\comparison-windows.json
.\.venv\Scripts\python.exe scripts\smoke_vlm.py --model 2.2b --videos ..\sample_videos --requests 20 --window-manifest reports\comparison-windows.json
```

Both reports then use the same source names and frame fractions; source hashes,
selected frame indices, prompts, validation failures, timings, and peak memory
remain separate per variant. This is not an accuracy comparison until the windows
have human labels.

The worker's runtime dependencies are in `requirements-gpu.txt`. CUDA availability
and model-class imports have been verified with the pinned versions on this laptop;
model generation is a separate validation gate. Each run records its full
`environment.txt`. Use the official PyTorch CUDA wheel index for Windows.

If large full-file downloads stall, `scripts/download_gpu_assets.py torch` and
`scripts/download_gpu_assets.py model:500m` resume bounded chunks and verify official
SHA256 digests. The sharded 2.2B checkpoint is downloaded with
`smoke_vlm.py --model 2.2b --download`.
The CUDA wheel is retained under `data/wheels/` for offline reinstall; model weights
stay under `models/`. Both are excluded from Git.

Generation uses local JSON-constrained decoding and a 384-token ceiling, followed
by independent semantic validation. `--unconstrained` enables a diagnostic comparison.
Prompt-only pilots failed formatting; constrained pilots returned valid JSON but
abstained (`unknown`). Valid JSON is not detection accuracy, and this model configuration
must not auto-accept incidents. Inspect outcome counts and raw responses before integration.
The controlled visual outcomes are `person_down`, `crowd_gathering`,
`suspected_tailgating`, `aggression_fight`, `fire_smoke`,
`intrusion_unauthorized_access`, `none`, and `unknown`. `none` means the scene is
visible but no controlled incident is supported; `unknown` means the frames or
context are insufficient. Required category evidence is checked independently:
visual restricted-area evidence cannot establish authorization, and a different
model category remains a category conflict rather than changing the stored event.
Person-down, crowd, tailgating and XCLIP aggression candidates are integrated
with candidate-conditioned verification. Fire/smoke and restricted-area
intrusion remain contract/audit-only because they do not yet have production
candidate generators. None of the categories has a frozen labelled accuracy
evaluation yet.

Each run creates a new `reports/vlm-<UTC timestamp>/` directory containing source
hashes, hardware, exact model/prompt/runtime configuration, raw outputs and failures,
per-request timings, peak PyTorch memory, and summary counts. Warm-up is separate.
There is no hosted inference fallback or model-generated SQL. Replay verification
writes the raw structured attempt, timing, model provenance, and deterministic evidence
verdict to the existing incident; it never auto-confirms an event.
Malformed JSON is a failed analysis, never a normal scene. No automatic retry is
used in the smoke test, so first-attempt validity is visible.

These are offline sampled windows, not a real-time benchmark or event precision/recall
evaluation. Windows use four distinct frames spanning up to four seconds, and PTS
is estimated from frame index/FPS. Label footage and verify timestamps before formal
evaluation. Smaller clips may yield repeated windows across requests. Source rights,
split membership, and expected events must be recorded before benchmark use.

Integrated candidate verification forms a causal pool of at most 16 replay frames.
When the pool contains more than four frames, it applies third-order Swift Sampling
to globally pooled layer-0 SmolVLM attention-key features and sends four selected
frames to generation. The incident record keeps the pool size, selected indices/PTS,
strategy, Taylor order, and selection latency. This is a short-window adaptation of
[Swift Sampling](https://arxiv.org/abs/2605.22678), whose published gains concern
long-video benchmarks and different VLMs; GPU compatibility and event recall remain
unverified here. Exactly four available frames are all retained.
Irregular timestamps, incompatible vision internals, or invalid selector output trigger
a recorded uniform fallback instead of failing the incident-verification request.
`HACKHIVE_VLM_SAMPLING` defaults to `swift`; set it to `uniform` before starting the
service to run the same four-frame-budget baseline for an ablation.

Admission requires at least 3 GiB free system RAM and a variant-specific GPU budget:
1800 MiB for 500M and 6144 MiB for 2.2B (with an 8192 MiB allocator cap), with a 1 GiB free-VRAM reserve. The 2.2B
floor is a conservative preflight based on the official model-card estimate of about
5.2 GB GPU RAM for video inference, not a measured success claim. It targets the
Kaggle T4 first and is not a claim that 2.2B fits the 6 GiB RTX 3060 Laptop.
PyTorch allocator limits do not include all driver/context memory. The adapter uses
bfloat16 + SDPA and disables image splitting; each variant uses its native longest
edge (500M: 512, 2.2B: 384).
See the official [2.2B model card](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct),
[500M model card](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)
and [processor documentation](https://huggingface.co/docs/transformers/v4.57.1/model_doc/smolvlm).

Without detector weights, replay is deliberately reported as `decode_only`; it does
not create incidents. Candidate events from YOLO/rules remain unreviewed even after
VLM verification. Tailgating is disabled unless a gate line is supplied, and the default
crowd zone is the whole frame; configure geometry for each camera before interpreting
those categories.

Set `HACKHIVE_VLM_VARIANT=500m` or `2.2b` to select a variant; the application
defaults to 500M until the 2.2B runtime gate passes. Set `HACKHIVE_VLM_MODEL_DIR` to override its
variant directory. `GET /api/status` reports the VLM state,
queue depth, failures, and model provenance. A `ready` worker means the runtime loaded;
it is not an accuracy claim.

## Real video replay setup

Install the detector dependency and fetch the small YOLO11n weights once while online:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('models/yolo11n.pt')"
```

The backend creates `data/ultralytics` and configures `YOLO_CONFIG_DIR` there
before loading Ultralytics, avoiding writes to protected per-user Windows settings.
Set `YOLO_CONFIG_DIR` before starting the backend only if you need another
writable settings directory.

For action-ranking candidates, install the GPU dependencies and download the pinned
XCLIP model once while online:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
.\.venv\Scripts\python.exe scripts\download_action_model.py
```

The action ranker runs on CUDA when available and otherwise on CPU. Its overlapping
two-second windows are sampled as eight ordered frames, once per second. Predictions
retain the raw class scores, normalized ranking scores, model revision, prompt version,
and an explicit uncalibrated limitation. Low-margin/low-score windows abstain as
`unknown`; normal `none`/`unknown` windows are logged, not turned into incidents.
All model-generated incidents remain `needs_review`. Local weights remain under the
Git-ignored `models/` directory.

The first integrated pilot replay completed all six supplied clips and evaluated
29 action windows: 13 ranked `aggression_fight`, 12 `person_down`, one
`crowd_gathering`, and three abstained as `unknown`. Every replay-created event was
unreviewed and had a ready keyframe and event clip. Warm per-window inference latency
was 176 ms p50 / 281 ms p95 on this RTX 3060 laptop; the first request was 1.50 s.
These are unlabelled observations; model-only inference latency excludes model loading,
video decoding, YOLO, and evidence export. The results are not precision/recall or
proof that categories are correct. The full scores, timings, and event evidence status are in
[`reports/action-pilot-20261016.json`](reports/action-pilot-20261016.json).

To replay the supplied clips directly, point the service at their folder before
starting it:

```powershell
$env:HACKHIVE_RECORDINGS_DIR = (Resolve-Path ..\sample_videos).Path
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Use a source filename from that folder in the dashboard. The optional gate-line field
is a normalized horizontal line expressed as a percentage from the top; leave it blank
unless that line actually corresponds to a gate. The service reports `decode_only` if
weights are missing, and the replay status reports the analysis mode and event count.

## Run locally

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000 to inspect API status. The React dashboard provides
deterministic fixture buttons plus real local replay controls. Runtime SQLite data
and generated evidence are written under `data/` and must not be committed.

The API includes `/api/status`, `/api/events`, `/api/events/{id}`,
`/api/events/{id}/reviews`, `/api/query`, `/api/replays`, and fixture endpoints.
The replay status explicitly distinguishes real detector analysis from decode-only.

To run a real replay, place a video under `recordings/` (or configure
`HACKHIVE_RECORDINGS_DIR`) and start a bounded background job:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/replays `
  -ContentType 'application/json' `
  -Body '{"source_name":"demo.mp4","sample_fps":5,"pace":true}'
```

Poll `GET /api/replays/{session_id}` for progress and the latest sampled frame.
Use `POST /api/replays/{session_id}/stop` to interrupt a paced replay. Source
paths are confined to `recordings/`, and decoding runs outside the request thread.

## React dashboard

The dashboard is in `frontend/` and uses Vite with an API proxy for local
development:

```powershell
cd frontend
npm install
npm run dev
```

Start the FastAPI service first. Open http://127.0.0.1:5173 to use the React
dashboard. Query and timeline results share evidence cards with source, camera,
video-offset interval, review and VLM state, and keyframe. The evidence drawer exposes
the verification verdict, predicates, actual sampling strategy, and durable VLM-selected
frames with source offsets. Playable event clips seek to the incident start while
showing available and missing requested coverage. Replay timestamps are labeled as
replay times, not original CCTV capture times. Replay events retain a video offset
even if the recording filename was not associated with that session; the source is
then shown as unavailable. Exact totals and readable filters are
shown separately from semantic retrieval's bounded candidate scan.
New evidence clips use VP9/WebM for browser playback; older MP4 event clips are
converted to WebM on first playback and retain their media IDs and coverage metadata.

VLM confirmation remains advisory and requires human review; webcam/RTSP input and
validated event accuracy are still incomplete. A model ranking on any supplied clip
is a pilot observation only. The six clips have provisional labels and no held-out
camera coverage.

The unreviewed timeline keeps every candidate in the operator queue, but puts VLM
contradictions, inconclusive/insufficient results, and verification failures first.
VLM support is an advisory model assessment, not a human review decision.

## Label the supplied videos before evaluating

The sample AVI files are useful for decoder/inference smoke tests, but their event
labels and rights/provenance are not established. Review each video in a player and
record only clear event intervals or confirmed-normal footage:

```powershell
.\.venv\Scripts\python.exe scripts\label_videos.py `
  --videos ..\sample_videos `
  --output evaluation\sample_manifest.jsonl
.\.venv\Scripts\python.exe scripts\validate_manifest.py evaluation\sample_manifest.jsonl
```

The tool asks for provenance and permission/license basis separately for each unique
source; do not guess or reuse a global answer when sources differ. Use `s` for an
ambiguous clip rather than labeling it normal. The generated entries are marked
`human_confirmed`; provisional agent drafts must use `annotation_status: provisional`
and will not count as negatives or be considered freeze-ready by validation. Keep all
windows from a source in the same split. The labeling tool covers the integrated person-down, crowd,
tailgating, and aggression categories; experimental fire/smoke and restricted-area
intrusion are excluded. Byte-identical source files are skipped, and manifest
validation rejects identical content recorded under different source IDs, preventing
duplicate clips from leaking across splits or inflating the sample count. Six short
clips are only a pilot set; do not claim robust accuracy or deployment readiness
from them.

`evaluation/sample_manifest.provisional.jsonl` records labels for the judge-provided
clips. The user confirmed aggression is present in all six; the v026 interval
(00:00.000–00:02.000) is human-confirmed, while the other five interval boundaries
remain provisional visual estimates. The manifest is still not ready to freeze:
there is no confirmed-normal footage or held-out test source, so it cannot measure
false alerts per hour or support a held-out accuracy claim.

## Evaluation

Score a frozen JSON file containing `duration_ms`, `labels`, and `predictions`:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py evaluation\run.json --output evaluation\metrics.json
```

The scorer uses one-to-one same-source, same-category temporal matching and
reports raw counts alongside precision, recall, false alerts per hour, and
alert-latency percentiles. Commit the small manifests and metrics, not footage
or model weights.

Validate a source-separated JSONL dataset manifest before generating predictions:

```powershell
.\.venv\Scripts\python.exe scripts\validate_manifest.py evaluation\manifest.jsonl
```
