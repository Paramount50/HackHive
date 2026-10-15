# HackHive local event desk

## Person A: local GPU smoke test

The standalone worker is `backend/inference/smolvlm.py`. It serializes requests,
loads only local safetensors/configuration, and returns observations rather than
automatically accepting incidents. Replay frames can run through the local YOLO
detector, temporal candidate rules, event storage, and evidence exporter. The VLM
worker is connected as an optional candidate-verification path through one bounded
background queue. When the pinned local model is installed, a new P0 candidate submits
four chronological frames; its structured result is attached to that incident and
remains review-only. Model absence or failure leaves the CPU candidate path available.
The CPU rules detect candidate `person_down`, `crowd_gathering`, and
`suspected_tailgating` events. An optional local XCLIP zero-shot action ranker adds
reviewable `aggression_fight`, `person_down`, and `crowd_gathering` candidates.
Its scores are uncalibrated, not verified labels, and do not establish production
accuracy or generalization. Tailgating remains geometry/policy based.

From the HackHive directory, using a Python 3.12 virtual environment:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_vlm.py --hardware-only
# One-time network setup (model revision is pinned in the adapter):
.\.venv\Scripts\python.exe scripts\smoke_vlm.py --download
# Local-only inference after setup; keep models and videos on disk:
.\.venv\Scripts\python.exe scripts\smoke_vlm.py --videos ..\sample_videos --requests 20
```

The worker's runtime dependencies are in `requirements-gpu.txt`. CUDA availability
and model-class imports have been verified with the pinned versions on this laptop;
model generation is a separate validation gate. Each run records its full
`environment.txt`. Use the official PyTorch CUDA wheel index for Windows.

If large full-file downloads stall, `scripts/download_gpu_assets.py torch` and
`scripts/download_gpu_assets.py model` resume bounded chunks and verify official
SHA256 digests. Run `smoke_vlm.py --download` afterward to finish metadata setup.
The CUDA wheel is retained under `data/wheels/` for offline reinstall; model weights
stay under `models/`. Both are excluded from Git.

Generation uses local JSON-constrained decoding and a 256-token ceiling, followed
by independent semantic validation. `--unconstrained` enables a diagnostic comparison.
Prompt-only pilots failed formatting; constrained pilots returned valid JSON but
abstained (`unknown`). Valid JSON is not detection accuracy, and this model configuration
must not auto-accept incidents. Inspect outcome counts and raw responses before integration.

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

Admission requires at least 3 GiB free system RAM and 1800 MiB GPU budget, with the
budget capped at 4 GiB and a 1 GiB free-VRAM reserve. This is a conservative startup
gate, not a guarantee against competing applications. PyTorch allocator limits do
not include all driver/context memory. The adapter uses bfloat16 + SDPA, one native
512-pixel image tile per frame, and disables image splitting. This deliberately
differs from the proposed 384-pixel seed to match the model's native image tokens.
See the official [model card](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)
and [processor documentation](https://huggingface.co/docs/transformers/v4.57.1/model_doc/smolvlm).

Without detector weights, replay is deliberately reported as `decode_only`; it does
not create incidents. Candidate events from YOLO/rules remain unreviewed even after
VLM verification. Tailgating is disabled unless a gate line is supplied, and the default
crowd zone is the whole frame; configure geometry for each camera before interpreting
those categories.

Set `HACKHIVE_VLM_MODEL_DIR` if the pinned model is not stored at
`models/smolvlm2-500m`. `GET /api/status` reports the VLM state,
queue depth, failures, and model provenance. A `ready` worker means the runtime loaded;
it is not an accuracy claim.

## Real video replay setup

Install the detector dependency and fetch the small YOLO11n weights once while online:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('models/yolo11n.pt')"
```

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
dashboard. The dashboard displays replay frames, candidate incidents, and evidence when local
YOLO is ready. VLM confirmation, webcam/RTSP input, and validated event accuracy are
still incomplete. A model ranking on any supplied clip is a pilot observation only;
the six clips have no verified ground-truth annotations or held-out-camera coverage.

## Label the supplied videos before evaluating

The sample AVI files are useful for decoder/inference smoke tests, but their event
labels and rights/provenance are not established. Review each video in a player and
record only clear event intervals or confirmed-normal footage:

```powershell
.\.venv\Scripts\python.exe scripts\label_videos.py `
  --videos ..\sample_videos `
  --output evaluation\sample_manifest.jsonl `
  --provenance "<record how the source was obtained>" `
  --license "<record the permission/license basis>"
.\.venv\Scripts\python.exe scripts\validate_manifest.py evaluation\sample_manifest.jsonl
```

Use `s` for an ambiguous clip rather than labeling it normal. Keep all windows from a
source in the same split. Six short clips are only a pilot set; do not claim robust
accuracy or deployment readiness from them.

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
