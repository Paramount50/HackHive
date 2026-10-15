# HackHive local event desk

## Person A: local GPU smoke test

The standalone worker is `backend/inference/smolvlm.py`. It serializes requests,
loads only local safetensors/configuration, and returns observations rather than
automatically accepting incidents. The existing replay API is not yet wired to it.
The six-category application taxonomy remains unchanged; the VLM boundary currently
implements only the three planned P0 categories plus `none` and `unknown`.

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
There is no hosted inference fallback, model-generated SQL, or automatic DB write.
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

The first runnable slice is a CPU-only deterministic fixture path. It deliberately
does not claim real model inference: fixture responses are marked in API responses,
status, and query coverage. The inference adapter can be replaced by the local
perception worker in the next milestone without changing event/query storage.

## Run locally

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000, click **Run deterministic fixture**, then inspect,
query, and review the event. Runtime SQLite data is written under `data/` and
must not be committed.

The API includes `/api/status`, `/api/events`, `/api/events/{id}`,
`/api/events/{id}/reviews`, `/api/query`, `/api/fixtures/run`, and
`/api/fixtures/perception-run`. The perception fixture passes deterministic
CPU detections through temporal candidate rules before storing an event.

To run a real replay, place a video under `recordings/` and start a bounded
background job:

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
dashboard. It currently displays fixture incidents and their evidence metadata;
real frames, clips, capture, and inference are intentionally not represented
as complete.

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
