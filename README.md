# HackHive local event desk

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
