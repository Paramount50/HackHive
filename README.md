# HackHive local event desk

The first runnable slice is a CPU-only deterministic fixture path. It deliberately
does not claim real model inference: fixture responses are marked in API responses,
status, and query coverage. The inference adapter can be replaced by the local
perception worker in the next milestone without changing event/query storage.

## Run locally

```powershell
C:\Users\Ayushmaan\AppData\Local\Programs\Python\Python313\python.exe -m pytest -q
C:\Users\Ayushmaan\AppData\Local\Programs\Python\Python313\python.exe -m uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000, click **Run deterministic fixture**, then inspect,
query, and review the event. Runtime SQLite data is written under `data/` and
must not be committed.

The API includes `/api/status`, `/api/events`, `/api/events/{id}`,
`/api/events/{id}/reviews`, `/api/query`, and `/api/fixtures/run`.
