# RL-NAS Next.js Dashboard

This is the missing frontend for the RL-NAS repository. It is intentionally self-contained and uses only Next.js, React and TypeScript; charts are rendered with inline SVG so no chart package is required.

## What it does

- Connects to the existing FastAPI service in `../api/main.py`.
- Checks `/health` and shows backend connectivity.
- Starts `random`, `bayes`, `rl`, and `rl_acc` experiments.
- Configures budget, seed, and the existing smoke/default YAML config.
- Polls experiment status while a job is running.
- Displays best validation accuracy, reward, parameters and FLOPs.
- Renders reward and validation-accuracy learning curves.
- Renders the logged Pareto frontier.
- Shows the complete search history and lets you inspect an architecture.
- Supports the existing optional FastAPI `X-API-Key` through `NEXT_PUBLIC_API_KEY`.

## Run locally

Start the FastAPI backend from the repository root first:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Then in this directory:

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

If the backend is not on localhost:8000, create `.env.local` from `.env.example` and set `RLNAS_API_URL`. If the FastAPI service uses `RLNAS_API_KEY`, set the same key in the Next.js server environment; it is deliberately not exposed to the browser.

## Important

The dashboard does not replace the experiment runner. It is a UI for the existing API. The actual GPU/PyTorch experiment still needs to be run and validated separately as described in the root README.
