# RL-NAS — Reinforcement Learning–Driven Neural Architecture Search

LSTM controller (REINFORCE) proposes small CNNs, proxy-trains them on a reduced CIFAR-10 subset, and is rewarded with
`val_acc − λp·params − λf·FLOPs − λl·latency`. Compared against random search, GP Bayesian optimisation (EI),
a fixed reference CNN, and an accuracy-only RL ablation.

## Quick start
```bash
pip install -r requirements.txt
python -m unittest discover -s tests            # or: pytest -q
# 1) pipeline check on synthetic data (CPU, ~1 min)
python -m experiments.run --config configs/smoke.yaml --methods reference random bayes rl rl_acc
# 2) real experiment (GPU strongly recommended)
python -m experiments.run --config configs/default.yaml --methods reference random bayes rl rl_acc --seeds 0 1 2 3 4
python -m experiments.analyze results/default          # curves, Pareto plot, summary tables
python -m experiments.final_eval results/default       # long retrain of each best arch -> TEST accuracy (once)
# API
uvicorn api.main:app --port 8000   # RLNAS_EVALUATOR=torch for real jobs, RLNAS_API_KEY to require auth
```

## Design
- **Search space** (`rlnas/search_space.py`): depth 2–5; per-layer filters {16,32,64,128}, kernel {3,5,7}, pool {none,max,avg};
  global activation {relu,gelu,silu}, dropout {0,.1,.3}. 18 tokens; inactive layers masked from the policy gradient;
  >3 pools repaired to `none`. ~10^9 raw combinations.
- **Controller** (`controller.py`): LSTMCell + per-token heads, REINFORCE, EMA baseline, entropy bonus, batch of 4 per update.
- **Fair comparison**: all methods share the same split, budget, evaluator and cache; curves/summary tables recompute a
  *common* reward from logged metrics so `rl_acc` is judged on the same objective. Search cost uses recorded training time
  (cache hits keep the original cost).
- **Honest protocol**: search uses the 5k validation split only; test accuracy appears only in `final_eval.py`.
- **Latency** is measured on whatever device runs the search (batch 64, ms/sample); keep `lam_latency: 0` unless the
  search hardware equals the target hardware.

## Status / limitations
- Written and unit-tested in a sandbox *without PyTorch or network*. Verified there: search space, complexity counter,
  reward, random + Bayesian search, logging, analysis/plots. **Not yet executed**: torch model/trainer, LSTM controller,
  FastAPI app, Docker builds. `tests/test_core.py::TestTorch` checks these (incl. analytic-vs-real parameter count) — run it first.
- `SurrogateEvaluator` is an analytic fake for plumbing/API demos. Never report its numbers.
- Reward coefficients (λ) are untuned defaults; sweep them and report (ablation).
- Frontend (Next.js/Vercel) is included under `frontend/`; the API exposes everything it needs.

## Deployment
Vercel/Next.js frontend → HTTPS → FastAPI (`Dockerfile`) → GPU worker (`Dockerfile.worker`) → object storage for results.
Hardening already in API: optional `X-API-Key`, `RLNAS_MAX_BUDGET` job cap.

## Frontend
The repository now also includes `frontend/`, a Next.js dashboard for the existing FastAPI service. See `frontend/README.md` for setup. The dashboard does not alter the experiment protocol or substitute for the required GPU/PyTorch validation.
