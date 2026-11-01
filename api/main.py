"""FastAPI service. uvicorn api.main:app --host 0.0.0.0 --port 8000
Env: RLNAS_API_KEY (optional auth), RLNAS_MAX_BUDGET (default 200), RLNAS_EVALUATOR (surrogate|torch)."""
import os, threading, uuid
import yaml
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel
from rlnas.search import run_search
from rlnas.search_space import Architecture
from rlnas.reward import compute_reward
from rlnas.analysis import pareto_front
import pandas as pd

app = FastAPI(title="RL-NAS API")
EXPS, LOCK = {}, threading.Lock()
MAX_BUDGET = int(os.getenv("RLNAS_MAX_BUDGET", "200"))
_EV = {}


def auth(x_api_key: str = Header(default=None)):
    key = os.getenv("RLNAS_API_KEY")
    if key and x_api_key != key:
        raise HTTPException(401, "invalid API key")


def get_eval(cfg, kind):
    if kind not in _EV:
        from experiments.run import make_evaluator
        _EV[kind] = make_evaluator(cfg, kind)
    return _EV[kind]


class ExpCreate(BaseModel):
    method: str = "rl"
    budget: int = 20
    seed: int = 0
    config: str = "configs/smoke.yaml"


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/experiments", dependencies=[Depends(auth)])
def create(req: ExpCreate):
    if req.budget > MAX_BUDGET or req.budget < 1:
        raise HTTPException(400, f"budget must be 1..{MAX_BUDGET}")
    if req.method not in ("random", "bayes", "rl", "rl_acc"):
        raise HTTPException(400, "unknown method")
    eid = uuid.uuid4().hex[:8]
    EXPS[eid] = {"id": eid, "req": req.model_dump(), "status": "created", "history": [], "error": None}
    return EXPS[eid]["req"] | {"id": eid, "status": "created"}


@app.post("/experiments/{eid}/start", dependencies=[Depends(auth)])
def start(eid: str):
    e = EXPS.get(eid) or (_ for _ in ()).throw(HTTPException(404, "not found"))
    if e["status"] == "running":
        raise HTTPException(409, "already running")
    cfg = yaml.safe_load(open(e["req"]["config"]))
    kind = os.getenv("RLNAS_EVALUATOR", "surrogate")

    def job():
        try:
            e["status"] = "running"
            run_search(e["req"]["method"], get_eval(cfg, kind), cfg, e["req"]["seed"],
                       e["req"]["budget"], on_step=e["history"].append)
            e["status"] = "done"
        except Exception as ex:  # surfaced via status endpoint
            e["status"], e["error"] = "failed", str(ex)
    threading.Thread(target=job, daemon=True).start()
    return {"id": eid, "status": "running"}


@app.get("/experiments/{eid}", dependencies=[Depends(auth)])
def status(eid: str):
    e = EXPS.get(eid)
    if not e:
        raise HTTPException(404, "not found")
    return {"id": eid, "status": e["status"], "episodes_done": len(e["history"]),
            "budget": e["req"]["budget"], "error": e["error"]}


@app.get("/experiments/{eid}/results", dependencies=[Depends(auth)])
def results(eid: str):
    e = EXPS.get(eid)
    if not e:
        raise HTTPException(404, "not found")
    h = list(e["history"])
    if not h:
        return {"best": None, "pareto": [], "history": []}
    df = pd.DataFrame(h)
    return {"best": max(h, key=lambda r: r["reward"]),
            "pareto": pareto_front(df)[["key", "val_acc", "params", "flops"]].to_dict("records"),
            "history": h}


@app.get("/architectures/{key}", dependencies=[Depends(auth)])
def architecture(key: str):
    for e in EXPS.values():
        for r in e["history"]:
            if r["key"] == key:
                return r
    raise HTTPException(404, "not found")


class EvalReq(BaseModel):
    arch: dict
    config: str = "configs/smoke.yaml"


@app.post("/evaluate", dependencies=[Depends(auth)])
def evaluate(req: EvalReq):
    try:
        arch = Architecture.from_dict(req.arch)
    except Exception as ex:
        raise HTTPException(422, f"illegal architecture: {ex}")
    cfg = yaml.safe_load(open(req.config))
    m = get_eval(cfg, os.getenv("RLNAS_EVALUATOR", "surrogate")).evaluate(arch)
    m["reward"] = compute_reward(m["val_acc"], m["params"], m["flops"], m.get("latency_ms", 0.0), cfg["reward"])
    return {"key": arch.key(), **m}
