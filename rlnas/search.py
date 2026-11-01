"""Search algorithms with a common budget/logging interface: random, bayes, rl, reference."""
import warnings; warnings.filterwarnings("ignore", category=Warning, module="sklearn")
import json, time
import numpy as np
from scipy.stats import norm
from .search_space import sample_random, features, REFERENCE
from .reward import compute_reward, ACC_ONLY


def _log(hist, fh, step, method, seed, arch, m, reward, t0):
    best_r = max([h["reward"] for h in hist] + [reward])
    best_a = max([h["val_acc"] for h in hist] + [m["val_acc"]])
    rec = {"step": step, "method": method, "seed": seed, "key": arch.key(), "arch": arch.to_dict(),
           "val_acc": m["val_acc"], "params": m["params"], "flops": m["flops"],
           "latency_ms": m.get("latency_ms", 0.0), "train_time_s": m.get("train_time_s", 0.0),
           "reward": reward, "best_reward": best_r, "best_val_acc": best_a,
           "elapsed_s": time.time() - t0}
    hist.append(rec)
    if fh:
        fh.write(json.dumps(rec) + "\n"); fh.flush()
    return rec


def run_search(method, evaluator, cfg, seed, budget=None, out_path=None, on_step=None):
    """method in {random, bayes, rl, rl_acc, reference}. rl_acc = accuracy-only reward ablation."""
    budget = budget or cfg["search"]["budget"]
    rcfg = ACC_ONLY if method == "rl_acc" else cfg["reward"]
    rng = np.random.default_rng(seed)
    hist, t0 = [], time.time()
    fh = open(out_path, "w") if out_path else None

    def step_eval(i, arch):
        m = evaluator.evaluate(arch)
        r = compute_reward(m["val_acc"], m["params"], m["flops"], m.get("latency_ms", 0.0), rcfg)
        rec = _log(hist, fh, i, method, seed, arch, m, r, t0)
        if on_step:
            on_step(rec)
        return r

    try:
        if method == "reference":
            step_eval(0, REFERENCE)
        elif method == "random":
            for i in range(budget):
                step_eval(i, sample_random(rng))
        elif method == "bayes":
            _bayes(step_eval, cfg, rng, seed, budget)
        elif method in ("rl", "rl_acc"):
            _rl(step_eval, cfg, seed, budget)
        else:
            raise ValueError(method)
    finally:
        if fh:
            fh.close()
    return hist


def _bayes(step_eval, cfg, rng, seed, budget):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    bo = cfg.get("bo", {})
    n_init, n_cand = bo.get("n_init", 10), bo.get("n_candidates", 1000)
    X, y, seen = [], [], set()
    for i in range(budget):
        arch = None
        if i >= n_init:
            try:
                gp = GaussianProcessRegressor(
                    ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(1e-3),
                    normalize_y=True, n_restarts_optimizer=2, random_state=seed)
                gp.fit(np.array(X), np.array(y))
                cands = [a for a in (sample_random(rng) for _ in range(n_cand)) if a.key() not in seen]
                mu, sd = gp.predict(np.array([features(a) for a in cands]), return_std=True)
                imp = mu - max(y) - 0.01
                z = imp / np.maximum(sd, 1e-9)
                ei = imp * norm.cdf(z) + sd * norm.pdf(z)       # expected improvement
                arch = cands[int(np.argmax(ei))]
            except Exception:
                arch = None                                      # fall back to random proposal
        if arch is None:
            arch = sample_random(rng)
        seen.add(arch.key())
        r = step_eval(i, arch)
        X.append(features(arch)); y.append(r)


def _rl(step_eval, cfg, seed, budget):
    import torch
    from .controller import Controller
    rl = cfg.get("rl", {})
    torch.manual_seed(seed)
    ctrl = Controller(hidden=rl.get("hidden", 64))
    opt = torch.optim.Adam(ctrl.parameters(), lr=rl.get("lr", 5e-3))
    B, beta, decay = rl.get("batch", 4), rl.get("entropy", 0.01), rl.get("baseline_decay", 0.9)
    baseline, i = None, 0
    while i < budget:
        logps, ents, rewards = [], [], []
        for _ in range(min(B, budget - i)):
            arch, lp, en = ctrl.sample_arch()
            rewards.append(step_eval(i, arch)); logps.append(lp); ents.append(en); i += 1
        r = torch.tensor(rewards, dtype=torch.float32)
        baseline = r.mean().item() if baseline is None else decay * baseline + (1 - decay) * r.mean().item()
        adv = r - baseline
        loss = -(adv * torch.stack(logps)).mean() - beta * torch.stack(ents).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ctrl.parameters(), 5.0)
        opt.step()
