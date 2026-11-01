"""python -m experiments.run --config configs/default.yaml --methods random bayes rl rl_acc --seeds 0 1 2"""
import argparse, os, yaml
from rlnas.search import run_search


def make_evaluator(cfg, kind):
    if kind == "surrogate":
        from rlnas.evaluators import SurrogateEvaluator
        return SurrogateEvaluator()
    from rlnas.evaluators import TorchEvaluator
    return TorchEvaluator(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--methods", nargs="+", default=["reference", "random", "bayes", "rl", "rl_acc"])
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--budget", type=int)
    ap.add_argument("--out", default=None)
    ap.add_argument("--evaluator", choices=["torch", "surrogate"], default="torch")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    out = a.out or os.path.join("results", cfg.get("name", "run"))
    os.makedirs(out, exist_ok=True)
    ev = make_evaluator(cfg, a.evaluator)
    for seed in a.seeds or cfg["search"]["seeds"]:
        for m in a.methods:
            path = os.path.join(out, f"{m}_seed{seed}.jsonl")
            if os.path.exists(path) and os.path.getsize(path) > 0:
                print("skip (exists)", path); continue
            h = run_search(m, ev, cfg, seed, a.budget, path)
            b = max(h, key=lambda r: r["reward"])
            print(f"{m:9s} seed={seed} best_acc={max(r['val_acc'] for r in h):.4f} "
                  f"best_reward={b['reward']:.4f} params={b['params']:,}", flush=True)


if __name__ == "__main__":
    main()
