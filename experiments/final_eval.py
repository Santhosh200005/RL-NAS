"""Retrain each method's best architecture longer on the full train split; report TEST accuracy once.
python -m experiments.final_eval results/default --config configs/default.yaml"""
import argparse, json, os, yaml
import torch
from rlnas.analysis import load_histories, add_common_reward
from rlnas.search_space import Architecture, REFERENCE
from rlnas.evaluators import TorchEvaluator
from rlnas.trainer import train_eval

ap = argparse.ArgumentParser()
ap.add_argument("results_dir"); ap.add_argument("--config", default="configs/default.yaml")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))
df = add_common_reward(load_histories(a.results_dir), cfg["reward"])
ev = TorchEvaluator(cfg, with_test=True, full_train=True)
tcfg = dict(cfg["train"]); tcfg["epochs"] = cfg.get("final", {}).get("epochs", 30)
out = {}
for m, g in df.groupby("method"):
    arch = REFERENCE if m == "reference" else Architecture.from_dict(g.loc[g.common_reward.idxmax()].arch)
    r = train_eval(arch, ev.data, tcfg, seed=0, device=ev.device, test=True)
    out[m] = {"arch": arch.to_dict(), **r}
    print(f"{m:9s} test_acc={r['test_acc']:.4f} params={r['params']:,} MFLOPs={r['flops']/1e6:.1f}")
json.dump(out, open(os.path.join(a.results_dir, "final_eval.json"), "w"), indent=2)
