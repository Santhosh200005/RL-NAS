"""Learning curves, Pareto fronts and summary tables from JSONL search logs."""
import glob, json, os
import numpy as np
import pandas as pd
from .reward import compute_reward


def load_histories(d):
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        with open(p) as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return pd.DataFrame(rows)


def add_common_reward(df, rcfg):
    df = df.copy()
    df["common_reward"] = [compute_reward(a, p, f, l, rcfg) for a, p, f, l in
                           zip(df.val_acc, df.params, df.flops, df.latency_ms)]
    return df


def curve_table(df, col):
    df = df.sort_values(["method", "seed", "step"]).copy()
    df["best"] = df.groupby(["method", "seed"])[col].cummax()
    g = df.groupby(["method", "step"]).best
    return pd.DataFrame({"mean": g.mean(), "std": g.std().fillna(0.0)}).reset_index()


def pareto_front(df, acc="val_acc", cost="params"):
    d = df.sort_values([cost, acc], ascending=[True, False])
    front, best = [], -1
    for i, r in d.iterrows():
        if r[acc] > best:
            front.append(i); best = r[acc]
    return d.loc[front]


def summary(df):
    rows = []
    for (m, s), g in df[df.method != "reference"].groupby(["method", "seed"]):
        b = g.loc[g.common_reward.idxmax()]
        rows.append({"method": m, "seed": s, "best_val_acc": g.val_acc.max(),
                     "best_reward": b.common_reward, "acc@best_reward": b.val_acc,
                     "params@best_reward": b.params, "MFLOPs@best_reward": b.flops / 1e6,
                     "unique_archs": g.key.nunique(), "search_cost_s": g.train_time_s.sum()})
    per = pd.DataFrame(rows)
    agg = per.groupby("method").agg(["mean", "std"]).drop(columns="seed", level=0)
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return per, agg.reset_index()


def make_report(results_dir, rcfg, out_dir=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = out_dir or results_dir
    df = add_common_reward(load_histories(results_dir), rcfg)
    per, agg = summary(df)
    per.to_csv(os.path.join(out_dir, "per_seed_summary.csv"), index=False)
    agg.to_csv(os.path.join(out_dir, "method_summary.csv"), index=False)
    for col, name in [("val_acc", "best_val_acc"), ("common_reward", "best_reward")]:
        ct = curve_table(df[df.method != "reference"], col)
        plt.figure(figsize=(6, 4))
        for m, g in ct.groupby("method"):
            plt.plot(g.step, g["mean"], label=m)
            plt.fill_between(g.step, g["mean"] - g["std"], g["mean"] + g["std"], alpha=0.15)
        plt.xlabel("episodes"); plt.ylabel(f"{name} (running max)"); plt.legend(); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"curve_{name}.png"), dpi=150); plt.close()
    plt.figure(figsize=(6, 4.5))
    for m, g in df.groupby("method"):
        plt.scatter(g.params, g.val_acc, s=10, alpha=.4, label=m)
    fr = pareto_front(df)
    plt.plot(fr.params, fr.val_acc, "k--", lw=1, label="global Pareto front")
    plt.xscale("log"); plt.xlabel("parameters"); plt.ylabel("val accuracy"); plt.legend(fontsize=7)
    plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(os.path.join(out_dir, "pareto.png"), dpi=150); plt.close()
    return per, agg
