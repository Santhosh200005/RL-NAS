"""python -m experiments.analyze results/default [--config configs/default.yaml]"""
import argparse, yaml
from rlnas.analysis import make_report

ap = argparse.ArgumentParser()
ap.add_argument("results_dir")
ap.add_argument("--config", default="configs/default.yaml")
a = ap.parse_args()
per, agg = make_report(a.results_dir, yaml.safe_load(open(a.config))["reward"])
print(agg.round(4).to_string(index=False))
