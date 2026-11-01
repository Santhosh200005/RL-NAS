"""Multi-objective reward: accuracy minus bounded, normalised efficiency penalties."""

def compute_reward(val_acc, params, flops, latency_ms, cfg: dict) -> float:
    def pen(x, ref, lam):
        return lam * min(x / ref, 3.0) if lam else 0.0
    return (val_acc
            - pen(params, cfg.get("ref_params", 1e6), cfg.get("lam_params", 0.0))
            - pen(flops, cfg.get("ref_flops", 5e7), cfg.get("lam_flops", 0.0))
            - pen(latency_ms, cfg.get("ref_latency_ms", 0.1), cfg.get("lam_latency", 0.0)))

ACC_ONLY = {"lam_params": 0.0, "lam_flops": 0.0, "lam_latency": 0.0}
