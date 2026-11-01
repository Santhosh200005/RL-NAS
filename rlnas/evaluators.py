"""Evaluators map Architecture -> metrics dict. Results are cached by architecture key."""
import hashlib
import numpy as np
from .search_space import Architecture, complexity


class _Cached:
    def __init__(self):
        self.cache = {}

    def evaluate(self, arch: Architecture) -> dict:
        k = arch.key()
        if k not in self.cache:
            self.cache[k] = self._eval(arch)
        return dict(self.cache[k])


class TorchEvaluator(_Cached):
    def __init__(self, cfg: dict, device=None, with_test=False, full_train=False):
        super().__init__()
        import torch
        from .data import load_data
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        self.data = load_data(cfg, self.device, with_test=with_test, full_train=full_train)

    def _eval(self, arch):
        from .trainer import train_eval
        seed = int(arch.key()[:8], 16) % (2 ** 31)   # deterministic per architecture
        return train_eval(arch, self.data, self.cfg["train"], seed, self.device)


class SurrogateEvaluator(_Cached):
    """Analytic stand-in for pipeline testing / API demos. NEVER use for reported results."""

    def _eval(self, arch):
        p, f = complexity(arch)
        h = int(hashlib.sha1(arch.key().encode()).hexdigest()[:8], 16)
        noise = np.random.default_rng(h).normal(0, 0.004)
        cap = np.log10(p)
        acc = 0.86 - 0.35 * np.exp(-(cap - 3.6) * 1.1)
        acc += 0.01 * np.mean([k == 3 for _, k, _ in arch.layers])
        acc += 0.02 * min(sum(pl != "none" for _, _, pl in arch.layers), 2)
        acc += {"relu": 0.0, "gelu": 0.004, "silu": 0.005}[arch.act] - 0.02 * (arch.dropout == 0.3)
        return {"val_acc": float(np.clip(acc + noise, 0.1, 0.95)), "params": p, "flops": f,
                "latency_ms": f / 4e9 * 1000, "train_time_s": 1e-3 * f / 1e6, "failed": False}
