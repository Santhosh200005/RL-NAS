"""Short proxy training + evaluation of a candidate architecture."""
import math, time
import torch
import torch.nn.functional as F
from .models import CandidateCNN
from .search_space import complexity


@torch.no_grad()
def accuracy(model, x, y, bs=1000):
    model.eval()
    correct = 0
    for i in range(0, len(x), bs):
        correct += (model(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
    return correct / len(x)


@torch.no_grad()
def latency_ms(model, x, device, reps=10):
    """ms per sample at batch 64 (measured on the current device)."""
    model.eval()
    xb = x[:64]
    for _ in range(3):
        model(xb)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        model(xb)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps / len(xb) * 1000


def train_eval(arch, data, tcfg: dict, seed: int, device, test: bool = False):
    t0 = time.time()
    torch.manual_seed(seed)
    epochs, bs, lr = tcfg.get("epochs", 3), tcfg.get("batch_size", 128), tcfg.get("lr", 3e-3)
    params_a, macs = complexity(arch)
    try:
        model = CandidateCNN(arch).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        xtr, ytr = data["xtr"], data["ytr"]
        n = len(xtr)
        steps = epochs * math.ceil(n / bs)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
        amp = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        diverged = False
        for _ in range(epochs):
            model.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                xb, yb = xtr[idx], ytr[idx]
                flip = torch.rand(len(idx), device=device) < 0.5
                xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    loss = F.cross_entropy(model(xb), yb)
                if not torch.isfinite(loss):
                    diverged = True
                    break
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update(); sched.step()
            if diverged:
                break
        val = 0.1 if diverged else accuracy(model, data["xva"], data["yva"])
        out = {"val_acc": val, "params": n_params, "flops": macs,
               "latency_ms": latency_ms(model, data["xva"], device), "failed": diverged}
        if test:
            out["test_acc"] = accuracy(model, data["xte"], data["yte"])
    except RuntimeError as e:  # e.g. CUDA OOM -> treat as chance-level, keep the search alive
        if device.type == "cuda":
            torch.cuda.empty_cache()
        out = {"val_acc": 0.1, "params": params_a, "flops": macs, "latency_ms": 0.0,
               "failed": True, "error": str(e)[:120]}
    out["train_time_s"] = time.time() - t0
    return out
