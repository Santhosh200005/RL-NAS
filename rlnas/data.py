"""Dataset loading. All data is preloaded to the target device as tensors (fast proxy training)."""
import numpy as np
import torch

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def _to_tensor(x_uint8, device):
    x = torch.from_numpy(np.ascontiguousarray(x_uint8)).float().div(255).permute(0, 3, 1, 2)
    m = torch.tensor(MEAN).view(1, 3, 1, 1)
    s = torch.tensor(STD).view(1, 3, 1, 1)
    return ((x - m) / s).to(device)


def load_data(cfg: dict, device, with_test: bool = False, full_train: bool = False):
    name = cfg.get("dataset", "cifar10")
    n_val = cfg.get("n_val", 5000)
    n_train = cfg.get("n_train", 10000)
    g = np.random.default_rng(1234)  # fixed split shared by every method/seed
    if name == "synthetic":
        protos = g.normal(size=(10, 3, 32, 32)).astype("float32")
        def make(n):
            y = g.integers(0, 10, n)
            x = protos[y] + 1.5 * g.normal(size=(n, 3, 32, 32)).astype("float32")
            return torch.from_numpy(x).to(device), torch.from_numpy(y).long().to(device)
        xtr, ytr = make(n_train); xva, yva = make(n_val); xte, yte = make(n_val)
    else:
        from torchvision.datasets import CIFAR10
        root = cfg.get("data_dir", "./data_cache")
        tr = CIFAR10(root, train=True, download=True)
        perm = g.permutation(len(tr.data))
        val_idx, train_pool = perm[:5000], perm[5000:]       # 5k held-out validation
        val_idx = val_idx[:n_val]
        tr_idx = train_pool if full_train else train_pool[:n_train]
        y_all = np.array(tr.targets)
        xtr, ytr = _to_tensor(tr.data[tr_idx], device), torch.from_numpy(y_all[tr_idx]).long().to(device)
        xva, yva = _to_tensor(tr.data[val_idx], device), torch.from_numpy(y_all[val_idx]).long().to(device)
        xte = yte = None
        if with_test:
            te = CIFAR10(root, train=False, download=True)
            xte, yte = _to_tensor(te.data, device), torch.tensor(te.targets).long().to(device)
    out = {"xtr": xtr, "ytr": ytr, "xva": xva, "yva": yva}
    if with_test:
        out.update(xte=xte, yte=yte)
    return out
