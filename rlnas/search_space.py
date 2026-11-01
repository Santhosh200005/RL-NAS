"""Search space: encoding, decoding, validation and analytic complexity.

Token layout (18 decisions): [depth, activation, dropout] + 5 x [filters, kernel, pool].
Layers beyond `depth` are inactive (ignored when building and when computing log-probs).
Illegal pooling (more than MAX_POOLS downsamplings) is repaired deterministically by decode().
"""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
import numpy as np

MAX_LAYERS = 5
DEPTHS = [2, 3, 4, 5]
ACTS = ["relu", "gelu", "silu"]
DROPS = [0.0, 0.1, 0.3]
FILTERS = [16, 32, 64, 128]
KERNELS = [3, 5, 7]
POOLS = ["none", "max", "avg"]
MAX_POOLS = 3          # 32x32 input -> at least 4x4 before the head
NUM_CLASSES = 10

CHOICES = [DEPTHS, ACTS, DROPS] + [FILTERS, KERNELS, POOLS] * MAX_LAYERS
CARDS = [len(c) for c in CHOICES]
N_TOKENS = len(CHOICES)


def layer_of_token(t: int) -> int:
    """-1 for global tokens, else layer index."""
    return -1 if t < 3 else (t - 3) // 3


@dataclass(frozen=True)
class Architecture:
    act: str
    dropout: float
    layers: tuple  # tuple of (filters, kernel, pool)

    @property
    def depth(self) -> int:
        return len(self.layers)

    def to_dict(self) -> dict:
        return {"act": self.act, "dropout": self.dropout,
                "layers": [list(l) for l in self.layers]}

    @staticmethod
    def from_dict(d: dict) -> "Architecture":
        a = Architecture(d["act"], float(d["dropout"]),
                         tuple((int(f), int(k), str(p)) for f, k, p in d["layers"]))
        validate(a)
        return a

    def key(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]

    def short(self) -> str:
        body = "-".join(f"{f}k{k}{'' if p == 'none' else p[0]}" for f, k, p in self.layers)
        return f"{body}|{self.act}|d{self.dropout}"


def validate(a: Architecture) -> bool:
    if not (DEPTHS[0] <= a.depth <= DEPTHS[-1]):
        raise ValueError(f"illegal depth {a.depth}")
    if a.act not in ACTS or a.dropout not in DROPS:
        raise ValueError("illegal activation/dropout")
    n_pool = 0
    for f, k, p in a.layers:
        if f not in FILTERS or k not in KERNELS or p not in POOLS:
            raise ValueError(f"illegal layer {(f, k, p)}")
        n_pool += p != "none"
    if n_pool > MAX_POOLS:
        raise ValueError("too many pooling layers")
    return True


def decode(tokens) -> Architecture:
    """Token indices -> legal Architecture (excess pooling repaired to 'none')."""
    assert len(tokens) == N_TOKENS
    depth = DEPTHS[tokens[0]]
    act, drop = ACTS[tokens[1]], DROPS[tokens[2]]
    layers, pools = [], 0
    for l in range(depth):
        f, k, p = (CHOICES[3 + 3 * l + j][tokens[3 + 3 * l + j]] for j in range(3))
        if p != "none":
            if pools >= MAX_POOLS:
                p = "none"
            else:
                pools += 1
        layers.append((f, k, p))
    a = Architecture(act, drop, tuple(layers))
    validate(a)
    return a


def encode_tokens(a: Architecture) -> list:
    t = [DEPTHS.index(a.depth), ACTS.index(a.act), DROPS.index(a.dropout)]
    for l in range(MAX_LAYERS):
        if l < a.depth:
            f, k, p = a.layers[l]
            t += [FILTERS.index(f), KERNELS.index(k), POOLS.index(p)]
        else:
            t += [0, 0, 0]
    return t


def features(a: Architecture) -> np.ndarray:
    """Normalised vector for Bayesian optimisation (inactive layers are zeroed)."""
    t = encode_tokens(a)
    v = np.array([ti / max(c - 1, 1) for ti, c in zip(t, CARDS)], dtype=float)
    for l in range(a.depth, MAX_LAYERS):
        v[3 + 3 * l: 6 + 3 * l] = 0.0
    return v


def sample_random(rng: np.random.Generator) -> Architecture:
    return decode([int(rng.integers(c)) for c in CARDS])


def complexity(a: Architecture, input_size: int = 32, in_ch: int = 3):
    """Analytic (params, MACs). Matches models.CandidateCNN (conv no bias + BN + linear head)."""
    params = macs = 0
    c, s = in_ch, input_size
    for f, k, p in a.layers:
        params += c * f * k * k + 2 * f          # conv + BN affine
        macs += s * s * c * f * k * k            # 'same' padding
        if p != "none":
            s //= 2
        c = f
    params += c * NUM_CLASSES + NUM_CLASSES      # head
    macs += c * NUM_CLASSES
    return int(params), int(macs)


REFERENCE = Architecture("relu", 0.1, ((32, 3, "max"), (64, 3, "max"), (128, 3, "max")))
