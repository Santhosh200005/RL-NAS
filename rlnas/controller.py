"""LSTM policy controller (REINFORCE)."""
import torch
import torch.nn as nn
from torch.distributions import Categorical
from .search_space import CARDS, N_TOKENS, DEPTHS, layer_of_token, decode


class Controller(nn.Module):
    def __init__(self, hidden: int = 64, emb: int = 32):
        super().__init__()
        self.hidden = hidden
        self.embs = nn.ModuleList(nn.Embedding(c, emb) for c in CARDS)
        self.heads = nn.ModuleList(nn.Linear(hidden, c) for c in CARDS)
        self.start = nn.Parameter(torch.zeros(emb))
        self.lstm = nn.LSTMCell(emb, hidden)

    def sample(self, greedy: bool = False):
        h = torch.zeros(1, self.hidden)
        c = torch.zeros(1, self.hidden)
        x = self.start.unsqueeze(0)
        toks, logp, ent, depth = [], torch.zeros(()), torch.zeros(()), None
        for t in range(N_TOKENS):
            h, c = self.lstm(x, (h, c))
            dist = Categorical(logits=self.heads[t](h))
            a = dist.probs.argmax(-1) if greedy else dist.sample()
            if t == 0:
                depth = DEPTHS[a.item()]
            l = layer_of_token(t)
            if l < depth:                      # global tokens (l=-1) and active layers only
                logp = logp + dist.log_prob(a).squeeze()
                ent = ent + dist.entropy().squeeze()
            toks.append(int(a.item()))
            x = self.embs[t](a)
        return toks, logp, ent

    def sample_arch(self, greedy=False):
        toks, logp, ent = self.sample(greedy)
        return decode(toks), logp, ent
