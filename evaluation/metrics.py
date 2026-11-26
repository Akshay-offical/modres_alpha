"""
metrics.py - accuracy, NLL, Brier, ECE, temperature scaling, reliability diagram data.
All functions take numpy probabilities p (n, 4) and integer labels y (n,).
"""
import numpy as np
import torch


def softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def accuracy(p, y):
    return float((p.argmax(1) == y).mean())


def nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean())


def brier(p, y):
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((p - onehot) ** 2).sum(1).mean())


def reliability_bins(p, y, n_bins: int = 10):
    """Top-label confidence vs accuracy in equal-width bins. Returns list of (count, mean_conf, acc)."""
    conf = p.max(1)
    hit = (p.argmax(1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum():
            out.append((int(m.sum()), float(conf[m].mean()), float(hit[m].mean())))
    return out


def ece(p, y, n_bins: int = 10):
    n = len(y)
    return float(sum(c / n * abs(a - cf) for c, cf, a in reliability_bins(p, y, n_bins)))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """One scalar T > 0 minimising validation NLL (Guo et al. 2017)."""
    z = torch.tensor(logits, dtype=torch.float64)
    t = torch.tensor(y, dtype=torch.long)
    log_T = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.5, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z / log_T.exp(), t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_T.exp().item())


def summarise(p, y):
    return dict(n=int(len(y)), acc=accuracy(p, y), nll=nll(p, y), brier=brier(p, y), ece=ece(p, y))
