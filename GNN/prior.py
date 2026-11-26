"""
prior.py - the baselines every learned model must beat. Both read exactly the same Data object as the GNN.

    MarginalPrior     P(type of a stub) from training counts only (no context)
    ConditionalPrior  P(type of a stub | type of the visited room whose doorway leads to it,
                                          whether a kitchen is already known)
                      i.e. the Markov-style table: the unknown room depends only on its neighbour
                      (plus the 'exactly one kitchen' fact).  Counts are smoothed towards the marginal.
"""
import numpy as np

from views import N_TYPES


def _stub_contexts(d):
    """For each stub node (in node order): (type of lowest-index visited neighbour via a doorway or None, kitchen_known)."""
    x = d.x.numpy()
    ei = d.edge_index.numpy()
    et = d.edge_type.numpy()
    stub_idx = np.where(d.stub_mask.numpy())[0]
    kitchen_known = int((x[:, 3] == 1).any())
    out = []
    for s in stub_idx:
        nbr_types = [int(x[a, :4].argmax()) for a, b, t in zip(ei[0], ei[1], et)
                     if b == s and t == 0 and x[a, 5] == 1]
        out.append((min(nbr_types) if nbr_types else None, kitchen_known))
    return stub_idx, out


class MarginalPrior:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.counts = np.zeros(N_TYPES)

    def fit(self, dataset):
        for d in dataset:
            y = d.y[d.stub_mask].numpy()
            self.counts += np.bincount(y, minlength=N_TYPES)
        return self

    def predict(self, d):
        p = (self.counts + self.alpha) / (self.counts.sum() + self.alpha * N_TYPES)
        return np.tile(p, (int(d.stub_mask.sum()), 1))


class ConditionalPrior:
    def __init__(self, alpha: float = 2.0):
        self.alpha = alpha
        self.marg = MarginalPrior()
        self.table = {}

    def fit(self, dataset):
        self.marg.fit(dataset)
        for d in dataset:
            _, ctx = _stub_contexts(d)
            y = d.y[d.stub_mask].numpy()
            for c, t in zip(ctx, y):
                self.table.setdefault(c, np.zeros(N_TYPES))[t] += 1
        m = self.marg.counts + self.marg.alpha
        self.marginal_p = m / m.sum()
        return self

    def predict(self, d):
        _, ctx = _stub_contexts(d)
        rows = []
        for c in ctx:
            cnt = self.table.get(c, np.zeros(N_TYPES))
            rows.append((cnt + self.alpha * self.marginal_p) / (cnt.sum() + self.alpha))
        return np.array(rows)
