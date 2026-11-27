"""splits.py - which plans are used, and the train / val / test split by plan.

Kept free of torch so the simulator's table export (simulator/tables.py) can use exactly the same
plans and split as the GNN (gnn/train.py).
"""
import random


def sample_ids(ids, limit, split_seed):
    """All ids, or a random sample of `limit` of them.

    ModRes is sorted by flat size (ids 0-999 are mostly 1-bedroom flats), so the first N plans are not
    representative. The sample depends only on split_seed.
    """
    if limit and limit < len(ids):
        return sorted(random.Random(split_seed).sample(ids, limit))
    return ids


def split_by_plan(plans, seed):
    """Shuffle the plans (seeded) and cut 70 / 15 / 15 into train / val / test."""
    rng = random.Random(seed)
    idx = list(range(len(plans)))
    rng.shuffle(idx)
    n = len(idx)
    tr, va = int(0.70 * n), int(0.85 * n)
    pick = lambda ix: [plans[i] for i in ix]
    return pick(idx[:tr]), pick(idx[tr:va]), pick(idx[va:])
