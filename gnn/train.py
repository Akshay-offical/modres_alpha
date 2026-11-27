"""
train.py - build views, train the models, evaluate every model on the SAME frozen test views.

Run from the repository root.

Synthetic smoke test (no dataset needed; the numbers mean nothing scientifically):
    python -m gnn.train --source synthetic --plans 800 --models all --epochs 40

Real data (needs the dataset + shapely; plan ids from validation_out/benchmark_plan_ids.txt):
    python -m gnn.train --source modres --root "D:\\ModRes\\dataset" --ids validation_out\\benchmark_plan_ids.txt --models all

What happens
  1. plans are split BY PLAN into train / val / test (70/15/15, fixed seed)
  2. train plans: `--train-walks` random walks each, snapshots at 25/50/75%
     val / test plans: `--eval-walks` walks each (fixed seeds, identical for every model)
  3. the frequency priors are counted on the training views
  4. each neural model: cross-entropy on stub nodes only, Adam, early stopping on validation NLL,
     then temperature scaling fitted on the validation set
  5. a table per model and per observed fraction: accuracy, NLL, Brier, ECE, before/after temperature;
     results.csv also splits every row by flat size (bedrooms 1 / 2 / 3 / 4+)
"""
import argparse
import copy
import time
import warnings
from pathlib import Path

# Importing torch_geometric triggers this deprecation warning from PyTorch itself. It comes from
# PyG's own code, not ours, and does not affect results. Hide only this one message.
warnings.filterwarnings("ignore", message=r"`torch\.jit\.script` is deprecated", category=FutureWarning)

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from baselines.prior import ConditionalPrior, MarginalPrior
from data.splits import sample_ids, split_by_plan
from evaluation import metrics as M
from gnn.models import MODEL_NAMES, Net
from simulator.views import make_views, plan_from_loader, synthetic_plans, view_to_data

FRACTIONS = (0.25, 0.5, 0.75)
FRACTION_COL = 11      # feature column holding the observed fraction (see the feature list in views.py)


def load_plans(a):
    if a.source == "synthetic":
        return synthetic_plans(a.plans, seed=a.seed)
    from data import modres_loader as ml
    if not a.ids:
        raise SystemExit("--ids is required with --source modres (e.g. validation_out\\benchmark_plan_ids.txt)")
    root = a.root or ml.DEFAULT_ROOT
    ids = [int(s) for s in Path(a.ids).read_text().split()]
    ids = sample_ids(ids, a.limit, a.split_seed)   # random sample for --limit (see data/splits.py)
    plans = []
    t0 = time.time()
    for k, i in enumerate(ids, 1):
        plans.append(plan_from_loader(ml.load_plan(i, root)))
        if k % 1000 == 0:
            print(f"  loaded {k}/{len(ids)} plans ({time.time() - t0:.0f}s)")
    return plans


def build(plans, walks, base_seed):
    data = []
    for w in range(walks):
        for p in plans:
            n_bed = sum(1 for t in p.types if t == 1)
            for v in make_views(p, FRACTIONS, seed=base_seed + w):
                d = view_to_data(v)
                d.n_bed = n_bed          # flat size, only used to split the results table; never a model input
                data.append(d)
    return data


def logits_on_stubs(model, loader, device):
    model.eval()
    L, Y, Fr = [], [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            out = model(b)
            m = b.stub_mask
            L.append(out[m].cpu().numpy())
            Y.append(b.y[m].cpu().numpy())
            Fr.append(b.fraction[b.batch][m].cpu().numpy())      # per-node fraction (graph attribute expanded)
    return np.concatenate(L), np.concatenate(Y), np.concatenate(Fr)


def train_one(name, train, val, a, device):
    torch.manual_seed(a.seed)
    model = Net(name, hidden=a.hidden, layers=a.layers, dropout=a.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-5)
    tl = DataLoader(train, batch_size=a.batch, shuffle=True)
    vl = DataLoader(val, batch_size=256)
    best, best_state, bad = 1e9, None, 0
    for ep in range(1, a.epochs + 1):
        model.train()
        for b in tl:
            b = b.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(b)[b.stub_mask], b.y[b.stub_mask])
            loss.backward()
            opt.step()
        z, y, _ = logits_on_stubs(model, vl, device)
        v = M.nll(M.softmax(z), y)
        if v < best - 1e-4:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    return model, ep, best


BEDROOM_GROUPS = (("all", 0, 99), ("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4+", 4, 99))


def rows_for(name, p, y, fr, bed, tag):
    """One row per (observed fraction, flat size); 'all' rows pool everything."""
    rows = []
    for f in list(FRACTIONS) + ["all"]:
        mf = np.ones(len(y), bool) if f == "all" else np.isclose(fr, f)
        for group, lo, hi in BEDROOM_GROUPS:
            m = mf & (bed >= lo) & (bed <= hi)
            if m.sum() == 0:
                continue
            rows.append(dict(model=name, calib=tag, fraction=f, bedrooms=group, **M.summarise(p[m], y[m])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "modres"], default="synthetic")
    ap.add_argument("--plans", type=int, default=800, help="synthetic only")
    ap.add_argument("--root", default=None)
    ap.add_argument("--ids", default=None, help="file with plan ids (benchmark_plan_ids.txt)")
    ap.add_argument("--limit", type=int, default=0, help="use a random sample of N plans for quick tests (the sample depends on --split-seed)")
    ap.add_argument("--models", default="all", help="comma list from " + ",".join(MODEL_NAMES) + " or 'all'")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--train-walks", type=int, default=5)
    ap.add_argument("--eval-walks", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0, help="model initialisation / training order")
    ap.add_argument("--split-seed", type=int, default=0,
                    help="train/val/test split; keep it fixed when repeating runs with different --seed")
    ap.add_argument("--no-fraction-feature", action="store_true",
                    help="set feature column 11 (observed fraction) to 0: a real robot does not know it, "
                         "and with the number of visited rooms it reveals the total room count")
    ap.add_argument("--out", default="results_gnn")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    plans = load_plans(a)
    tr_p, va_p, te_p = split_by_plan(plans, a.split_seed)
    assert not ({p.plan_id for p in tr_p} & {p.plan_id for p in te_p}), "plan in train and test"
    train = build(tr_p, a.train_walks, 10_000)
    val = build(va_p, a.eval_walks, 20_000)
    test = build(te_p, a.eval_walks, 30_000)
    if a.no_fraction_feature:
        for d in train + val + test:
            d.x[:, FRACTION_COL] = 0.0
    print(f"plans train/val/test = {len(tr_p)}/{len(va_p)}/{len(te_p)}   "
          f"views = {len(train)}/{len(val)}/{len(test)}   device = {device}")

    all_rows = []
    # test labels / fractions in the same order for every model
    y_test = np.concatenate([d.y[d.stub_mask].numpy() for d in test])
    fr_test = np.concatenate([np.full(int(d.stub_mask.sum()), d.fraction) for d in test])
    bed_test = np.concatenate([np.full(int(d.stub_mask.sum()), d.n_bed) for d in test])
    print(f"test stub rooms = {len(y_test)}   by bedrooms: "
          + "  ".join(f"{g}: {int(((bed_test >= lo) & (bed_test <= hi)).sum())}" for g, lo, hi in BEDROOM_GROUPS[1:]))

    for pname, prior in (("prior_marginal", MarginalPrior().fit(train)),
                         ("prior_conditional", ConditionalPrior().fit(train))):
        p = np.concatenate([prior.predict(d) for d in test])
        all_rows += rows_for(pname, p, y_test, fr_test, bed_test, "none")

    names = MODEL_NAMES if a.models == "all" else a.models.split(",")
    vl = DataLoader(val, batch_size=256)
    tel = DataLoader(test, batch_size=256, shuffle=False)
    for name in names:
        t0 = time.time()
        model, ep, best = train_one(name, train, val, a, device)
        zv, yv, _ = logits_on_stubs(model, vl, device)
        zt, yt, frt = logits_on_stubs(model, tel, device)
        assert np.array_equal(yt, y_test) and np.allclose(frt, fr_test), "test order differs"
        T = M.fit_temperature(zv, yv)
        all_rows += rows_for(name, M.softmax(zt), yt, frt, bed_test, "raw")
        all_rows += rows_for(name, M.softmax(zt, T), yt, frt, bed_test, f"T={T:.2f}")
        print(f"  {name:12s} epochs {ep:3d}  best val NLL {best:.3f}  T {T:.2f}  ({time.time() - t0:.0f}s)")

    import pandas as pd
    df = pd.DataFrame(all_rows)
    df.insert(0, "seed", a.seed)
    df.insert(1, "split_seed", a.split_seed)
    df.insert(2, "fraction_feature", not a.no_fraction_feature)
    df.to_csv(out / "results.csv", index=False)
    pd.set_option("display.width", 200)
    show = df[df.bedrooms == "all"].drop(columns="bedrooms")
    print("\n", show.round(3).to_string(index=False))

    # accuracy by flat size (temperature scaling does not change accuracy, so raw / none rows are enough)
    acc = df[df.calib.isin(["none", "raw"])].pivot_table(index=["model", "fraction"], columns="bedrooms",
                                                        values="acc", sort=False)
    acc = acc[[g for g, _, _ in BEDROOM_GROUPS if g in acc.columns]]
    print("\naccuracy by number of bedrooms in the flat:\n", acc.round(3).to_string())


if __name__ == "__main__":
    main()
