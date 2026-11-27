r"""
rules.py - rules baselines for Task 1 and Task 2, read from the tables of simulator/tables.py.

    python -m baselines.rules                  # score on val
    python -m baselines.rules --eval test      # final evaluation only

The rules and metrics: docs/simulator.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TYPES = [0, 1, 2, 3]
NAMES = {0: "living", 1: "bedroom", 2: "bathroom", 3: "kitchen"}
KITCHEN = 3
DOORWAY = 0
SMOOTH = 1.0          # add-one smoothing so unseen combinations never get probability 0


# --------------------------------------------------------------------------- metrics (numpy only)
def f1_per_class(y, pred, labels):
    out = []
    for t in labels:
        tp = np.sum((pred == t) & (y == t))
        fp = np.sum((pred == t) & (y != t))
        fn = np.sum((pred != t) & (y == t))
        out.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return np.array(out)


def auroc(y, score):
    """Probability that a random real connection gets a higher score than a random wall (ties count half)."""
    y = np.asarray(y)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --------------------------------------------------------------------------- helpers
def cond_table(df: pd.DataFrame, keys: list[str], label: str = "label") -> pd.DataFrame:
    """P(label | keys) with add-one smoothing; columns = TYPES."""
    counts = df.groupby(keys + [label]).size().unstack(label, fill_value=0)
    counts = counts.reindex(columns=TYPES, fill_value=0) + SMOOTH
    return counts.div(counts.sum(axis=1), axis=0)


def t1_scores(name, y, probs, p):
    pred = np.array(TYPES)[probs.argmax(axis=1)]
    rows = []
    for pv in [None] + sorted(set(p)):
        m = np.ones(len(y), bool) if pv is None else (p == pv)
        per = f1_per_class(y[m], pred[m], TYPES)
        rows.append(dict(task="task1", method=name, p="all" if pv is None else pv, n=int(m.sum()),
                         macro_f1=float(per[1:].mean()), accuracy=float((y[m] == pred[m]).mean()),
                         **{f"f1_{NAMES[t]}": float(per[i]) for i, t in enumerate(TYPES)}))
    return rows


def t2_scores(name, y, prob, p):
    rows = []
    for pv in [None] + sorted(set(p)):
        m = np.ones(len(y), bool) if pv is None else (p == pv)
        pred = (prob[m] >= 0.5).astype(int)
        rows.append(dict(task="task2", method=name, p="all" if pv is None else pv, n=int(m.sum()),
                         auroc=auroc(y[m], prob[m]), f1=float(f1_per_class(y[m], pred, [1])[0]),
                         accuracy=float((y[m] == pred).mean())))
    return rows


# --------------------------------------------------------------------------- task 1
def task1(tables: Path, eval_split: str, out: Path):
    snaps = pd.read_parquet(tables / "snapshots.parquet")
    nd = pd.read_parquet(tables / "nodes.parquet")
    ed = pd.read_parquet(tables / "edges.parquet")
    lab = pd.read_parquet(tables / "labels_stub.parquet").merge(snaps[["snapshot_id", "p"]], on="snapshot_id")

    # each stub with the types of the known rooms (visited or seen) whose doorways lead to it
    state = nd.set_index(["snapshot_id", "rid"])
    ed = ed[ed.edge_type == DOORWAY]
    pairs = pd.concat([ed.rename(columns={"u": "stub", "v": "from"}), ed.rename(columns={"v": "stub", "u": "from"})])
    key_stub = pd.MultiIndex.from_arrays([pairs.snapshot_id, pairs.stub])
    key_from = pd.MultiIndex.from_arrays([pairs.snapshot_id, pairs["from"]])
    from_state = state.state.reindex(key_from).to_numpy()
    pairs = pairs[(state.state.reindex(key_stub).to_numpy() == "stub") & np.isin(from_state, ["visited", "seen"])]
    pairs["from_type"] = state.type.reindex(pd.MultiIndex.from_arrays([pairs.snapshot_id, pairs["from"]])).to_numpy()
    pairs = pairs.merge(lab[["snapshot_id", "rid", "label"]], left_on=["snapshot_id", "stub"],
                        right_on=["snapshot_id", "rid"])
    kitchen_known = nd[(nd.state != "stub") & (nd.type == KITCHEN)].snapshot_id.unique()

    t_markov = cond_table(pairs[pairs.split == "train"], ["from_type"])
    prior = lab[lab.split == "train"].label.value_counts(normalize=True).reindex(TYPES, fill_value=0).to_numpy()

    ev = lab[lab.split == eval_split].reset_index(drop=True)
    ev_pairs = pairs[pairs.split == eval_split]
    # multiply the Markov rows of all known rooms that open onto the stub (in log space)
    rows_p = t_markov.reindex(ev_pairs.from_type).to_numpy()
    rows_p[np.isnan(rows_p).any(axis=1)] = prior              # a neighbour type never seen in training
    logp = pd.DataFrame(np.log(rows_p), columns=TYPES)
    logp[["snapshot_id", "rid"]] = ev_pairs[["snapshot_id", "rid"]].to_numpy()
    agg = logp.groupby(["snapshot_id", "rid"])[TYPES].sum()
    pr = np.exp(agg.reindex(pd.MultiIndex.from_frame(ev[["snapshot_id", "rid"]])).to_numpy())
    pr[np.isnan(pr).any(axis=1)] = prior                      # safety net; every stub has a doorway
    pr = pr / pr.sum(axis=1, keepdims=True)

    y, p = ev.label.to_numpy(), ev.p.to_numpy()
    rows = t1_scores("T1-majority", y, np.tile(prior, (len(ev), 1)), p)
    rows += t1_scores("T1-markov", y, pr, p)
    pr = pr.copy()
    pr[ev.snapshot_id.isin(kitchen_known).to_numpy(), TYPES.index(KITCHEN)] = 0.0
    pr = pr / pr.sum(axis=1, keepdims=True)
    rows += t1_scores("T1-full", y, pr, p)

    tab = t_markov.rename(columns=NAMES).reset_index()
    tab["from_type"] = tab.from_type.map(NAMES)
    tab.round(4).to_csv(out / "task1_markov_table.csv", index=False)
    return rows


# --------------------------------------------------------------------------- task 2
def task2(tables: Path, eval_split: str):
    c = pd.read_parquet(tables / "task2_candidates.parquet")
    c["pair"] = [f"{min(a, b)}-{max(a, b)}" for a, b in zip(c.u_type, c.v_type)]

    tr, ev = c[c.split == "train"], c[c.split == eval_split].copy()
    g = tr.groupby("pair").label.agg(["sum", "count"])
    p_pair = (g["sum"] + SMOOTH) / (g["count"] + 2 * SMOOTH)
    prob = ev.pair.map(p_pair).fillna(tr.label.mean()).to_numpy()
    y, p = ev.label.to_numpy(), ev.p.to_numpy()
    rows = t2_scores("T2-pair", y, prob, p)

    # reachability: every edge of an unknown room is a candidate, so an unknown room with one
    # candidate has only that link, and it must be a real connection
    prob2 = prob.copy()
    ev["row"] = np.arange(len(ev))
    ends = pd.concat([ev.loc[ev.u_known == 0, ["snapshot_id", "u", "row"]].rename(columns={"u": "room"}),
                      ev.loc[ev.v_known == 0, ["snapshot_id", "v", "row"]].rename(columns={"v": "room"})])
    n_cand = ends.groupby(["snapshot_id", "room"]).row.transform("size")
    forced = ends.loc[n_cand.to_numpy() == 1, "row"].to_numpy()
    prob2[forced] = 1.0
    rows += t2_scores("T2-pair+reach", y, prob2, p)
    return rows, len(np.unique(forced))


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="sim_tables", help="output folder of python -m simulator.tables")
    ap.add_argument("--out", default="results_rules")
    ap.add_argument("--eval", default="val", choices=["val", "test"])
    a = ap.parse_args()
    tables, out = Path(a.tables), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = task1(tables, a.eval, out)
    r2, n_forced = task2(tables, a.eval)
    res = pd.DataFrame(rows + r2)
    res.to_csv(out / f"results_{a.eval}.csv", index=False)

    t1 = res[res.task == "task1"].pivot(index="method", columns="p", values="macro_f1")
    t1 = t1.loc[["T1-majority", "T1-markov", "T1-full"]]
    t2 = res[res.task == "task2"].pivot(index="method", columns="p", values="auroc")
    full = res[(res.task == "task1") & (res.method == "T1-full") & (res.p == "all")].iloc[0]
    text = "\n".join([
        f"evaluated on: {a.eval}   (tables counted on train only)   tables: {tables}", "",
        "Task 1 - behind the door, macro-F1 over bedroom/bathroom/kitchen, by share revealed:",
        t1.round(3).to_string(), "",
        "T1-full per-class F1 (all p): " + ", ".join(f"{NAMES[t]} {full[f'f1_{NAMES[t]}']:.3f}" for t in (1, 2, 3)),
        f"T1-full accuracy (all p): {full.accuracy:.3f}", "",
        "Task 2 - hidden doors, AUROC by share revealed:",
        t2.round(3).to_string(), "",
        f"reachability rule forced {n_forced} candidate pairs to 'connected'"])
    (out / f"summary_{a.eval}.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
