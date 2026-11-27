r"""
tables.py - write the simulator's views as tables (same plans, split and walks as gnn/train.py).

    python -m simulator.tables --source modres --root "D:\ModRes\dataset" --ids validation_out\benchmark_plan_ids.txt

Output files and columns: docs/simulator.md.
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from data.splits import sample_ids, split_by_plan
from simulator.explore import FRACTIONS, POLICIES, STOP_RULES, TYPE_NAMES, build_view, walk_snapshots
from simulator.views import DOORWAY, PASSAGE, _hops_from_entrance, plan_from_loader, synthetic_plans

STATE_NAMES = {0: "visited", 1: "seen", 2: "stub"}
TABLES = ("snapshots", "nodes", "edges", "labels_stub", "labels_remaining", "task2")
# walk seeds per split, the same as build(...) in gnn/train.py
BASE_SEED = {"train": 10_000, "val": 20_000, "test": 30_000}


def load_plans(a):
    if a.source == "synthetic":
        return synthetic_plans(a.plans, seed=a.seed)
    from data import modres_loader as ml
    if not a.ids:
        raise SystemExit("--ids is required with --source modres (e.g. validation_out\\benchmark_plan_ids.txt)")
    ids = sample_ids([int(s) for s in Path(a.ids).read_text().split()], a.limit, a.split_seed)
    root = a.root or ml.DEFAULT_ROOT
    plans, t0 = [], time.time()
    for k, i in enumerate(ids, 1):
        plans.append(plan_from_loader(ml.load_plan(i, root)))
        if k % 1000 == 0:
            print(f"  loaded {k}/{len(ids)} plans ({time.time() - t0:.0f}s)")
    return plans


def plan_rows(plan, split, walks, policy, stop, rows):
    """Add the rows of every view of one plan to `rows`."""
    made = set()                                  # (p, visited rooms) already seen in this plan
    for w in range(walks):
        seed = BASE_SEED[split] + w
        snapshots = walk_snapshots(plan, FRACTIONS, seed, policy, stop)
        for p in FRACTIONS:
            if p not in snapshots:                # robot knows more than p already in the entrance
                rows["dropped"].append((plan.plan_id, p, seed))
                continue
            state = snapshots[p]
            if not state.behind_door:             # nothing to predict -> no view (as in make_views)
                rows["dropped"].append((plan.plan_id, p, seed))
                continue
            key = (p, frozenset(state.visited))
            is_duplicate = key in made
            made.add(key)

            view = build_view(plan, state, p)
            hops = _hops_from_entrance(view)
            known = state.known
            base = dict(snapshot_id=f"{plan.plan_id}_{int(p * 100)}_{w}", plan_id=plan.plan_id, split=split)

            for room in view.nodes:
                is_stub = room in state.behind_door
                rows["nodes"].append(dict(**base, rid=room, state=STATE_NAMES[view.state[room]],
                                          type=-1 if is_stub else plan.types[room],
                                          area=float("nan") if is_stub else plan.areas[room],
                                          is_entrance=int(room == plan.entrance), hops=hops[room]))
            for u, v, kind in view.edges:
                rows["edges"].append(dict(**base, u=u, v=v, edge_type=kind))
            for room, t in view.truth.items():
                rows["labels_stub"].append(dict(**base, rid=room, label=t, label_name=TYPE_NAMES[t]))

            unknown = [room for room in range(plan.n) if room not in known]
            per_type = Counter(plan.types[room] for room in unknown)
            rows["labels_remaining"].append(dict(**base, n_unknown=len(unknown),
                                                 **{f"unknown_{name}": per_type[t] for t, name in TYPE_NAMES.items()}))
            for (u, v), kind in sorted(plan.edges.items()):
                if u in known and v in known:
                    continue
                rows["task2"].append(dict(**base, p=p, u=u, v=v, u_known=int(u in known), v_known=int(v in known),
                                          u_type=plan.types[u], v_type=plan.types[v],
                                          true_edge_type=kind, label=int(kind in (DOORWAY, PASSAGE))))

            rows["snapshots"].append(dict(**base, p=p, walk=w, seed=seed, policy=policy, stop=stop, n_rooms=plan.n,
                                          n_visited=len(state.visited), n_seen=len(state.seen),
                                          n_stub=len(state.behind_door),
                                          n_hidden=plan.n - len(known) - len(state.behind_door),
                                          observed_fraction=round(len(known) / plan.n, 4),
                                          visit_order=list(state.visited),
                                          is_duplicate=int(is_duplicate)))


def summary(tables, n_dropped, a):
    sn, nd, lab, t2 = tables["snapshots"], tables["nodes"], tables["labels_stub"], tables["task2"]
    lab_p = lab.merge(sn[["snapshot_id", "p"]], on="snapshot_id")
    return "\n".join([
        f"source: {a.source}   policy: {a.policy}   stop: {a.stop}   walks per plan: train {a.train_walks}, val/test {a.eval_walks}"
        f"   split seed: {a.split_seed}",
        f"plans: {sn.plan_id.nunique()}   views: {len(sn)}   of which duplicates of an earlier walk: "
        f"{int(sn.is_duplicate.sum())}   snapshots dropped (none within the limit, or nothing behind a door): {n_dropped}", "",
        "views per split and p (duplicates included, as the GNN gets them):",
        sn.groupby(["split", "p"]).size().unstack().to_string(), "",
        "actual observed fraction (mean) by target p:",
        sn.groupby("p").observed_fraction.mean().round(3).to_string(), "",
        "rooms per state (all views):", nd.state.value_counts().to_string(), "",
        "Task 1 stubs (questions) per split and p:",
        lab_p.groupby(["split", "p"]).size().unstack().to_string(), "",
        "Task 1 label distribution (train):",
        lab[lab.split == "train"].label_name.value_counts(normalize=True).round(3).to_string(), "",
        "Task 2 share of candidates that are real connections (train, by p):",
        t2[t2.split == "train"].groupby("p").label.mean().round(3).to_string()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "modres"], default="synthetic")
    ap.add_argument("--plans", type=int, default=300, help="synthetic only")
    ap.add_argument("--root", default=None)
    ap.add_argument("--ids", default=None, help="file with plan ids (benchmark_plan_ids.txt)")
    ap.add_argument("--limit", type=int, default=0, help="random sample of N plans (same sample as gnn.train)")
    ap.add_argument("--seed", type=int, default=0, help="synthetic plans only")
    ap.add_argument("--split-seed", type=int, default=0, help="same meaning as in gnn.train")
    ap.add_argument("--train-walks", type=int, default=5, help="same default as gnn.train")
    ap.add_argument("--eval-walks", type=int, default=3, help="same default as gnn.train")
    ap.add_argument("--policy", choices=sorted(POLICIES), default="bfs")
    ap.add_argument("--stop", choices=STOP_RULES, default="nearest")
    ap.add_argument("--out", default="sim_tables")
    a = ap.parse_args()

    plans = load_plans(a)
    rows = {name: [] for name in TABLES + ("dropped",)}
    for split, part in zip(("train", "val", "test"), split_by_plan(plans, a.split_seed)):
        walks = a.train_walks if split == "train" else a.eval_walks
        for plan in part:
            plan_rows(plan, split, walks, a.policy, a.stop, rows)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = {name: pd.DataFrame(rows[name]) for name in TABLES}
    for name, table in tables.items():
        fname = "task2_candidates" if name == "task2" else name
        table.to_parquet(out / f"{fname}.parquet", index=False)
    text = summary(tables, len(rows["dropped"]), a)
    (out / "sim_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
