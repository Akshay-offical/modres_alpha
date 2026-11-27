r"""
compare.py - evaluate the exploration methods (policies) and stop rules on the same plans and seeds.

    python -m simulator.compare --root "D:\ModRes\dataset" --ids validation_out\benchmark_plan_ids.txt [--limit 300]

Writes sim_compare/walks.parquet (one row per plan, method, stop rule, seed and fraction) and
sim_compare/summary.csv / summary.txt. What each column means: docs/simulator.md.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from data.splits import sample_ids
from simulator.explore import FRACTIONS, POLICIES, STOP_RULES, walk_snapshots
from simulator.views import plan_from_loader, synthetic_plans

KITCHEN = 3


def plan_rows(plan, seeds, rows):
    for policy in POLICIES:
        for stop in STOP_RULES:
            made = set()
            for seed in range(seeds):
                snaps = walk_snapshots(plan, FRACTIONS, seed, policy, stop)
                for p in FRACTIONS:
                    s = snaps.get(p)
                    row = dict(plan_id=plan.plan_id, n_rooms=plan.n, policy=policy, stop=stop, seed=seed, p=p,
                               has_view=int(s is not None and bool(s.behind_door)))
                    if s is not None:
                        key = (p, frozenset(s.visited))
                        known = len(s.known)
                        row.update(observed=known / plan.n, visits=len(s.visited),
                                   stubs=len(s.behind_door), hidden=plan.n - known - len(s.behind_door),
                                   kitchen_stubs=sum(plan.types[r] == KITCHEN for r in s.behind_door),
                                   duplicate=int(key in made))
                        made.add(key)
                    rows.append(row)


def summarise(df):
    v = df[df.has_view == 1]
    g = v.groupby(["policy", "stop", "p"])
    out = pd.DataFrame({
        "views": g.size(),
        "coverage": df.groupby(["policy", "stop", "p"])["has_view"].mean(),
        "observed_mean": g["observed"].mean(),
        "observed_max": g["observed"].max(),
        "above_p": g.apply(lambda d: (d.observed > d.p + 1e-9).mean()),
        "gap_to_p": g.apply(lambda d: (d.observed - d.p).abs().mean()),
        "visits": g["visits"].mean(),
        "stubs_per_view": g["stubs"].mean(),
        "hidden_per_view": g["hidden"].mean(),
        "kitchen_share": g["kitchen_stubs"].sum() / g["stubs"].sum(),
        "duplicates": g["duplicate"].mean(),
    })
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "modres"], default="modres")
    ap.add_argument("--plans", type=int, default=300, help="synthetic only")
    ap.add_argument("--root", default=None)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3, help="walks per plan and method")
    ap.add_argument("--out", default="sim_compare")
    a = ap.parse_args()

    if a.source == "synthetic":
        plans = synthetic_plans(a.plans, seed=0)
    else:
        from data import modres_loader as ml
        if not a.ids:
            raise SystemExit("--ids is required with --source modres")
        ids = sample_ids([int(s) for s in Path(a.ids).read_text().split()], a.limit, a.split_seed)
        plans = [plan_from_loader(ml.load_plan(i, a.root or ml.DEFAULT_ROOT)) for i in ids]

    rows, t0 = [], time.time()
    for k, plan in enumerate(plans, 1):
        plan_rows(plan, a.seeds, rows)
        if k % 1000 == 0:
            print(f"  {k}/{len(plans)} plans ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "walks.parquet", index=False)
    summary = summarise(df)
    summary.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 220)
    text = (f"{len(plans)} plans, {a.seeds} walks per plan and method\n\n"
            + summary.round(3).to_string(index=False))
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
