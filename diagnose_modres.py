"""
diagnose_modres.py - Team Alpha, Task 1 (follow-up to the first validation run)

Two jobs:

  python diagnose_modres.py doors
      Looks at every door polygon that did not match a doorway edge and counts WHY
      (touches 0 / 1 / 2 / 3+ rooms; for 2 rooms: what edge type joins them).
      Writes door_diagnosis_summary.txt and door_diagnosis.csv in --out.

  python diagnose_modres.py plans 1060 7083 1006
      Prints, for each plan, the JSON rooms next to the CSV nodes and the edge
      differences, so we can see exactly what disagrees.

Common options: --root "D:\\ModRes-main\\ModRes-main\\dataset" --out validation_out
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

import modres_loader as ml
from validate_modres import load_csvs, CSV_TOL_XY, CSV_TOL_VAL


def door_diagnosis(root: Path, out: Path, first, tol: float):
    n = len(pd.read_csv(ml.ModResPaths(root).adjacency_csv_dir() / "graphs.csv", usecols=["graph_id"]))
    ids = range(n if first is None else min(first, n))
    rows = []
    for i in ids:
        try:
            plan = ml.load_plan(i, root, tol)
        except Exception:
            continue
        ed = plan.edge_dict()
        for d in plan.doors:
            if d.status == "ok":
                continue
            near = d.near_rooms
            if len(near) == 0:
                cat = "touches_0_rooms"
                detail = ""
            elif len(near) == 1:
                cat = "touches_1_room"
                detail = ml.ROOM_TYPE_NAMES[plan.rooms[near[0]].type]
            elif len(near) == 2:
                e = ed.get((min(near), max(near)))
                cat = "touches_2_rooms_" + (e.type_name if e else "not_adjacent")
                detail = "-".join(sorted(plan.rooms[r].type_name for r in near))
            else:
                cat = "touches_3plus_rooms"
                detail = "-".join(sorted(plan.rooms[r].type_name for r in near))
            rows.append(dict(plan_id=i, door=d.index, category=cat, detail=detail,
                             door_area=round(d.polygon.area, 4),
                             door_w=round(max(d.polygon.bounds[2] - d.polygon.bounds[0],
                                              d.polygon.bounds[3] - d.polygon.bounds[1]), 3)))
    df = pd.DataFrame(rows, columns=["plan_id", "door", "category", "detail", "door_area", "door_w"])
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "door_diagnosis.csv", index=False)
    lines = [f"unmatched doors: {len(df)} in {df.plan_id.nunique()} plans (touch tol {tol} m)", ""]
    lines += [f"  {k}: {v}" for k, v in df.category.value_counts().items()]
    one = df[df.category == "touches_1_room"]
    if len(one):
        lines += ["", "doors touching exactly 1 room, by that room's type:"]
        lines += [f"  {k}: {v}" for k, v in one.detail.value_counts().items()]
    two = df[df.category.str.startswith("touches_2")]
    if len(two):
        lines += ["", "doors touching 2 rooms, by room-type pair (top 8):"]
        lines += [f"  {k}: {v}" for k, v in two.groupby(["category", "detail"]).size().sort_values(ascending=False).head(8).items()]
    text = "\n".join(lines)
    (out / "door_diagnosis_summary.txt").write_text(text, encoding="utf-8")
    print(text)


def plan_dump(root: Path, plan_ids, tol: float):
    graphs, nodes_by_g, edges_by_g = load_csvs(root)
    tname = ml.ROOM_TYPE_NAMES
    for i in plan_ids:
        print("=" * 70)
        print(f"PLAN {i}")
        plan = ml.load_plan(i, root, tol)
        cn = nodes_by_g[i].reset_index(drop=True)
        print(f"JSON rooms: {len(plan.rooms)}   CSV nodes: {len(cn)}")

        # match every JSON room to a CSV node of the same type at the same centroid (area ignored)
        used, mapping, rid_to_node = set(), {}, {}
        print("\nJSON rooms -> CSV partner (centroid match, area ignored):")
        for r in plan.rooms:
            best, bd = None, 1e9
            for _, row in cn.iterrows():
                nid = int(row["node_id"])
                if nid in used or int(row["label"]) != r.type:
                    continue
                d = max(abs(r.cx - row["X"]), abs(r.cy - row["Y"]))
                if d < bd:
                    best, bd = (nid, row), d
            if best is not None and bd <= CSV_TOL_XY:
                nid, row = best
                used.add(nid)
                mapping[nid] = r.rid
                rid_to_node[r.rid] = nid
                flag = "" if abs(r.area - row["feat_0"]) <= CSV_TOL_VAL else f"   CSV AREA = {row['feat_0']:.2f}"
                print(f"  rid {r.rid} {r.type_name:9s} area {r.area:6.2f} -> node {nid}{flag}")
            else:
                near = sorted(((max(abs(r.cx - x), abs(r.cy - y)), int(n), int(l))
                               for n, l, x, y in zip(cn.node_id, cn.label, cn.X, cn.Y)))[:2]
                print(f"  rid {r.rid} {r.type_name:9s} area {r.area:6.2f} -> NO PARTNER   "
                      f"(nearest CSV nodes: {[(round(d, 3), 'node', n, 'label', l) for d, n, l in near]})")
        extra = cn[~cn["node_id"].isin(used)]
        print("\nCSV nodes with no JSON room:")
        for _, row in extra.iterrows():
            print(f"  node {int(row['node_id'])} {tname.get(int(row['label']), '?'):9s} area {row['feat_0']:.2f}  "
                  f"at ({row['X']:.3f}, {row['Y']:.3f})")
        if len(extra) == 0:
            print("  none")

        # edges
        ce = edges_by_g.get(i)
        csv_pairs = {}
        extra_edges = []
        if ce is not None:
            for a_, b_, l in zip(ce.src_id, ce.dst_id, ce.label):
                a_, b_, l = int(a_), int(b_), int(l)
                if a_ in mapping and b_ in mapping:
                    u, v = sorted((mapping[a_], mapping[b_]))
                    csv_pairs[(u, v)] = l
                elif a_ < b_:
                    extra_edges.append((a_, b_, l))
        json_pairs = {(e.u, e.v): e.type for e in plan.edges}
        print("\nedge differences among matched rooms (CSV type vs JSON type):")
        diff = False
        for pair in sorted(set(csv_pairs) | set(json_pairs)):
            c, j = csv_pairs.get(pair), json_pairs.get(pair)
            if c != j:
                diff = True
                doors_between = sum(1 for d in plan.doors if pair[0] in d.near_rooms and pair[1] in d.near_rooms)
                nm = lambda t: ml.EDGE_TYPE_NAMES.get(t, "absent") if t is not None else "absent"
                print(f"  rooms {pair} ({plan.rooms[pair[0]].type_name}-{plan.rooms[pair[1]].type_name}): "
                      f"CSV={nm(c)}  JSON={nm(j)}   door polygons touching both rooms: {doors_between}")
        if not diff:
            print("  none")
        print("\nCSV edges that involve extra nodes (one direction):")
        for a_, b_, l in extra_edges:
            print(f"  node {a_} - node {b_}  type {ml.EDGE_TYPE_NAMES.get(l, l)}   "
                  f"(JSON room for node {a_}: {mapping.get(a_, 'none')}, node {b_}: {mapping.get(b_, 'none')})")
        if not extra_edges:
            print("  none")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["doors", "plans"])
    ap.add_argument("ids", nargs="*", type=int)
    ap.add_argument("--root", default=str(ml.DEFAULT_ROOT))
    ap.add_argument("--out", default="validation_out")
    ap.add_argument("--first", type=int, default=None)
    ap.add_argument("--touch-tol", type=float, default=ml.DEFAULT_TOUCH_TOL)
    a = ap.parse_args()
    if a.mode == "doors":
        door_diagnosis(Path(a.root), Path(a.out), a.first, a.touch_tol)
    else:
        if not a.ids:
            ap.error("give plan ids, e.g.  plans 1060 7083")
        plan_dump(Path(a.root), a.ids, a.touch_tol)


if __name__ == "__main__":
    main()