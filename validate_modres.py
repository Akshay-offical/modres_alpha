"""
validate_modres.py - Team Alpha, Task 1

Checks every ModRes plan: adjacency JSON vs adjacency CSV vs geometry JSON,
room-type constraints, edges, entrance and doors.

Usage (Windows, from the folder holding both .py files):
    python validate_modres.py
    python validate_modres.py --first 50          # quick test on plans 0..49
    python validate_modres.py --root "D:\\ModRes-main\\ModRes-main\\dataset" --out validation_out

Needs: shapely, pandas  (pip install shapely pandas)
Writes in --out: validation_report.csv, validation_summary.txt,
                 usable_plan_ids.txt, excluded_plan_ids.txt, benchmark_plan_ids.txt
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

import modres_loader as ml

CSV_TOL_XY = 1e-3        # centroid match between JSON and CSV (observed difference ~1e-6)
CSV_TOL_VAL = 0.011      # CSV values are rounded to 2 decimals
GEOM_SYMDIFF_TOL = 0.01  # relative symmetric difference between JSON and geometry polygons


# --------------------------------------------------------------------------- csv loading
def load_csvs(root: Path):
    d = ml.ModResPaths(root).adjacency_csv_dir()
    graphs = pd.read_csv(d / "graphs.csv").set_index("graph_id")
    nodes = pd.read_csv(d / "nodes.csv")
    edges = pd.read_csv(d / "edges.csv")
    nodes_by_g = {g: df for g, df in nodes.groupby("graph_id")}
    edges_by_g = {g: df for g, df in edges.groupby("graph_id")}
    return graphs, nodes_by_g, edges_by_g


# --------------------------------------------------------------------------- checks
def check_csv(plan: ml.Plan, graphs, nodes_by_g, edges_by_g, errors, warns):
    i = plan.plan_id
    if i not in graphs.index or i not in nodes_by_g:
        errors.append("csv_missing_graph")
        return
    g = graphs.loc[i]
    p = plan.props
    pairs = [(g["feat_0"], p["area"]), (g["feat_1"], p["net_area"]),
             (g["feat_2"], p["perimeter"]), (g["feat_3"], p["compactness"])]
    if any(abs(a - b) > CSV_TOL_VAL for a, b in pairs):
        errors.append("csv_graph_mismatch")
    # inferred (not named in the files): label = bedrooms, feat_4..7 = unit type one-hot
    if int(round(g["label"])) != int(p["bedrooms"]):
        warns.append("csv_label_not_bedrooms")
    unit = [g[f"feat_{k}"] for k in (4, 5, 6, 7)]
    if any(abs(unit[k] - p[f"unit_type_{k}"]) > 1e-6 for k in range(4)):
        warns.append("csv_unit_type_mismatch")

    # CSV is only a cross-check: JSON + geometry are the source of truth (see diagnose_modres.py).
    # nodes: every JSON room needs a CSV node of the same type at the same centroid.
    # Findings from plans 1060 / 7083: the CSV can contain EXTRA zero-area bathroom nodes
    # (and sometimes shows a real room with area 0.00). The JSON agrees with the geometry
    # files, so JSON is the source of truth: extra / zero-area CSV nodes are warnings,
    # a JSON room with no CSV partner is an error.
    cn = nodes_by_g[i]
    mapping = {}
    used = set()
    for r in plan.rooms:
        best, bd = None, 1e9
        for _, row in cn.iterrows():
            nid = int(row["node_id"])
            if nid in used or int(row["label"]) != r.type:
                continue
            d = max(abs(r.cx - row["X"]), abs(r.cy - row["Y"]))
            if d < bd:
                best, bd = (nid, row), d
        if best is None or bd > CSV_TOL_XY:
            # diagnosed on plans 1066/1067: the CSV keeps the room's position and edges but
            # writes a wrong type and area 0.00. The JSON agrees with geometry + door polygons,
            # so this is a CSV export defect, not a reason to drop the plan.
            warns.append("csv_node_mismatch")
            return
        nid, row = best
        used.add(nid)
        mapping[nid] = r.rid
        if abs(r.area - row["feat_0"]) > CSV_TOL_VAL:
            warns.append("csv_zero_area_node" if row["feat_0"] == 0 else "csv_area_mismatch")
    extra = cn[~cn["node_id"].isin(used)]
    if len(extra):
        warns.append("csv_extra_nodes")
        if (extra["feat_0"] == 0).all():
            warns.append("csv_extra_nodes_all_zero_area")

    # edges: compare directed sets
    ce = edges_by_g.get(i)
    csv_set = set()
    if ce is not None:
        for _, row in ce.iterrows():
            a, b = int(row["src_id"]), int(row["dst_id"])
            if a not in mapping or b not in mapping:
                warns.append("csv_edge_on_extra_node")
                continue
            csv_set.add((mapping[a], mapping[b], int(row["label"])))
    json_set = set()
    for e in plan.edges:
        json_set.add((e.u, e.v, e.type))
        json_set.add((e.v, e.u, e.type))
    if csv_set != json_set:
        # diagnosed on plans 1060/1063: the CSV routes a doorway through an extra zero-area node.
        # JSON doorway edges each have exactly one door polygon, so the JSON is trusted.
        warns.append("csv_edge_mismatch")


def check_geometry(plan: ml.Plan, root, errors, warns):
    try:
        geom = ml.load_geometry_for_check(plan.plan_id, root)
    except Exception:
        errors.append("geometry_unreadable")
        return
    for key, t in ml.GEOMETRY_ROOM_KEYS.items():
        rooms = plan.rooms_of_type(t)
        polys = list(geom[key])
        if len(rooms) != len(polys):
            errors.append("geom_room_count_mismatch")
            return
        for r in rooms:
            hit = None
            for k, gp in enumerate(polys):
                if r.polygon.area > 0 and r.polygon.symmetric_difference(gp).area / r.polygon.area < GEOM_SYMDIFF_TOL:
                    hit = k
                    break
            if hit is None:
                errors.append("geom_room_polygon_mismatch")
                return
            polys.pop(hit)


def check_structure(plan: ml.Plan, errors, warns):
    types = [r.type for r in plan.rooms]
    if any(t not in ml.ROOM_TYPE_NAMES for t in types):
        errors.append("bad_room_type")
        return
    c = Counter(types)
    if c[3] != 1:
        errors.append("kitchen_count_not_1")
    if c[0] < 1 or c[1] < 1 or c[2] < 1:
        errors.append("missing_required_room_type")
    # inferred rule from the paper: bedrooms property = number of bedrooms - 1
    if c[1] != int(plan.props["bedrooms"]) + 1:
        warns.append("bedroom_property_mismatch")

    for note in plan.notes:
        tag = note.split(":")[0]
        if tag in ("edge_unknown_endpoint", "edge_self_loop", "edge_conflict"):
            errors.append(tag)
        elif tag == "edge_duplicate":
            warns.append(tag)
        elif tag.startswith("geometry_polygon"):
            warns.append(tag)
    if any(not r.polygon_valid for r in plan.rooms):
        warns.append("room_polygon_repaired")

    # connectivity (own union-find, no networkx needed)
    def connected(edge_filter):
        parent = list(range(len(plan.rooms)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for e in plan.edges:
            if edge_filter(e):
                parent[find(e.u)] = find(e.v)
        return len({find(x) for x in range(len(plan.rooms))}) == 1

    if not connected(lambda e: True):
        errors.append("adjacency_disconnected")
    if not connected(lambda e: e.connected):
        warns.append("accessible_disconnected")


def check_entrance_and_doors(plan: ml.Plan, errors, warns):
    ent = plan.entrance
    if ent.status in ("no_front_door", "not_found"):
        errors.append("entrance_" + ent.status)
    elif ent.status == "ambiguous":
        warns.append("entrance_ambiguous")
    if ent.n_front_doors > 1:
        warns.append("multiple_front_doors")

    for d in plan.doors:
        if d.status != "ok":
            warns.append("door_" + d.status)
    covered = {d.rooms for d in plan.doors if d.rooms}
    for e in plan.edges:
        if e.type == 0 and (e.u, e.v) not in covered:
            warns.append("doorway_edge_without_door")


# --------------------------------------------------------------------------- main
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ml.DEFAULT_ROOT))
    ap.add_argument("--out", default="validation_out")
    ap.add_argument("--first", type=int, default=None, help="only plans 0..N-1")
    ap.add_argument("--touch-tol", type=float, default=ml.DEFAULT_TOUCH_TOL)
    return ap.parse_args()


def main():
    args = parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    graphs, nodes_by_g, edges_by_g = load_csvs(root)
    n_total = len(graphs)
    ids = list(range(n_total if args.first is None else min(args.first, n_total)))
    print(f"graphs.csv rows: {n_total}; validating {len(ids)} plans")

    rows = []
    t0 = time.time()
    for k, i in enumerate(ids):
        errors, warns = [], []
        row = dict(plan_id=i)
        try:
            plan = ml.load_plan(i, root, args.touch_tol)
        except Exception as ex:
            errors.append("load_failed")
            row.update(status="FAIL", errors=";".join(errors), warnings=str(ex)[:120])
            rows.append(row)
            continue
        check_structure(plan, errors, warns)
        check_csv(plan, graphs, nodes_by_g, edges_by_g, errors, warns)
        check_geometry(plan, root, errors, warns)
        check_entrance_and_doors(plan, errors, warns)

        ec = Counter(e.type for e in plan.edges)
        tc = Counter(r.type for r in plan.rooms)
        row.update(
            n_rooms=len(plan.rooms), n_bedrooms=tc[1], n_bathrooms=tc[2],
            n_doorway=ec[0], n_passage=ec[1], n_wall=ec[2],
            n_doors=len(plan.doors), n_front_doors=plan.entrance.n_front_doors,
            entrance_type=(ml.ROOM_TYPE_NAMES[plan.rooms[plan.entrance.room].type]
                           if plan.entrance.room is not None else ""),
            entrance_dist=round(plan.entrance.distance, 4),
            status="FAIL" if errors else "PASS",
            errors=";".join(sorted(set(errors))), warnings=";".join(sorted(set(warns))))
        rows.append(row)
        if (k + 1) % 500 == 0:
            print(f"  {k + 1}/{len(ids)}  ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(out / "validation_report.csv", index=False)
    usable = df[df.status == "PASS"].plan_id.tolist()
    excluded = df[df.status != "PASS"].plan_id.tolist()
    (out / "usable_plan_ids.txt").write_text("\n".join(map(str, usable)))
    (out / "excluded_plan_ids.txt").write_text("\n".join(map(str, excluded)))
    bench = df[(df.status == "PASS") & ~df.warnings.fillna("").str.contains("accessible_disconnected")].plan_id.tolist()
    (out / "benchmark_plan_ids.txt").write_text("\n".join(map(str, bench)))

    # summary
    err_ids, warn_ids = defaultdict(list), defaultdict(list)
    for _, r in df.iterrows():
        for e in str(r.get("errors", "") or "").split(";"):
            if e and e != "nan":
                err_ids[e].append(r.plan_id)
        for w in str(r.get("warnings", "") or "").split(";"):
            if w and w != "nan":
                warn_ids[w].append(r.plan_id)
    lines = [f"plans checked: {len(df)}", f"PASS: {len(usable)}", f"FAIL: {len(excluded)}",
             f"BENCHMARK (PASS and fully walkable through doorways/passages): {len(bench)}", "",
             "ERRORS (plan excluded):"]
    lines += [f"  {k}: {len(v)}  e.g. {v[:5]}" for k, v in sorted(err_ids.items())] or ["  none"]
    lines += ["", "WARNINGS (plan kept, look at them):"]
    lines += [f"  {k}: {len(v)}  e.g. {v[:5]}" for k, v in sorted(warn_ids.items())] or ["  none"]
    ok = df[df.status == "PASS"]
    if len(ok):
        lines += ["", "among PASS plans:",
                  f"  rooms per plan   : {ok.n_rooms.value_counts().sort_index().to_dict()}",
                  f"  bedrooms per plan: {ok.n_bedrooms.value_counts().sort_index().to_dict()}",
                  f"  entrance room type: {ok.entrance_type.value_counts().to_dict()}",
                  f"  plans with >=1 passage: {(ok.n_passage > 0).sum()}",
                  f"  entrance distance max: {ok.entrance_dist.max()}"]
    text = "\n".join(lines)
    (out / "validation_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    sys.exit(main())