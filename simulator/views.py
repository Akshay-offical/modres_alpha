"""
views.py - PLACEHOLDER simulator output, so the GNN can be built before the real simulator exists.

A "view" is what the robot has seen after exploring part of a plan:
    visited  rooms entered           (type, area known)
    seen     rooms visible through an open passage from a visited room (type known, area NOT known)
    stub     rooms behind an unopened doorway (type UNKNOWN)  <- what Task 1 predicts
    unseen   everything else         (not in the view at all)

Nothing about unseen rooms, and nothing about the TYPE or AREA of a stub, may enter the features.
test_views.py checks this.

When the real simulator is ready, it only has to produce the same View objects (or call
`view_to_data`), and nothing else in the pipeline changes.

Edge types: 0 doorway, 1 passage, 2 wall.  Room types: 0 living, 1 bedroom, 2 bathroom, 3 kitchen.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

N_TYPES = 4
N_EDGE_TYPES = 3
DOORWAY, PASSAGE, WALL = 0, 1, 2
VISITED, SEEN, STUB = 0, 1, 2
AREA_SCALE = 30.0      # m^2, only to keep the feature near 0..1
HOP_SCALE = 6.0

# feature columns (documented once, used in tests)
#  0-3  room type one-hot (all zero for stubs)      4 type unknown (1 for stubs)
#  5-7  state one-hot visited/seen/stub             8 is_entrance
#  9    area / AREA_SCALE (visited rooms only, else 0)
#  10   hops from the entrance inside the view / HOP_SCALE
#  11   observed fraction of the plan (0.25/0.5/0.75)
N_FEATURES = 12


# --------------------------------------------------------------------------------------------
# A plan, reduced to what the simulator needs
# --------------------------------------------------------------------------------------------
@dataclass
class PlanGraph:
    plan_id: int
    types: List[int]                      # room type per rid
    areas: List[float]
    edges: Dict[Tuple[int, int], int]     # (u<v) -> edge type
    entrance: int
    coords: Optional[List[Tuple[float, float]]] = None   # used only to pick the nearest next room

    @property
    def n(self) -> int:
        return len(self.types)

    def etype(self, a: int, b: int) -> Optional[int]:
        return self.edges.get((min(a, b), max(a, b)))

    def connected_neighbours(self, a: int) -> List[int]:
        out = []
        for (u, v), t in self.edges.items():
            if t in (DOORWAY, PASSAGE):
                if u == a:
                    out.append(v)
                elif v == a:
                    out.append(u)
        return out


def plan_from_loader(plan) -> PlanGraph:
    """Convert a modres_loader.Plan (needs plan.entrance.room)."""
    return PlanGraph(
        plan_id=plan.plan_id,
        types=[r.type for r in plan.rooms],
        areas=[r.area for r in plan.rooms],
        edges={(e.u, e.v): e.type for e in plan.edges},
        entrance=plan.entrance.room,
        coords=[(r.cx, r.cy) for r in plan.rooms],
    )


# --------------------------------------------------------------------------------------------
# Synthetic plans (only for testing the code without the dataset; NOT a result of any kind)
# --------------------------------------------------------------------------------------------
def synthetic_plans(n_plans: int, seed: int = 0) -> List[PlanGraph]:
    rng = random.Random(seed)
    plans = []
    for pid in range(n_plans):
        n_bed = rng.choice([1, 2, 2, 3, 4])
        n_bath = rng.choice([1, 1, 2])
        types = [0, 3] + [1] * n_bed + [2] * n_bath          # living, kitchen, bedrooms, bathrooms
        areas = []
        for t in types:
            lo, hi = {0: (18, 30), 1: (9, 16), 2: (4, 8), 3: (7, 12)}[t]
            areas.append(rng.uniform(lo, hi))
        edges: Dict[Tuple[int, int], int] = {}

        def add(a, b, t):
            edges[(min(a, b), max(a, b))] = t

        add(0, 1, rng.choice([DOORWAY, PASSAGE]))             # living - kitchen
        bedrooms = [i for i, t in enumerate(types) if t == 1]
        baths = [i for i, t in enumerate(types) if t == 2]
        for b in bedrooms:                                    # bedrooms hang off living (or off another bedroom)
            parent = 0 if rng.random() < 0.7 else rng.choice(bedrooms[: bedrooms.index(b)] or [0])
            add(parent, b, DOORWAY)
        for k, ba in enumerate(baths):                        # bathrooms next to a bedroom or living
            parent = rng.choice(bedrooms) if rng.random() < 0.75 else 0
            add(parent, ba, DOORWAY)
        # a few wall edges (touching, not connected)
        ids = list(range(len(types)))
        for _ in range(rng.randint(1, 3)):
            a, b = rng.sample(ids, 2)
            if (min(a, b), max(a, b)) not in edges:
                add(a, b, WALL)
        # shuffle rids so the order carries no information
        perm = list(range(len(types)))
        rng.shuffle(perm)
        inv = {old: new for new, old in enumerate(perm)}
        types2 = [None] * len(types)
        areas2 = [None] * len(types)
        for old, new in inv.items():
            types2[new], areas2[new] = types[old], areas[old]
        edges2 = {(min(inv[a], inv[b]), max(inv[a], inv[b])): t for (a, b), t in edges.items()}
        plans.append(PlanGraph(pid, types2, areas2, edges2, entrance=inv[0]))
    return plans


# --------------------------------------------------------------------------------------------
# A view
# --------------------------------------------------------------------------------------------
@dataclass
class View:
    plan_id: int
    fraction: float                      # requested fraction (0.25 / 0.5 / 0.75)
    nodes: List[int]                     # rids of rooms that are in the view
    state: Dict[int, int]                # rid -> VISITED / SEEN / STUB
    edges: List[Tuple[int, int, int]]    # (u, v, type) among nodes, u<v
    truth: Dict[int, int]                # stub rid -> true room type (kept apart from the features)
    entrance: int
    n_visited: int
    n_rooms: int
    features: np.ndarray = field(default=None, repr=False)   # (len(nodes), N_FEATURES)


def _target_count(n_rooms: int, fraction: float) -> int:
    k = int(round(fraction * n_rooms))
    return max(1, min(k, n_rooms - 1))


def make_views(plan: PlanGraph, fractions=(0.25, 0.5, 0.75), seed: int = 0) -> List[View]:
    """One walk, snapshots at each fraction (so 25% is inside 50% is inside 75%).

    Placeholder walk rule: from everything the robot has reached through doorways / passages, go to the
    nearest unvisited room (centroid distance, if coordinates exist; random otherwise), random tie-break.
    Passages are 'seen' from the neighbouring room; doorways are stubs until the robot opens them.
    The REAL simulator replaces this function.
    """
    rng = random.Random((seed * 1_000_003) ^ (plan.plan_id * 7919))
    visited = [plan.entrance]
    vset = {plan.entrance}
    targets = {f: _target_count(plan.n, f) for f in fractions}
    snaps: Dict[float, List[int]] = {}
    current = plan.entrance

    def check_snaps():
        for f in fractions:
            if f not in snaps and len(visited) >= targets[f]:
                snaps[f] = list(visited)

    check_snaps()
    while len(visited) < max(targets.values()):
        frontier = sorted({v for a in visited for v in plan.connected_neighbours(a) if v not in vset})
        if not frontier:
            break
        if plan.coords is not None:
            cx, cy = plan.coords[current]
            dist = {v: math.hypot(plan.coords[v][0] - cx, plan.coords[v][1] - cy) for v in frontier}
            best = min(dist.values())
            cand = [v for v in frontier if dist[v] <= best + 1e-9]
        else:
            cand = frontier
        nxt = rng.choice(cand)
        visited.append(nxt)
        vset.add(nxt)
        current = nxt
        check_snaps()

    views = []
    for f in fractions:
        if f in snaps:
            v = _build_view(plan, snaps[f], f)
            if v is not None:
                views.append(v)
    return views


def _build_view(plan: PlanGraph, visited_list: List[int], fraction: float) -> Optional[View]:
    vset = set(visited_list)
    state: Dict[int, int] = {r: VISITED for r in visited_list}
    # seen via passage first (type known), then stubs via doorway
    for a in visited_list:
        for b in plan.connected_neighbours(a):
            if b not in vset and plan.etype(a, b) == PASSAGE:
                state[b] = SEEN
    for a in visited_list:
        for b in plan.connected_neighbours(a):
            if b not in vset and b not in state and plan.etype(a, b) == DOORWAY:
                state[b] = STUB
    truth = {r: plan.types[r] for r, s in state.items() if s == STUB}
    if not truth:
        return None                      # nothing to predict in this view
    nodes = sorted(state)
    known = {r for r, s in state.items() if s in (VISITED, SEEN)}
    edges = []
    for (u, v), t in plan.edges.items():
        if u in state and v in state:
            if t == WALL:
                if u in known and v in known:          # wall adjacency is only revealed between known rooms
                    edges.append((u, v, t))
            elif (u in vset) or (v in vset):           # a door / passage the robot has actually seen
                edges.append((u, v, t))
    view = View(plan.plan_id, fraction, nodes, state, edges, truth, plan.entrance,
                len(visited_list), plan.n)
    view.features = _features(plan, view)
    return view


def _hops_from_entrance(view: View) -> Dict[int, int]:
    adj: Dict[int, List[int]] = {r: [] for r in view.nodes}
    for u, v, t in view.edges:
        if t in (DOORWAY, PASSAGE):
            adj[u].append(v)
            adj[v].append(u)
    hops = {view.entrance: 0}
    queue = [view.entrance]
    for a in queue:
        for b in adj[a]:
            if b not in hops:
                hops[b] = hops[a] + 1
                queue.append(b)
    return hops


def _features(plan: PlanGraph, view: View) -> np.ndarray:
    hops = _hops_from_entrance(view)
    X = np.zeros((len(view.nodes), N_FEATURES), dtype=np.float32)
    for i, r in enumerate(view.nodes):
        s = view.state[r]
        if s == STUB:
            X[i, 4] = 1.0                                   # type unknown; nothing else about the room
        else:
            X[i, plan.types[r]] = 1.0
        X[i, 5 + s] = 1.0
        X[i, 8] = 1.0 if r == view.entrance else 0.0
        if s == VISITED:
            X[i, 9] = plan.areas[r] / AREA_SCALE
        X[i, 10] = hops.get(r, 0) / HOP_SCALE
        X[i, 11] = view.fraction
    return X


# --------------------------------------------------------------------------------------------
# View -> PyTorch Geometric Data
# --------------------------------------------------------------------------------------------
def view_to_data(view: View):
    import torch
    from torch_geometric.data import Data

    index = {r: i for i, r in enumerate(view.nodes)}
    src, dst, et = [], [], []
    for u, v, t in view.edges:                              # both directions
        a, b = index[u], index[v]
        src += [a, b]
        dst += [b, a]
        et += [t, t]
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    edge_type = torch.tensor(et, dtype=torch.long)
    y = torch.full((len(view.nodes),), -1, dtype=torch.long)
    for r, t in view.truth.items():
        y[index[r]] = t
    stub_mask = y >= 0
    d = Data(x=torch.from_numpy(view.features), edge_index=edge_index, edge_type=edge_type,
             y=y, stub_mask=stub_mask)
    d.plan_id = view.plan_id
    d.fraction = float(view.fraction)
    return d
