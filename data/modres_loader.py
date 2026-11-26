"""
modres_loader.py - Team Alpha, Task 1

Reads one ModRes plan into a clean Plan object.

Sources used (all paths relative to ROOT = ...\\ModRes-main\\ModRes-main\\dataset):
  Graphs/Adjacency Graphs/Continuous/JSON/adjacency_graph_XXXX.json  -> rooms, edges
  Geometry/2D/geometry_XXXX.json                                     -> front door, doors

Facts taken from the supplied files (plan 0000) and the ModRes paper:
  room type : 0 living, 1 bedroom, 2 bathroom, 3 kitchen
  edge type : 0 doorway, 1 passage (open, no door), 2 wall (shared wall, NOT connected)
  units     : metres (inferred from coordinates, not stated in the files)
  JSON edges reference the vertices dictionary KEYS ("Vertex_0003"), not the "id" field.
  The room index used here (rid) is the position of the key in the JSON "vertices"
  dictionary. It is NOT the node_id of the CSV files (the order differs).

Requires: shapely  (pip install shapely)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from shapely.geometry import Polygon

DEFAULT_ROOT = Path(r"D:\ModRes-main\ModRes-main\dataset")

ROOM_TYPE_NAMES = {0: "living", 1: "bedroom", 2: "bathroom", 3: "kitchen"}
EDGE_TYPE_NAMES = {0: "doorway", 1: "passage", 2: "wall"}
GEOMETRY_ROOM_KEYS = {"living": 0, "bedroom": 1, "bathroom": 2, "kitchen": 3}

# Distance (metres) within which a door / front-door polygon counts as touching a room.
# First run at 0.15 m gave 684 ambiguous doors and 90 ambiguous entrances; at 0.03 m only 27 and 0.
# Door polygons overlap their rooms in plan 0000, so a tight tolerance is enough.
DEFAULT_TOUCH_TOL = 0.03


class ModResError(Exception):
    """Raised when a plan file cannot be parsed."""


# --------------------------------------------------------------------------- paths
class ModResPaths:
    def __init__(self, root: Path | str = DEFAULT_ROOT):
        self.root = Path(root)

    def adjacency_json(self, i: int) -> Path:
        return self.root / "Graphs" / "Adjacency Graphs" / "Continuous" / "JSON" / f"adjacency_graph_{i:04d}.json"

    def geometry_json(self, i: int) -> Path:
        return self.root / "Geometry" / "2D" / f"geometry_{i:04d}.json"

    def adjacency_csv_dir(self) -> Path:
        return self.root / "Graphs" / "Adjacency Graphs" / "Continuous" / "CSV"


# --------------------------------------------------------------------------- data classes
@dataclass
class Room:
    rid: int                  # position in the JSON "vertices" dictionary
    key: str                  # JSON dictionary key, e.g. "Vertex_0003"
    type: int                 # 0..3
    area: float
    perimeter: float
    compactness: float
    cx: float
    cy: float
    polygon: Polygon
    polygon_valid: bool

    @property
    def type_name(self) -> str:
        return ROOM_TYPE_NAMES.get(self.type, f"unknown_{self.type}")


@dataclass
class Edge:
    u: int                    # always u < v
    v: int
    type: int                 # 0 doorway, 1 passage, 2 wall

    @property
    def type_name(self) -> str:
        return EDGE_TYPE_NAMES.get(self.type, f"unknown_{self.type}")

    @property
    def connected(self) -> bool:
        """True for doorway and passage (the robot can walk through)."""
        return self.type in (0, 1)


@dataclass
class Door:
    index: int
    polygon: Polygon
    rooms: Optional[Tuple[int, int]]   # matched doorway edge (u, v) or None
    status: str                        # ok | no_doorway_pair | ambiguous
    near_rooms: List[int] = field(default_factory=list)


@dataclass
class Entrance:
    room: Optional[int]        # rid of the entrance room, None if not found
    status: str                # ok | ambiguous | not_found | no_front_door
    distance: float            # distance from front door polygon to that room
    candidates: List[int] = field(default_factory=list)
    n_front_doors: int = 0


@dataclass
class Plan:
    plan_id: int
    props: dict
    rooms: List[Room]
    edges: List[Edge]
    entrance: Entrance
    doors: List[Door]
    raw_edge_count: int
    notes: List[str] = field(default_factory=list)   # parse-level problems (self loops, conflicts...)

    # ---- convenience
    def edge_dict(self) -> Dict[Tuple[int, int], Edge]:
        return {(e.u, e.v): e for e in self.edges}

    def rooms_of_type(self, t: int) -> List[Room]:
        return [r for r in self.rooms if r.type == t]

    def neighbours(self, rid: int, connected_only: bool = False) -> List[int]:
        out = []
        for e in self.edges:
            if connected_only and not e.connected:
                continue
            if e.u == rid:
                out.append(e.v)
            elif e.v == rid:
                out.append(e.u)
        return out


# --------------------------------------------------------------------------- helpers
def to_polygon(points) -> Tuple[Polygon, bool]:
    """3D point list [[x, y, z], ...] -> shapely Polygon (xy only). Second value: was it valid?"""
    xy = [(float(p[0]), float(p[1])) for p in points]
    if len(xy) < 3:
        raise ModResError(f"polygon with {len(xy)} points")
    poly = Polygon(xy)
    if poly.is_valid:
        return poly, True
    fixed = poly.buffer(0)
    if fixed.is_empty or fixed.geom_type != "Polygon":
        raise ModResError("polygon invalid and not repairable with buffer(0)")
    return fixed, False


def centroid_distance(plan: Plan, u: int, v: int) -> float:
    """Default edge length: centroid to centroid, in metres.
    TEAM DECISION PENDING: swap for a door-to-door path length by passing another function
    to plan_to_tables(edge_length_fn=...)."""
    a, b = plan.rooms[u], plan.rooms[v]
    return math.hypot(a.cx - b.cx, a.cy - b.cy)


# --------------------------------------------------------------------------- parsing
def parse_adjacency(data: dict) -> Tuple[dict, List[Room], List[Edge], int, List[str]]:
    for k in ("properties", "vertices", "edges"):
        if k not in data:
            raise ModResError(f"adjacency json has no '{k}'")
    notes: List[str] = []

    key_to_rid: Dict[str, int] = {}
    rooms: List[Room] = []
    for rid, (key, vtx) in enumerate(data["vertices"].items()):
        poly, valid = to_polygon(vtx["geometry"])
        rooms.append(Room(
            rid=rid, key=key, type=int(vtx["type"]),
            area=float(vtx["area"]), perimeter=float(vtx["perimeter"]),
            compactness=float(vtx["compactness"]),
            cx=float(vtx["x"]), cy=float(vtx["y"]),
            polygon=poly, polygon_valid=valid))
        key_to_rid[key] = rid

    edges: Dict[Tuple[int, int], Edge] = {}
    raw_edges = list(data["edges"].values())
    for e in raw_edges:
        if e["source"] not in key_to_rid or e["target"] not in key_to_rid:
            notes.append(f"edge_unknown_endpoint:{e['source']}->{e['target']}")
            continue
        a, b = key_to_rid[e["source"]], key_to_rid[e["target"]]
        if a == b:
            notes.append(f"edge_self_loop:{e['source']}")
            continue
        u, v = min(a, b), max(a, b)
        t = int(e["edge_type"])
        if (u, v) in edges:
            if edges[(u, v)].type != t:
                notes.append(f"edge_conflict:{u}-{v}:{edges[(u, v)].type}vs{t}")
            else:
                notes.append(f"edge_duplicate:{u}-{v}")
            continue
        edges[(u, v)] = Edge(u, v, t)
    return data["properties"], rooms, list(edges.values()), len(raw_edges), notes


def find_entrance(rooms: List[Room], front_polys: List[Polygon], tol: float) -> Entrance:
    if not front_polys:
        return Entrance(None, "no_front_door", float("nan"), [], 0)
    # use the first front door; the count is reported so multi-door plans are visible
    fd = front_polys[0]
    dists = [(fd.distance(r.polygon), r.rid) for r in rooms]
    dists.sort()
    near = [rid for d, rid in dists if d <= tol]
    if not near:
        return Entrance(None, "not_found", dists[0][0], [], len(front_polys))
    status = "ok" if len(near) == 1 else "ambiguous"
    return Entrance(near[0], status, dists[0][0], near, len(front_polys))


def match_doors(rooms: List[Room], edges: List[Edge], door_polys: List[Polygon], tol: float) -> List[Door]:
    doorway_pairs = {(e.u, e.v) for e in edges if e.type == 0}
    out: List[Door] = []
    for i, dp in enumerate(door_polys):
        near = [r.rid for r in rooms if dp.distance(r.polygon) <= tol]
        pairs = [(a, b) for a in near for b in near if a < b and (a, b) in doorway_pairs]
        if len(pairs) == 1:
            out.append(Door(i, dp, pairs[0], "ok", near))
        elif not pairs:
            out.append(Door(i, dp, None, "no_doorway_pair", near))
        else:
            out.append(Door(i, dp, None, "ambiguous", near))
    return out


def load_geometry_parts(path: Path) -> Tuple[Dict[str, List[Polygon]], List[str]]:
    """Return {key: [Polygon, ...]} for rooms, 'door', 'front_door', plus notes."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    notes: List[str] = []
    parts: Dict[str, List[Polygon]] = {}
    for key in list(GEOMETRY_ROOM_KEYS) + ["door", "front_door"]:
        polys = []
        for pts in data.get(key, []):
            try:
                p, valid = to_polygon(pts)
                if not valid:
                    notes.append(f"geometry_polygon_repaired:{key}")
                polys.append(p)
            except ModResError as ex:
                notes.append(f"geometry_polygon_skipped:{key}:{ex}")
        parts[key] = polys
    return parts, notes


def load_plan(i: int, root: Path | str = DEFAULT_ROOT, touch_tol: float = DEFAULT_TOUCH_TOL) -> Plan:
    paths = ModResPaths(root)
    try:
        with open(paths.adjacency_json(i), "r", encoding="utf-8") as f:
            adj = json.load(f)
    except FileNotFoundError as ex:
        raise ModResError(f"missing adjacency file for plan {i}") from ex
    props, rooms, edges, raw_n, notes = parse_adjacency(adj)

    try:
        geom, gnotes = load_geometry_parts(paths.geometry_json(i))
    except FileNotFoundError as ex:
        raise ModResError(f"missing geometry file for plan {i}") from ex
    notes += gnotes

    entrance = find_entrance(rooms, geom["front_door"], touch_tol)
    doors = match_doors(rooms, edges, geom["door"], touch_tol)
    return Plan(i, props, rooms, edges, entrance, doors, raw_n, notes)


def load_geometry_for_check(i: int, root: Path | str = DEFAULT_ROOT) -> Dict[str, List[Polygon]]:
    """Room polygons from the geometry file (used by the validator to cross-check IDs)."""
    parts, _ = load_geometry_parts(ModResPaths(root).geometry_json(i))
    return parts


# --------------------------------------------------------------------------- tables
def plan_to_tables(plan: Plan, edge_length_fn: Callable[[Plan, int, int], float] = centroid_distance):
    """Plain-dict rows for building DataFrames / PyG tensors later.
    edges are stored once (u < v); add the reverse direction when building a GNN graph."""
    ent = plan.entrance.room
    nodes = [dict(plan_id=plan.plan_id, rid=r.rid, type=r.type, type_name=r.type_name,
                  area=r.area, perimeter=r.perimeter, compactness=r.compactness,
                  cx=r.cx, cy=r.cy, is_entrance=int(r.rid == ent)) for r in plan.rooms]
    edges = [dict(plan_id=plan.plan_id, u=e.u, v=e.v, type=e.type, type_name=e.type_name,
                  connected=int(e.connected), length=edge_length_fn(plan, e.u, e.v)) for e in plan.edges]
    graph = dict(plan_id=plan.plan_id, **{k: v for k, v in plan.props.items()})
    return nodes, edges, graph


def iter_plans(ids: Iterable[int], root: Path | str = DEFAULT_ROOT):
    for i in ids:
        yield load_plan(i, root)