"""
explore.py - the exploration simulator: turns a plan into the partial views the models read.

The walk is an MDP (ExplorationEnv) with swappable policies; "bfs" (nearest first from the entrance)
is the default. Words, rules and how to run it: docs/simulator.md.
"""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Tuple

from simulator.views import DOORWAY, PASSAGE, SEEN, STUB, VISITED, WALL, PlanGraph, View, _features

FRACTIONS = (0.25, 0.5, 0.75)                     # snapshot targets: share of rooms known
TYPE_NAMES = {0: "living", 1: "bedroom", 2: "bathroom", 3: "kitchen"}


# --------------------------------------------------------------------------- plan helpers
def walkable_neighbours(plan: PlanGraph) -> Dict[int, List[int]]:
    """For each room, the rooms you can walk to through a doorway or passage (sorted, so the result
    does not depend on the order the edges are stored in)."""
    neighbours: Dict[int, List[int]] = {room: [] for room in range(plan.n)}
    for (u, v), kind in plan.edges.items():
        if kind in (DOORWAY, PASSAGE):
            neighbours[u].append(v)
            neighbours[v].append(u)
    for room in neighbours:
        neighbours[room].sort()
    return neighbours


def count_steps(entrance: int, neighbours: Dict[int, List[int]]) -> Dict[int, int]:
    """Fewest steps from the entrance to every reachable room (breadth-first search).

    The queue is first-in, first-out, so every room 1 step away is handled before any room 2 steps
    away, and the first time a room is reached is always by the shortest route.
    """
    steps = {entrance: 0}
    queue = deque([entrance])
    while queue:
        room = queue.popleft()
        for next_room in neighbours[room]:
            if next_room not in steps:
                steps[next_room] = steps[room] + 1
                queue.append(next_room)
    return steps


def rooms_to_know(p: float, n_rooms: int) -> int:
    """How many rooms the robot should know (visited + seen) for fraction p."""
    target = math.floor(p * n_rooms + 0.5)        # round half up (Python round() sends 2.5 to 2)
    return min(max(target, 1), n_rooms - 1)       # 0 = knows nothing, n = nothing left to predict


def look_around(plan: PlanGraph, visited: set) -> Tuple[set, set]:
    """What the robot can tell from the rooms it has visited.

    1. Through an open passage of a visited room it sees into the next room (seen).
    2. It notices the closed doorways of every room it knows - visited rooms, and seen rooms it is
       looking into. The room behind such a doorway is known to exist, nothing more (behind_door).
    A room it can also see through some passage counts as seen.
    """
    seen = set()
    for (u, v), kind in plan.edges.items():
        if kind != PASSAGE:
            continue
        for here, there in ((u, v), (v, u)):      # the robot may be on either end of the edge
            if here in visited and there not in visited:
                seen.add(there)

    known = visited | seen
    behind_door = set()
    for (u, v), kind in plan.edges.items():
        if kind != DOORWAY:                       # walls hide the room behind them completely
            continue
        for here, there in ((u, v), (v, u)):
            if here in known and there not in known:
                behind_door.add(there)
    return seen, behind_door


def edges_robot_knows(plan: PlanGraph, known: set, behind_door: set) -> List[Tuple[int, int, int]]:
    """Edges in the view as (u, v, kind) with u < v. Rules: docs/simulator.md."""
    view = []
    for (u, v), kind in sorted(plan.edges.items()):
        if u in known and v in known:
            view.append((u, v, kind))
        elif kind == DOORWAY and ((u in known and v in behind_door) or (v in known and u in behind_door)):
            view.append((u, v, kind))
    return view


# --------------------------------------------------------------------------- the MDP
@dataclass
class ExplorationState:
    """What the robot knows at one moment."""
    visited: List[int]                            # in the order they were entered; the last is where it stands
    seen: set = field(default_factory=set)
    behind_door: set = field(default_factory=set)

    @property
    def known(self) -> set:
        return set(self.visited) | self.seen


class ExplorationEnv:
    """One plan as an exploration environment: reset() gives the start state, actions() the rooms the
    robot may enter next, step() the state after entering one."""

    def __init__(self, plan: PlanGraph):
        self.plan = plan
        self.neighbours = walkable_neighbours(plan)
        self.steps = count_steps(plan.entrance, self.neighbours)   # true steps; policies may use them
        if len(self.steps) != plan.n:
            raise ValueError(f"plan {plan.plan_id} not fully walkable from the entrance - use benchmark plans only")

    def reset(self) -> ExplorationState:
        """The robot stands in the entrance room."""
        return self._observe([self.plan.entrance])

    def actions(self, state: ExplorationState) -> List[int]:
        """Rooms not yet visited that a visited room leads to through a doorway or passage."""
        visited = set(state.visited)
        return sorted({nxt for room in state.visited for nxt in self.neighbours[room] if nxt not in visited})

    def step(self, state: ExplorationState, room: int) -> ExplorationState:
        """Enter `room`."""
        if room not in self.actions(state):
            raise ValueError(f"room {room} cannot be entered from the visited rooms")
        return self._observe(state.visited + [room])

    def _observe(self, visited: List[int]) -> ExplorationState:
        seen, behind_door = look_around(self.plan, set(visited))
        return ExplorationState(visited=visited, seen=seen, behind_door=behind_door)


# --------------------------------------------------------------------------- policies
# A policy is made once per walk: make_policy(env, rng) returns choose(state) -> room.
Policy = Callable[[ExplorationState], int]


def bfs_policy(env: ExplorationEnv, rng: random.Random) -> Policy:
    """Nearest room first (steps from the entrance); rooms with equal steps in a random seeded order.

    The order is fixed at the start: shuffle all rooms, then sort by steps. Python's sort is stable,
    so rooms with equal steps keep their shuffled order. The next room in this order can always be
    entered, because a room one step nearer on its shortest route has already been visited.
    """
    rooms = list(env.steps)
    rng.shuffle(rooms)
    order = sorted(rooms, key=lambda room: env.steps[room])

    def choose(state: ExplorationState) -> int:
        visited = set(state.visited)
        return next(room for room in order if room not in visited)
    return choose


def random_policy(env: ExplorationEnv, rng: random.Random) -> Policy:
    """Any room the robot can enter next, chosen at random."""
    def choose(state: ExplorationState) -> int:
        return rng.choice(env.actions(state))
    return choose


def door_first_policy(env: ExplorationEnv, rng: random.Random) -> Policy:
    """Rooms behind a door before rooms already seen (opening a door teaches the robot more), then
    nearest first, then random."""
    def choose(state: ExplorationState) -> int:
        options = env.actions(state)
        rng.shuffle(options)
        return min(options, key=lambda room: (room in state.seen, env.steps[room]))
    return choose


POLICIES: Dict[str, Callable[[ExplorationEnv, random.Random], Policy]] = {
    "bfs": bfs_policy,
    "random": random_policy,
    "door_first": door_first_policy,
}
STOP_RULES = ("nearest", "at_most", "at_least")


# --------------------------------------------------------------------------- views
def build_view(plan: PlanGraph, state: ExplorationState, p: float) -> View:
    """Turn what the robot knows into a View (features included)."""
    labels = {room: VISITED for room in state.visited}
    labels.update({room: SEEN for room in state.seen})
    labels.update({room: STUB for room in state.behind_door})
    view = View(plan_id=plan.plan_id,
                fraction=p,
                nodes=sorted(labels),
                state=labels,
                edges=edges_robot_knows(plan, state.known, state.behind_door),
                truth={room: plan.types[room] for room in sorted(state.behind_door)},
                entrance=plan.entrance,
                n_visited=len(state.visited),
                n_rooms=plan.n)
    view.features = _features(plan, view)
    return view


def walk(plan: PlanGraph, seed: int = 0, policy: str = "bfs") -> Iterator[ExplorationState]:
    """One walk through the plan: the robot's state in the entrance, then after each visit, until every
    room has been visited. The caller may stop early."""
    env = ExplorationEnv(plan)
    choose = POLICIES[policy](env, random.Random(plan.plan_id * 1_000_003 + seed))
    state = env.reset()
    yield state
    while env.actions(state):
        state = env.step(state, choose(state))
        yield state


def rooms_allowed(p: float, n_rooms: int) -> int:
    """At most how many rooms the robot may know (visited + seen) for fraction p: p * n, rounded down."""
    return math.floor(p * n_rooms + 1e-9)        # + 1e-9: 0.75 * 4 must give 3, not 2.999...


def walk_snapshots(plan: PlanGraph, fractions=FRACTIONS, seed: int = 0, policy: str = "bfs",
                   stop: str = "nearest") -> Dict[float, ExplorationState]:
    """What the robot knows at each fraction p, in one walk. Includes snapshots with nothing behind a door.

    nearest  (default) the moment its share of known rooms is closest to p, below or above (a tie goes
             to below). Every walk gets every p, so every flat is in every bucket.
    at_most  the last moment it knows at most p of the rooms: never above p. A p is left out if the robot
             already knows more than that in the entrance room (most small flats at 25 %).
    at_least the first moment it knows at least round(p * n) rooms; often goes above p.
    """
    snapshots: Dict[float, ExplorationState] = {}
    if stop == "nearest":
        gaps: Dict[float, float] = {}
        for state in walk(plan, seed, policy):
            share = len(state.known) / plan.n
            for p in fractions:
                gap = abs(share - p)
                if p not in gaps or gap < gaps[p] - 1e-9:   # strictly closer; known only grows, so a tie keeps the lower
                    gaps[p], snapshots[p] = gap, state
            if share > max(fractions):                       # further states only get further away
                break
        return snapshots
    if stop == "at_most":
        limits = {p: rooms_allowed(p, plan.n) for p in fractions}
        for state in walk(plan, seed, policy):
            known = len(state.known)
            if known > max(limits.values()):          # known only grows: nothing later can fit
                break
            for p, limit in limits.items():
                if known <= limit:
                    snapshots[p] = state              # keep the latest state still within the limit
        return snapshots
    if stop != "at_least":
        raise ValueError(f"unknown stop rule {stop!r}; use one of {STOP_RULES}")
    targets = {p: rooms_to_know(p, plan.n) for p in fractions}
    for state in walk(plan, seed, policy):
        for p, target in targets.items():
            if p not in snapshots and len(state.known) >= target:
                snapshots[p] = state
        if len(snapshots) == len(targets):
            break
    return snapshots


def make_views(plan: PlanGraph, fractions=FRACTIONS, seed: int = 0, policy: str = "bfs",
               stop: str = "nearest") -> List[View]:
    """One walk through the plan with the given policy, and a View at each fraction.

    With "bfs", different seeds give different walks only where rooms tie on steps from the entrance.
    """
    snapshots = walk_snapshots(plan, fractions, seed, policy, stop)
    return [build_view(plan, snapshots[p], p) for p in fractions
            if p in snapshots and snapshots[p].behind_door]   # nothing to predict -> no view


# --------------------------------------------------------------------------- demo
def _demo():
    """python -m simulator.explore [--plan ID --root DATASET] [--seed N] [--policy NAME] [--stop RULE]

    Walks one plan step by step and prints what the robot knows. Without --plan it uses a synthetic plan.
    """
    import argparse
    from simulator.views import plan_from_loader, synthetic_plans

    ap = argparse.ArgumentParser(description="Show one exploration walk step by step.")
    ap.add_argument("--plan", type=int, default=None, help="ModRes plan id (needs --root)")
    ap.add_argument("--root", default=None, help="ModRes dataset folder (with Graphs and Geometry)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", choices=sorted(POLICIES), default="bfs")
    ap.add_argument("--stop", choices=STOP_RULES, default="nearest")
    a = ap.parse_args()

    if a.plan is None:
        plan = synthetic_plans(1, seed=a.seed)[0]
    else:
        from data import modres_loader as ml
        plan = plan_from_loader(ml.load_plan(a.plan, a.root or ml.DEFAULT_ROOT))

    kinds = {DOORWAY: "door", PASSAGE: "passage", WALL: "wall"}
    room = lambda r: f"{r}:{TYPE_NAMES[plan.types[r]]}"
    steps = ExplorationEnv(plan).steps
    rule = {"nearest": lambda p: f"closest to {p * plan.n:g}",
            "at_most": lambda p: f"at most {rooms_allowed(p, plan.n)}",
            "at_least": lambda p: f"at least {rooms_to_know(p, plan.n)}"}[a.stop]
    rule = {p: rule(p) for p in FRACTIONS}
    snapshots = walk_snapshots(plan, FRACTIONS, a.seed, a.policy, a.stop)

    print(f"plan {plan.plan_id}: {plan.n} rooms, entrance = room {room(plan.entrance)}")
    print("connections: " + ", ".join(f"{room(u)}-{room(v)} {kinds[t]}" for (u, v), t in sorted(plan.edges.items())))
    print("steps from the entrance: " + ", ".join(f"{room(r)}={s}" for r, s in sorted(steps.items(), key=lambda x: x[1])))
    print(f"snapshots ({a.stop}, rooms known): " + ", ".join(f"{int(p * 100)}% -> {r}" for p, r in rule.items()))
    print(f"policy: {a.policy}, seed: {a.seed}\n")

    for state in walk(plan, a.seed, a.policy):
        known = len(state.known)
        here = [f"{int(p * 100)}% snapshot" for p, s in snapshots.items() if s.visited == state.visited]
        print(f"visit {len(state.visited)}: entered {room(state.visited[-1]):12s} "
              f"seen [{', '.join(room(r) for r in sorted(state.seen))}]  "
              f"behind a door [{', '.join(room(r) for r in sorted(state.behind_door))}]  "
              f"known {known}/{plan.n}"
              + (f"  <- {', '.join(here)}" if here else ""))
    missing = [f"{int(p * 100)}%" for p in FRACTIONS if p not in snapshots]
    if missing:
        print(f"\nno snapshot for {', '.join(missing)}: the robot already knows more than that in the entrance room")


if __name__ == "__main__":
    _demo()
