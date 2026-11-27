# Exploration simulator and its tables

How the robot explores a floor plan and what the models get to see.
Code: `simulator/explore.py`, `simulator/tables.py`.

## Words used

| Word | Meaning |
|---|---|
| step | Walking from one room to the next through a doorway or a passage. Walls are never crossed. "Steps from the entrance" = the fewest steps needed to reach a room. |
| visited room | The robot has been inside. Type and area known. |
| seen room | Not entered, but visible through an open passage of a visited room. Type known. |
| behind_door room | Behind a closed doorway of a visited or seen room. The robot knows a room is there but not its type. Task 1 predicts its type. Called `STUB` in `views.py` and `"stub"` in the tables. |
| hidden room | The robot does not know the room exists. Not in the view at all. |
| known room | Visited or seen. Only these count towards the 25 / 50 / 75 % targets. |

## How the robot explores (`simulator/explore.py`)

The simulator works on the room graph only and returns the `View` objects defined in `views.py`, so the
GNN, the priors and the tests use it unchanged.

### Exploration as a Markov decision process (MDP)

| Part | In this project | In the code |
|---|---|---|
| state | What the robot knows: the rooms it has visited (in order), the rooms it has seen, and the rooms it knows are behind a door. | `ExplorationState` |
| action | The next room to enter. Allowed: any room not yet visited that a visited room leads to through a doorway or passage. | `ExplorationEnv.actions` |
| transition | Entering the room makes it visited; looking around from the visited rooms updates seen and behind_door. Deterministic. | `ExplorationEnv.step` |
| policy | The rule that picks the action. Fixed rules, nothing is learned. | `POLICIES` |
| reward | None, because nothing is optimised. A learned policy (reinforcement learning) would add one. | – |

Policies:

| Policy | Rule |
|---|---|
| `bfs` (default, baseline) | Nearest room first (steps from the entrance), rooms with equal steps in a random seeded order. |
| `random` | Any allowed room, chosen at random. |
| `door_first` | Rooms behind a door before rooms already seen, then nearest first. |

The views in our results come from `bfs`. The others are there to check that the models do not depend
on how the robot explores.

An A* policy (shortest walk from where the robot stands) was tried and dropped: it walked about 22 % less
than `bfs` up to 75 %, but its walks repeat across seeds far more often (about 67 % duplicate views), and
walking distance plays no part in what the models see.

### What the robot learns after each visit (`look_around`)

- A passage from a visited room makes the next room **seen**.
- A doorway from a visited or seen room makes the next room **behind_door**.
- A wall shows nothing.
- A room visible through any passage counts as seen, even if it also sits behind a door.

### Snapshots

One walk per seed; a snapshot for each fraction p, so the 25 % view is inside the 50 % view, which is
inside the 75 % view. Three stop rules (`stop=`):

| Rule | Snapshot | Note |
|---|---|---|
| `nearest` (default) | the moment the share of known rooms is closest to p, below or above (a tie goes to below) | every walk gets every p, so every flat is in every bucket; at 25 % it is often above 25 %, because the front-door view alone is often a third of the flat |
| `at_most` | the last moment visited + seen rooms are at most `floor(p * n_rooms)`: never above p | no snapshot for p if the robot already knows more than that in the entrance room (about 65 % of walks at 25 %, mostly small flats) |
| `at_least` | the first moment visited + seen rooms reach `round(p * n_rooms)` | often goes above p (about 32 % on average at 25 %) |

Always report the actual share known (`observed_fraction` in the tables) next to p.

Views with no behind_door room are dropped, because there is nothing to predict.

### What goes into a view's edges

- Every edge between two known rooms (visited or seen), walls included.
- The doorway between a known room and a behind_door room.

Nothing else: the robot has never been inside a behind_door or hidden room.

### No leaks

Policies may use the true steps from the entrance to choose a room; that only shapes the walk. The
hops feature is computed in `views.py` from the view's own edges, so nothing about the unexplored part
reaches the model.

### Try it

```
python -m simulator.explore                                                   # a synthetic plan
python -m simulator.explore --plan 210 --root "D:\ModRes\dataset" --seed 1    # a real plan
python -m simulator.explore --plan 210 --root "D:\ModRes\dataset" --policy random --stop at_least
```

Prints one line per visit: the room entered, what is seen, what is behind a door, the known count, and
where each snapshot is taken.

## The views as tables (`simulator/tables.py`)

Writes the simulator's views as tables, to look at them and count them. It uses
the same simulator, the same plans, the same train / val / test split and the same walk seeds as
`gnn/train.py`, so each table row describes a view the GNN actually trains or is tested on. Nothing here
is read by the GNN.

```
python -m simulator.tables --source modres --root "D:\ModRes\dataset" --ids validation_out\benchmark_plan_ids.txt
python -m simulator.tables --source synthetic --plans 300        # quick check, numbers mean nothing
```

Add `--limit 300` for a random sample, `--policy random` (or `door_first`) for another policy, and
`--stop at_most` or `--stop at_least` for another stop rule. `snapshots.parquet` also records `stop`.

Output (default folder `sim_tables/`), one row per ...

| File | One row per | Columns |
|---|---|---|
| `snapshots.parquet` | view | plan, split, p, walk, seed, counts, observed fraction, visit order, `is_duplicate` (same visited rooms as an earlier walk of this plan at this p) |
| `nodes.parquet` | room in a view | state (visited / seen / stub), type (-1 for stubs), area (visited and seen rooms; NaN for stubs), is_entrance, hops |
| `edges.parquet` | edge in a view | u, v, edge_type |
| `labels_stub.parquet` | behind_door room | **answers for Task 1**: its true type |
| `labels_remaining.parquet` | view | **answers**: unknown rooms per type (behind_door + hidden) |
| `task2_candidates.parquet` | true edge touching an unknown room | label 1 if walkable; `u_type` / `v_type` are the rooms' true types (Task 2 gives all rooms) |
| `sim_summary.txt` | – | counts per split and p, observed fraction, label distribution |

Two differences from what the GNN's features contain, both on purpose:

- area is given here for seen rooms too; the GNN features give area for visited rooms only;
- duplicates are kept and marked, because the GNN trains on them too (it does not remove them).

## Comparing the methods (`simulator/compare.py`)

Runs every policy with every stop rule on the same plans and seeds, and summarises per method, rule and p:
coverage (share of walks that give a view), share of rooms known (mean, max, share above p), visits,
rooms behind a door and hidden rooms per view, kitchen share of the answers, and duplicates.

```
python -m simulator.compare --root "D:\ModRes\dataset" --ids validation_out\benchmark_plan_ids.txt
```

Writes `sim_compare/walks.parquet` (one row per plan, method, rule, seed and p) and `summary.csv` / `summary.txt`.
