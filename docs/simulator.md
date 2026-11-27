# Exploration simulator, tables and rules baselines

How the robot explores a floor plan, what the models get to see, and the simple baselines every model
has to beat. Code: `simulator/explore.py`, `simulator/tables.py`, `baselines/rules.py`.

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

### What the robot learns after each visit (`look_around`)

- A passage from a visited room makes the next room **seen**.
- A doorway from a visited or seen room makes the next room **behind_door**.
- A wall shows nothing.
- A room visible through any passage counts as seen, even if it also sits behind a door.

### Snapshots

For each fraction p, the walk stops as soon as visited + seen rooms reach `round(p * n_rooms)`
(rounded half up, kept between 1 and n_rooms - 1), and a snapshot is taken. The same walk is used for
every p, so the 25 % view is inside the 50 % view, which is inside the 75 % view. Views with no
behind_door room are dropped, because there is nothing to predict.

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
python -m simulator.explore --plan 210 --root "D:\ModRes\dataset" --policy door_first
```

Prints one line per visit: the room entered, what is seen, what is behind a door, the known count, and
where each snapshot is taken.

## The views as tables (`simulator/tables.py`)

Writes the simulator's views as tables, to look at them, count them and run the rules baselines. It uses
the same simulator, the same plans, the same train / val / test split and the same walk seeds as
`gnn/train.py`, so each table row describes a view the GNN actually trains or is tested on. Nothing here
is read by the GNN.

```
python -m simulator.tables --source modres --root "D:\ModRes\dataset" --ids validation_out\benchmark_plan_ids.txt
python -m simulator.tables --source synthetic --plans 300        # quick check, numbers mean nothing
```

Add `--limit 300` for a random sample, `--policy random` or `--policy door_first` for another policy.

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

## Rules baselines (`baselines/rules.py`)

Simple rules, no machine learning. Run `simulator/tables.py` first; the scores can then be compared with
the GNN's, because they use the same plans, split and walks. Every table is counted on the **train**
split only and scored on **val** (or **test** with `--eval test`, only for the final evaluation). Each
variant adds one rule, so the result table doubles as an ablation. The tables keep duplicate views, so
the rules see the same data as the GNN.

```
python -m baselines.rules                  # score on val
python -m baselines.rules --eval test      # final evaluation only
```

Output (default `results_rules/`): `results_<split>.csv`, `summary_<split>.txt`, `task1_markov_table.csv`.

### Task 1: what type is the room behind the door?

| Method | Rule |
|---|---|
| T1-majority | Always the most common training label. |
| T1-markov | P(type \| type of a known room whose doorway leads to it), a Markov table. A room with doorways from several known rooms multiplies their rows. |
| T1-full | T1-markov, plus: a plan has exactly one kitchen, so if a kitchen is already known, "kitchen" is ruled out. |

Metric: macro-F1 over bedroom / bathroom / kitchen (main), accuracy, per-class F1. "living" is left out
of the macro average: it is the entrance, so it is almost never behind a door and its F1 is 0 for every
method, which would only add noise. The report must say so.

### Task 2: are two rooms connected or only separated by a wall?

All rooms are given in Task 2; only the connections touching unknown rooms are hidden.

| Method | Rule |
|---|---|
| T2-pair | P(connected \| unordered pair of room types). |
| T2-pair+reach | T2-pair, plus: every room is reachable, so an unknown room with exactly one candidate pair must be connected through it (p = 1). |

Metric: AUROC (main), F1 at threshold 0.5, accuracy.
