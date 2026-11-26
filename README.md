# modres_alpha

Team Alpha's code for the LeReSI project in Applications of Deep Learning
(University of Rostock, winter semester 2026/27).

A robot exploring a flat only sees part of it. We try to predict the rest: what kind
of room is behind each closed door, which rooms are connected, and how many rooms
are still unseen. Every prediction comes with a probability, and we check both how
often a model is right and whether its confidence can be trusted. We compare a
simple frequency prior, several GNNs and, later, an LLM.

Data: ModRes (Shelley et al., 2026), 8,431 residential floor plans stored as room graphs.

## Where things stand

- Loading and checking the data is done. 8,419 plans are usable, and 8,098 of them
  can be walked completely from the front door. We use those 8,098.
- The exploration simulator is done (`simulator/explore.py`) and the GNN pipeline
  reads its views.
- The LLM part hasn't started.

## Layout

    data/         read the ModRes plans and check them
    simulator/    turn a plan into partial views at 25/50/75 % explored (explore.py walks, views.py holds the View format)
    baselines/    frequency priors
    gnn/          GNN models and the training script
    evaluation/   accuracy, NLL, Brier score, ECE, temperature scaling
    tests/        checks that views don't leak hidden information
    docs/         notes

Run everything from the repository root with `python -m ...`.

## Setup

Python 3.10 or newer. We keep the virtual environment one folder above the repo.

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt

No GPU needed. The graphs have 4 to 13 rooms and train fine on a laptop CPU.

Download ModRes separately. The examples below assume its `dataset` folder (the
one containing `Graphs` and `Geometry`) is at `D:\ModRes\dataset`.

## Running

Check the data. This writes `validation_out/`, including `benchmark_plan_ids.txt`:

    python -m data.validate_modres --root "D:\ModRes\dataset"

Check that the views don't leak anything:

    python -m tests.test_views

Quick run without the dataset (synthetic plans, the numbers mean nothing):

    python -m gnn.train --source synthetic --plans 300 --epochs 15

Training on ModRes (add `--limit 300` for a quick test on a random sample of 300 plans):

    python -m gnn.train --source modres --root "D:\ModRes\dataset" --ids validation_out\benchmark_plan_ids.txt --no-fraction-feature

Results go to `results_gnn/results.csv`, one row per model, explored fraction,
calibration setting and flat size (number of bedrooms). For repeated runs change
`--seed`; `--split-seed` keeps the train/val/test split the same.

## Notes on the data

- We read everything from the JSON files. The CSV export is wrong in about 500 plans
  (rooms with area 0 and the wrong type, extra nodes inside doorways).
- ModRes' own train/val/test masks are per node and would leak within a plan, so we
  split by plan (70/15/15).
- The front door isn't in the graph. We find the entrance from the front-door polygon
  in the geometry file. It is always the living room.
- Closed doors come from the doorway edges in the graph, not from the door polygons.
  Many door polygons lead to spaces that aren't rooms in the graph.
- Plan ids are sorted by flat size: ids below 1000 are mostly 1-bedroom flats, the
  last ones have 3 or more bedrooms. Never use the first N plans as a sample.
- Edge length is centre to centre for now, using the stored room position (the average
  of the polygon's corners).

## How the simulator explores

- The robot visits rooms nearest to the front door first, counting steps through doorways
  and passages (never walls). Rooms the same number of steps away are visited in a random
  order fixed by the seed.
- Rooms visible through an open passage of a visited room count as seen. Rooms behind a
  closed doorway of a visited or seen room are stubs, the rooms Task 1 predicts.
- "25 % explored" means 25 % of the rooms (visited + seen), not floor area. The robot stops
  the first time it reaches that, so on average it knows a bit more (about 31 % at 25 %).
- Each seed is one walk, and the 25 / 50 / 75 % views are snapshots of it, so they are nested.
- The walk is written as a Markov decision process (`ExplorationEnv` in `simulator/explore.py`):
  the state is what the robot knows, an action is the next room to enter, and a policy picks it.
  Nearest-first (`bfs`) is the baseline and gives all our views. `random` and `door_first` are
  there to test whether the models depend on how the robot explores
  (`make_views(plan, policy="random")`). Nothing is learned yet; a learned policy would need a reward.

## Open questions

- Whether the models may know how much of the flat was explored. For now use
  `--no-fraction-feature` for all real results: together with the number of visited rooms,
  the fraction gives away the total number of rooms, which a real robot wouldn't know.
- About 300 doors sit between rooms the graph calls "wall". That could mean wrong
  labels for the hidden-door task.

## Team

Akshay Ram Murugan, Vishnu Kaivalya Gudapati, Prasanna Bhat Shridhar.
Supervisors: Ashwin Nedungadi and Bharathi K. Nithyanantham.
