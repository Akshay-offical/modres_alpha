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
- The GNN pipeline runs end to end, but on placeholder views. The real robot
  simulator is still being written.
- The LLM part hasn't started.

## Layout

    data/         read the ModRes plans and check them
    simulator/    turn a plan into partial views at 25/50/75 % explored (views.py is a placeholder)
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

Training on ModRes (add `--limit 300` for a quick test):

    python -m gnn.train --source modres --root "D:\ModRes\dataset" --ids validation_out\benchmark_plan_ids.txt --no-fraction-feature

Results go to `results_gnn/results.csv`, one row per model, explored fraction and
calibration setting. For repeated runs change `--seed`; `--split-seed` keeps the
train/val/test split the same.

## Notes on the data

- We read everything from the JSON files. The CSV export is wrong in about 500 plans
  (rooms with area 0 and the wrong type, extra nodes inside doorways).
- ModRes' own train/val/test masks are per node and would leak within a plan, so we
  split by plan (70/15/15).
- The front door isn't in the graph. We find the entrance from the front-door polygon
  in the geometry file. It is always the living room.
- Closed doors come from the doorway edges in the graph, not from the door polygons.
  Many door polygons lead to spaces that aren't rooms in the graph.
- Edge length is centre to centre for now, using the stored room position (the average
  of the polygon's corners).

## Open questions

- How the robot picks the next room, whether rooms behind an open passage count as
  seen, and whether "25 % explored" means rooms or floor area.
- Whether the models may know how much of the flat was explored (`--no-fraction-feature`).
- About 300 doors sit between rooms the graph calls "wall". That could mean wrong
  labels for the hidden-door task.

## Team

Akshay Ram Murugan, Vishnu Kaivalya Gudapati, Prasanna Bhat Shridhar.
Supervisors: Ashwin Nedungadi and Bharathi K. Nithyanantham.
