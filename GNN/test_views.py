"""python test_views.py  - checks on the placeholder views (run before trusting any result)."""
import copy

import numpy as np

from views import (DOORWAY, PASSAGE, SEEN, STUB, VISITED, WALL, make_views, synthetic_plans, view_to_data)

plans = synthetic_plans(300, seed=1)
n_views = 0
for p in plans:
    vs = make_views(p, seed=3)
    again = make_views(p, seed=3)
    # 1. deterministic for a fixed seed
    assert [v.nodes for v in vs] == [v.nodes for v in again], "not deterministic"
    prev = set()
    for v in vs:
        n_views += 1
        vis = {r for r, s in v.state.items() if s == VISITED}
        # 2. nested: the 25% visited set is inside the 50% one, inside the 75% one
        assert prev <= vis, "views are not nested"
        prev = vis
        # 3. the robot only walked through doorways / passages: every visited room is reachable that way
        reach, q = {p.entrance}, [p.entrance]
        for a in q:
            for b in p.connected_neighbours(a):
                if b in vis and b not in reach:
                    reach.add(b); q.append(b)
        assert reach == vis, "visited rooms not connected through doors"
        # 4. every stub really is behind a doorway from a visited room, and has a truth label
        for r, s in v.state.items():
            if s == STUB:
                assert any(p.etype(a, r) == DOORWAY for a in vis), "stub without a doorway"
                assert r in v.truth
        # 5. wall edges only between rooms whose type the robot knows
        for u, w, t in v.edges:
            if t == WALL:
                assert v.state[u] != STUB and v.state[w] != STUB, "wall edge touches a stub"
        # 6. rooms not in the view are really absent
        assert set(v.nodes) == set(v.state)
        # 7. NO LEAKAGE: change the hidden type / area of every stub and unseen room -> features must not change
        hidden_p = copy.deepcopy(p)
        for r in range(p.n):
            if v.state.get(r) not in (VISITED, SEEN):
                hidden_p.types[r] = (p.types[r] + 1) % 4
                hidden_p.areas[r] = p.areas[r] * 3 + 1
        from views import _features
        X2 = _features(hidden_p, v)
        assert np.array_equal(v.features, X2), "features depend on a hidden room"
        # 8. stub rows: type unknown flag set, type one-hot empty, area zero
        d = view_to_data(v)
        sx = d.x[d.stub_mask]
        assert (sx[:, :4].sum(1) == 0).all() and (sx[:, 4] == 1).all() and (sx[:, 9] == 0).all()
        # 9. seen rooms do not reveal area
        assert (d.x[(d.x[:, 6] == 1), 9] == 0).all()
print(f"all checks passed on {n_views} views from {len(plans)} synthetic plans")
