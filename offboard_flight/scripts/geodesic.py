#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Geodesic cost-to-go: how far the goal really is, around what is in the way.

WHAT THIS REPLACES
PlanarMPPI's goal terms use ||p - goal||, the straight-line distance
(planar_mppi.py:202). That is a greedy potential: it points at the goal
regardless of what is between them. Pressed against a wall with the goal
behind it, every direction increases the distance, so the planner sits there.
Measured on `detour`: with the Euclidean cost the vehicle never reaches the
goal in 40 s on any of three seeds.

A geodesic field fixes that at the source rather than patching it. Distance is
measured ALONG traversable space, so:

    * moving sideways toward a gap DECREASES the cost, because the path
      through the gap is genuinely shorter
    * there is no local minimum to escape from -- the field is monotone along
      every shortest path by construction
    * it has no tuning parameters, unlike an exploration bonus

WHAT COUNTS AS TRAVERSABLE, AND WHY UNKNOWN IS OPTIMISTIC HERE
The field is built over cells whose clearance from a MEASURED obstacle is at
least r_safe, and unknown cells are treated as passable.

That looks unsafe and is not, because this is a GUIDANCE field and not a gate.
Nothing is ever flown because the field says so: `PlanarSafetyValidator` still
rejects any trajectory entering unknown space, exactly as before. Optimism here
is what makes the vehicle willing to head toward a gap it has not yet seen
through -- the same "free-space assumption" classical frontier planners use.
Pessimism would make the field infinite everywhere until the whole corridor had
been observed, which is the deadlock this is meant to remove.

UNREACHABLE CELLS
A cell with no traversable path to the goal gets `d_geo = inf`. Handing that to
MPPI would make every sample's cost inf and the weights undefined, so `query`
falls back to `reach_penalty + euclidean` there: still finite, still ordered,
and always worse than any genuinely reachable cell.
"""

import numpy as np

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.ndimage import distance_transform_edt

# 8-connectivity. Diagonals cost sqrt(2) cells, so the field approximates true
# Euclidean geodesic distance to about 8% -- ample for a cost gradient.
_NB = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
       (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)),
       (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))]


class CostToGo(object):
    """Geodesic distance to the goal over the traversable set."""

    def __init__(self, occ, goal, r_safe, unknown_free=True,
                 reach_penalty=50.0):
        self.res = float(occ.res)
        self.origin = (float(occ.origin[0]), float(occ.origin[1]))
        self.goal = np.asarray(goal, dtype=np.float64).reshape(2)
        self.reach_penalty = float(reach_penalty)

        occupied = np.asarray(getattr(occ, "occupied", occ.unsafe), dtype=bool)
        unknown = np.asarray(getattr(occ, "unknown",
                                     np.zeros_like(occupied, dtype=bool)))
        H, W = occupied.shape
        self.H, self.W = H, W

        # Clearance from MEASURED obstacles only. occ.clearance() cannot be
        # used: it returns 0 inside unknown, which would make unknown
        # impassable and defeat the optimism this field depends on.
        d_occ = distance_transform_edt(~occupied) * self.res
        free = d_occ >= r_safe
        if not unknown_free:
            free &= ~unknown

        self.free = free
        self.field = self._solve(free)

    # ------------------------------------------------------------- solving

    def _solve(self, free):
        H, W = self.H, self.W
        idx = -np.ones((H, W), dtype=np.int64)
        ys, xs = np.nonzero(free)
        if ys.size == 0:
            return np.full((H, W), np.inf)
        idx[ys, xs] = np.arange(ys.size)

        rows, cols, vals = [], [], []
        for dy, dx, w in _NB:
            y2, x2 = ys + dy, xs + dx
            ok = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
            ok &= free[np.clip(y2, 0, H - 1), np.clip(x2, 0, W - 1)]
            rows.append(idx[ys[ok], xs[ok]])
            cols.append(idx[y2[ok], x2[ok]])
            vals.append(np.full(int(ok.sum()), w * self.res))
        G = coo_matrix((np.concatenate(vals),
                        (np.concatenate(rows), np.concatenate(cols))),
                       shape=(ys.size, ys.size)).tocsr()

        gi, gj = self.world_to_cell(self.goal[0], self.goal[1])
        src = idx[gj, gi] if (0 <= gi < W and 0 <= gj < H
                              and free[gj, gi]) else -1
        if src < 0:
            # The goal itself is not in the traversable set -- unobserved, or
            # inside the inflated obstacle. Seed from the traversable cell
            # nearest to it so the field still points the right way.
            src = int(idx[ys, xs][np.argmin((xs - gi) ** 2 + (ys - gj) ** 2)])

        d = dijkstra(G, directed=False, indices=src)
        field = np.full((H, W), np.inf)
        field[ys, xs] = d
        return field

    # ------------------------------------------------------------ querying

    def world_to_cell(self, x, y):
        ix = np.floor((np.asarray(x) - self.origin[0]) / self.res).astype(np.int64)
        iy = np.floor((np.asarray(y) - self.origin[1]) / self.res).astype(np.int64)
        return ix, iy

    def query(self, P):
        """Geodesic distance at each point. P (..., 2) -> (...)."""
        P = np.asarray(P, dtype=np.float64)
        ix, iy = self.world_to_cell(P[..., 0], P[..., 1])
        inb = (ix >= 0) & (ix < self.W) & (iy >= 0) & (iy < self.H)
        d = self.field[np.clip(iy, 0, self.H - 1), np.clip(ix, 0, self.W - 1)]

        # Unreachable and off-grid fall back to a penalised straight line, so
        # the cost stays finite and still orders samples sensibly.
        eu = np.linalg.norm(P - self.goal, axis=-1)
        bad = ~inb | ~np.isfinite(d)
        return np.where(bad, self.reach_penalty + eu, d)

    def describe(self):
        n = int(np.isfinite(self.field).sum())
        return ("CostToGo  %d/%d cells reachable (%.1f%%)   max %.2f m"
                % (n, self.field.size, 100.0 * n / self.field.size,
                   float(self.field[np.isfinite(self.field)].max())
                   if n else float("nan")))
