#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Geodesic cost-to-go: how far the goal really is, around what is in the way.

WHAT THIS REPLACES
PlanarMPPI's goal terms use ||p - goal||, the straight-line distance
(planner/mppi.py:201). That is a greedy potential: it points at the goal
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
falls back to `REACH_PENALTY_M + euclidean` there: still finite, still ordered,
and always worse than any genuinely reachable cell.
"""

import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.ndimage import distance_transform_edt

# 8-connectivity. Diagonals cost sqrt(2) cells, so the field approximates true
# Euclidean geodesic distance to about 8% -- ample for a cost gradient.
_NB = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
       (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)),
       (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))]

# Cost [m] added to an unreachable or off-grid node, on top of its straight-line
# distance. It only has to dominate every REACHABLE cost, so that "no path" can
# never look cheaper than "a long way round"; beyond that its value does not
# matter, and nothing tunes it. This used to be a constructor argument no caller
# ever passed, which made it look tunable.
#
# WHY 50 CLEARS THE BAR, measured (12 s clutter flight, 240 solves, 718k queried
# nodes): the largest FINITE geodesic distance the field ever held was 6.48 m and
# the map diagonal is 8.74 m, so any reachable node costs at most ~15 m. The
# fallback fires on 0.27% of nodes overall (max 5.3% in a single solve), i.e. it
# is a real path, not a dead branch.
REACH_PENALTY_M = 50.0


class MapCostCache(object):
    """One map's exact occupied EDT, passability, graph and distance field.

    Owned by a planner, rather than by the occupancy adapter: both frontier
    costs and geodesic guidance use the same SciPy EDT. Contents are compared
    against a copy because adapters can replace OR mutate their map arrays.
    This keeps one entry at each stage; no map history accumulates.
    """

    def __init__(self):
        self._occupied = None
        self._distance_res = None
        self._distance = None
        self._r_pass = None
        self._passable = None
        self._graph_free = None
        self._graph_res = None
        self._graph = None
        self._idx = None
        self._ys = None
        self._xs = None
        self._source = None
        self._field = None

    def passable(self, occupied, res, r_pass):
        """Exactly ``distance_transform_edt(~occupied) * res >= r_pass``."""
        occupied = np.asarray(occupied)
        res = float(res)
        if (self._occupied is None or self._distance_res != res
                or self._occupied.dtype != occupied.dtype
                or not np.array_equal(self._occupied, occupied)):
            self._occupied = occupied.copy()
            self._distance_res = res
            self._distance = distance_transform_edt(~self._occupied) * res
            self._passable = None
        if self._passable is None or self._r_pass != r_pass:
            self._r_pass = r_pass
            self._passable = self._distance >= r_pass
        return self._passable


def _grid_graph(free, res):
    """Build the same 8-connected graph directly as sorted CSR.

    Row-major cell indices make row-major neighbour offsets sorted already.
    The padded index grid supplies -1 for out-of-bounds and blocked cells,
    avoiding coordinate clipping, a COO allocation and conversion to CSR.
    """
    H, W = free.shape
    ys, xs = np.nonzero(free)
    n = ys.size
    # SciPy uses int32 indices at these grid sizes; construct them directly
    # instead of allocating int64 buffers only to downcast them in csr_matrix.
    index_dtype = np.int32 if n <= np.iinfo(np.int32).max // len(_NB) else np.int64
    padded = np.full((H + 2, W + 2), -1, dtype=index_dtype)
    idx = padded[1:-1, 1:-1]
    idx[free] = np.arange(n, dtype=index_dtype)
    neighbours = np.empty((n, len(_NB)), dtype=index_dtype)
    weights = []
    for j, (dy, dx, w) in enumerate(sorted(_NB)):
        neighbours[:, j] = padded[1 + dy:1 + dy + H,
                                   1 + dx:1 + dx + W][free]
        weights.append(w * res)
    present = neighbours >= 0
    indptr = np.empty(n + 1, dtype=index_dtype)
    indptr[0] = 0
    np.cumsum(present.sum(axis=1), out=indptr[1:])
    data = np.broadcast_to(np.asarray(weights), neighbours.shape)[present]
    graph = csr_matrix((data, neighbours[present], indptr), shape=(n, n))
    return graph, idx, ys, xs


class CostToGo(object):
    """Geodesic distance to the goal over the traversable set."""

    def __init__(self, occ, goal, r_safe, unknown_free=True, cache=None):
        self.res = float(occ.res)
        self.origin = (float(occ.origin[0]), float(occ.origin[1]))
        self.goal = np.asarray(goal, dtype=np.float64).reshape(2).copy()
        self._cache = MapCostCache() if cache is None else cache

        occupied = np.asarray(getattr(occ, "occupied", occ.unsafe), dtype=bool)
        unknown = np.asarray(getattr(occ, "unknown",
                                     np.zeros_like(occupied, dtype=bool)))
        H, W = occupied.shape
        self.H, self.W = H, W

        # Clearance from MEASURED obstacles only. occ.clearance() cannot be
        # used: it returns 0 inside unknown, which would make unknown
        # impassable and defeat the optimism this field depends on.
        free = self._cache.passable(occupied, self.res, r_safe)
        if not unknown_free:
            free = free & ~unknown

        self.free = free
        self.field = self._solve(free)

    # ------------------------------------------------------------- solving

    def _solve(self, free):
        H, W = self.H, self.W
        cache = self._cache
        if (cache._graph_free is None or cache._graph_res != self.res
                or not np.array_equal(cache._graph_free, free)):
            cache._graph, cache._idx, cache._ys, cache._xs = _grid_graph(
                free, self.res)
            cache._graph_free = free.copy()
            cache._graph_res = self.res
            cache._field = None
        idx, ys, xs = cache._idx, cache._ys, cache._xs

        if ys.size == 0:
            if cache._field is None:
                cache._source = -1
                cache._field = np.full((H, W), np.inf)
            return cache._field

        gi, gj = self.world_to_cell(self.goal[0], self.goal[1])
        src = idx[gj, gi] if (0 <= gi < W and 0 <= gj < H
                              and free[gj, gi]) else -1
        if src < 0:
            # The goal itself is not in the traversable set -- unobserved, or
            # inside the inflated obstacle. Seed from the traversable cell
            # nearest to it so the field still points the right way.
            src = int(idx[ys, xs][np.argmin((xs - gi) ** 2 + (ys - gj) ** 2)])

        if cache._field is not None and cache._source == src:
            return cache._field
        field = np.full((H, W), np.inf)
        # Every edge already has its reverse with the same weight.
        # directed=False would transpose and traverse both copies again.
        field[ys, xs] = dijkstra(cache._graph, directed=True, indices=src)
        cache._source = src
        cache._field = field
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
        return np.where(bad, REACH_PENALTY_M + eu, d)

    def describe(self):
        n = int(np.isfinite(self.field).sum())
        return ("CostToGo  %d/%d cells reachable (%.1f%%)   max %.2f m"
                % (n, self.field.size, 100.0 * n / self.field.size,
                   float(self.field[np.isfinite(self.field)].max())
                   if n else float("nan")))
