"""Geodesic cost-to-go field over the occupancy grid.

Replaces the Euclidean ``||p - goal||`` term in the MPPI reward with a
shortest-path-around-obstacles distance. Euclidean distance creates a local
minimum directly behind a head-on obstacle (the planner is pulled straight into
it and stalls); the geodesic cost-to-go is *high* behind an obstacle and only
decreases along a collision-free route, so the planner is naturally guided
around it. This is the standard fix used by cluttered-navigation MPPI stacks
(e.g. Nav2's path/cost critics, log-MPPI's grid integration).

The field is computed by a Dijkstra wavefront from the goal cell over the free
cells of the (tube-inflated) grid -- the grid is tiny (~GRID_W x GRID_H), so this
is a sub-millisecond recompute per plan tick.
"""
import heapq
import numpy as np

# 8-connected neighbours: (dy, dx, step-length-in-cells)
_NBRS = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, 1.4142135), (-1, 1, 1.4142135),
         (1, -1, 1.4142135), (1, 1, 1.4142135))


class CostToGoField:
    """Cached 2D cost-to-go array (meters) with a vectorized world-xy query."""

    def __init__(self, field, cell, xorigin, yorigin):
        self.field = field
        self.cell = float(cell)
        self.xo = float(xorigin)
        self.yo = float(yorigin)
        self.H, self.W = field.shape

    def query(self, X):
        """X: (..., >=2) world xy -> cost-to-go (...,) in meters (nearest cell)."""
        gx = np.floor((X[..., 0] - self.xo) / self.cell).astype(np.int32)
        gy = np.floor((X[..., 1] - self.yo) / self.cell).astype(np.int32)
        gx = np.clip(gx, 0, self.W - 1)
        gy = np.clip(gy, 0, self.H - 1)
        return self.field[gy, gx]


def compute_cost_to_go(occ, goal_xy, robot_radius):
    """Build a CostToGoField from a GridObstacleMap.

    Free cells get the 8-connected Dijkstra geodesic distance to the goal,
    routing around obstacles inflated by ``robot_radius``. Obstacle / unreachable
    cells fall back to the Euclidean distance to the goal so the field always has
    a gradient (those cells are hard-rejected by collision checking anyway, so the
    fallback only matters as a tie-breaking pull, never as a navigable route).
    """
    L = occ.L
    cell = float(occ.cell)
    H, W = L.shape

    # Cell-center world coordinates.
    ys, xs = np.mgrid[0:H, 0:W]
    wx = occ.xorigin + (xs + 0.5) * cell
    wy = occ.yorigin + (ys + 0.5) * cell

    # Obstacle mask = cells within robot_radius of an occupied cell (EDT-based,
    # same tightening the planner validates against).
    edt_cells = occ._ensure_edt()
    blocked = (edt_cells * cell) < float(robot_radius)
    # If the arena boundary is a known wall, also block cells within robot_radius
    # of the 5x7 edges so the geodesic never routes through the wall.
    if getattr(occ, 'wall_bounds', False):
        xmin, ymin, xmax, ymax = occ._world_bounds
        d_wall = np.minimum.reduce([wx - xmin, xmax - wx, wy - ymin, ymax - wy])
        blocked = blocked | (d_wall < float(robot_radius))

    # Euclidean base field (meters), evaluated at cell centers.
    field = np.sqrt((wx - goal_xy[0]) ** 2
                    + (wy - goal_xy[1]) ** 2).astype(np.float32)

    gx = int(np.floor((goal_xy[0] - occ.xorigin) / cell))
    gy = int(np.floor((goal_xy[1] - occ.yorigin) / cell))
    gx = int(np.clip(gx, 0, W - 1))
    gy = int(np.clip(gy, 0, H - 1))

    if not blocked[gy, gx]:
        dist = np.full((H, W), np.inf, dtype=np.float32)
        dist[gy, gx] = 0.0
        pq = [(0.0, gy, gx)]
        while pq:
            d, cy, cx = heapq.heappop(pq)
            if d > dist[cy, cx]:
                continue
            for dy, dx, step in _NBRS:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < H and 0 <= nx < W and not blocked[ny, nx]:
                    nd = d + step * cell
                    if nd < dist[ny, nx]:
                        dist[ny, nx] = nd
                        heapq.heappush(pq, (nd, ny, nx))
        reached = np.isfinite(dist)
        field[reached] = dist[reached]

    return CostToGoField(field, cell, occ.xorigin, occ.yorigin)
