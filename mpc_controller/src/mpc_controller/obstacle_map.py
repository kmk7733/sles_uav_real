"""Occupancy-grid obstacle map: the adapter the MPPI planner rolls out against.

The pasted simulation planner (``mppi.MPPIPlanner``) and the geodesic
``cost_to_go`` field both duck-type an ``occ`` object. In simulation that was a
synthetic map; here it is built from a live ``nav_msgs/OccupancyGrid`` (the ZED
depth -> ``/grid_map`` pipeline). This class provides exactly the surface those
two modules touch:

    collides(X, robot_radius)   -> (..., ) bool   [hard collision, _validate]
    near_obstacle(X, infl)      -> (..., ) float   [soft cost, _reward]
    _ensure_edt()               -> (H, W) float    [EDT in *cells*, cost_to_go]
    .L, .cell, .xorigin, .yorigin, ._world_bounds, .wall_bounds

Collision / proximity are computed from a Euclidean distance transform (EDT):
the distance from every free cell to the nearest occupied cell. A query point is
"in collision" when that distance (in meters) is below the robot radius, and
"near an obstacle" when it is below the inflation radius. This matches how the
simulation map behaved (body-radius inflation via EDT) so the planner's tuning
carries over unchanged.
"""
import numpy as np

try:
    from scipy.ndimage import distance_transform_edt as _edt
except Exception:  # pragma: no cover - scipy is expected on the Jetson
    _edt = None

try:
    import cv2 as _cv2  # cv2.distanceTransform is ~10x faster than scipy's EDT
except Exception:  # pragma: no cover
    _cv2 = None


def _edt_cells(occupied):
    """Distance (in cells) from every cell to the nearest occupied cell.
    Prefers cv2.distanceTransform (fast); falls back to scipy, then brute force."""
    if _cv2 is not None:
        # distanceTransform measures distance to the nearest ZERO pixel, so make
        # occupied cells 0 and free cells nonzero. DIST_MASK_5 is ~2% off exact
        # L2 -- negligible at 5 cm cells and much faster than the precise mask.
        src = np.where(occupied, 0, 1).astype(np.uint8)
        return _cv2.distanceTransform(src, _cv2.DIST_L2, 5).astype(np.float32)
    if _edt is not None:
        return _edt(~occupied).astype(np.float32)
    return _brute_edt(occupied)


class OccupancyGridMap:
    """Adapter over an occupancy grid for MPPI collision / proximity queries.

    Parameters
    ----------
    occupied : (H, W) bool ndarray
        True where a cell is an obstacle. Row 0 is the ``origin`` row (y = yorigin),
        matching ``nav_msgs/OccupancyGrid`` row-major layout.
    cell : float
        Cell size [m] (grid resolution).
    xorigin, yorigin : float
        World coordinate of the (0, 0) cell's lower-left corner.
    frame_id : str
        Frame the grid lives in (e.g. ``map``); carried for the node's convenience.
    oob_is_obstacle : bool
        If True, query points outside the grid extent count as collisions.
        If False (default) they are treated as free so the planner can steer
        toward a goal beyond the current map edge.
    wall_bounds : bool
        Passed through to ``cost_to_go``: treat the grid's rectangular boundary
        as a wall when building the geodesic field.
    """

    def __init__(self, occupied, cell, xorigin, yorigin,
                 frame_id="map", oob_is_obstacle=False, wall_bounds=False):
        self.L = np.ascontiguousarray(occupied, dtype=bool)
        self.cell = float(cell)
        self.xorigin = float(xorigin)
        self.yorigin = float(yorigin)
        self.frame_id = str(frame_id)
        self.oob_is_obstacle = bool(oob_is_obstacle)
        self.wall_bounds = bool(wall_bounds)
        self.H, self.W = self.L.shape
        self._world_bounds = (
            self.xorigin, self.yorigin,
            self.xorigin + self.W * self.cell,
            self.yorigin + self.H * self.cell,
        )
        self._edt_cells = None  # lazy EDT cache

    # -- classmethod builder ------------------------------------------------

    @classmethod
    def from_msg(cls, msg, occ_thresh=50, unknown_is_obstacle=False, **kwargs):
        """Build from a ``nav_msgs/OccupancyGrid``.

        Cell values: 0..100 = occupancy probability, -1 = unknown. A cell is an
        obstacle when its probability is >= ``occ_thresh``; unknown cells are
        obstacles only if ``unknown_is_obstacle`` is set.
        """
        info = msg.info
        data = np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width)
        occupied = data >= int(occ_thresh)
        if unknown_is_obstacle:
            occupied |= (data < 0)
        return cls(occupied, info.resolution,
                   info.origin.position.x, info.origin.position.y,
                   frame_id=msg.header.frame_id, **kwargs)

    # -- footprint clearing -------------------------------------------------

    def clear_disc(self, cx, cy, radius):
        """Force every cell within ``radius`` [m] of world point (cx, cy) to
        free. Used to clear the robot's own footprint: the cells it physically
        occupies must not read as obstacles (start-of-run self-noise, unknown
        cells, or the tripod/ground the ZED sees at takeoff would otherwise make
        the first rollout point collide, so no trajectory is ever feasible).

        Invalidates the EDT cache so the next query reflects the cleared cells."""
        if radius <= 0.0:
            return
        r_cells = int(np.ceil(radius / self.cell))
        gx = int(np.floor((cx - self.xorigin) / self.cell))
        gy = int(np.floor((cy - self.yorigin) / self.cell))
        y0 = max(0, gy - r_cells); y1 = min(self.H, gy + r_cells + 1)
        x0 = max(0, gx - r_cells); x1 = min(self.W, gx + r_cells + 1)
        if x0 >= x1 or y0 >= y1:
            return  # robot outside the grid extent
        ys, xs = np.mgrid[y0:y1, x0:x1]
        wx = self.xorigin + (xs + 0.5) * self.cell
        wy = self.yorigin + (ys + 0.5) * self.cell
        mask = (wx - cx) ** 2 + (wy - cy) ** 2 <= radius * radius
        sub = self.L[y0:y1, x0:x1]
        sub[mask] = False
        self.L[y0:y1, x0:x1] = sub
        self._edt_cells = None  # cleared cells change the distance transform

    # -- distance transform -------------------------------------------------

    def _ensure_edt(self):
        """(H, W) EDT in *cell* units: distance from each cell to the nearest
        occupied cell. Occupied cells are 0. Cached until the map is rebuilt."""
        if self._edt_cells is None:
            if self.L.any():
                self._edt_cells = _edt_cells(self.L)
            else:
                # No obstacles anywhere -> "infinitely" far (large finite value).
                big = float(self.H + self.W)
                self._edt_cells = np.full((self.H, self.W), big, dtype=np.float32)
        return self._edt_cells

    # -- world<->cell -------------------------------------------------------

    def _dist_m(self, X):
        """X: (..., >=2) world xy -> (...) distance-to-nearest-obstacle [m].

        Points outside the grid map to a large distance (free) unless
        ``oob_is_obstacle``, in which case they map to 0 (collision)."""
        edt = self._ensure_edt()
        gx = np.floor((X[..., 0] - self.xorigin) / self.cell).astype(np.int64)
        gy = np.floor((X[..., 1] - self.yorigin) / self.cell).astype(np.int64)
        inside = (gx >= 0) & (gx < self.W) & (gy >= 0) & (gy < self.H)
        gxc = np.clip(gx, 0, self.W - 1)
        gyc = np.clip(gy, 0, self.H - 1)
        dist = edt[gyc, gxc] * self.cell
        if self.oob_is_obstacle:
            dist = np.where(inside, dist, 0.0)
        else:
            big = (self.H + self.W) * self.cell
            dist = np.where(inside, dist, big)
        return dist.astype(np.float32)

    # -- planner interface --------------------------------------------------

    def collides(self, X, robot_radius):
        """X: (..., >=2) -> bool. Collision when the body disc of radius
        ``robot_radius`` overlaps an occupied cell."""
        return self._dist_m(X) < float(robot_radius)

    def near_obstacle(self, X, infl):
        """X: (..., >=2) -> float (0/1). 1 where within ``infl`` of an obstacle."""
        return (self._dist_m(X) < float(infl)).astype(np.float32)

    def clearance(self, X):
        """X: (..., >=2) world xy -> (...) distance to the nearest obstacle [m]."""
        return self._dist_m(X)


def _brute_edt(occupied):
    """Fallback EDT (cells) if scipy is unavailable. O(H*W*n_obstacles) --
    only used when scipy import failed; the real path uses scipy."""
    H, W = occupied.shape
    oy, ox = np.nonzero(occupied)
    if oy.size == 0:
        return np.full((H, W), float(H + W), dtype=np.float32)
    ys, xs = np.mgrid[0:H, 0:W]
    d = np.full((H, W), np.inf, dtype=np.float32)
    for cy, cx in zip(oy, ox):
        np.minimum(d, np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2), out=d)
    return d
