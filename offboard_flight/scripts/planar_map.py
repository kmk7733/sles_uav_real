#!/usr/bin/python
# -*- coding: utf-8 -*-
"""ROS-free adapter over an existing 2D occupancy grid.

This module does NOT build maps. It consumes the grid the perception stack
already publishes (`/grid_map`, 0.05 m cells, 0 free / 100 occupied / -1
unknown, produced by depth_to_grid.py) and answers the one question the planner
and the safety validator need:

    clearance(x, y) -> distance in metres from (x, y) to the nearest unsafe cell

Everything else -- inflation radii, what counts as a collision, swept-path
checking -- is built on top of that in planar_safety.py. Keeping the map down to
a single distance query is what lets the same planner run against a recorded
grid, a synthetic arena, or a live topic without changing a line.

UNSAFE = OCCUPIED or UNKNOWN or OUTSIDE THE MAP
Unknown space is not free space. The stereo pair has not seen it, so nothing is
known about what is in it, and on this vehicle 60-70% of the grid is unobserved
at any moment (see inflate_grid.py). Treating it as free is what lets a planner
confidently fly into a wall it has never looked at. Out-of-map is the same
argument at the grid boundary.

CONSERVATISM OF THE DISTANCE TRANSFORM
The EDT measures cell-centre to cell-centre, so a raw reading OVERSTATES the
real clearance on two counts: the obstacle starts at the near edge of its cell,
and the query point is somewhere inside its own cell rather than at the centre.
clearance() subtracts `edt_margin` to push the error the safe way. The default
is res/2, which covers the obstacle-cell extent along an axis. A bound that
holds for an arbitrary query point against an arbitrary obstacle cell corner is
sqrt(2)*res -- roughly 7 cm at 0.05 m cells. That is available by raising
edt_margin, but the cheaper place to buy it is d_clr in safe_radius(), which is
what it is for.
"""

import numpy as np

try:
    import cv2 as _cv2
except ImportError:                                    # pragma: no cover
    _cv2 = None

try:
    from scipy import ndimage as _ndimage
except ImportError:                                    # pragma: no cover
    _ndimage = None


def _edt_cells(unsafe):
    """Distance in CELLS from every cell to the nearest unsafe cell."""
    H, W = unsafe.shape
    if not unsafe.any():
        # Nothing unsafe anywhere: report a large finite distance rather than
        # inf, so downstream arithmetic (hinges, sums) stays finite.
        return np.full((H, W), float(H + W), dtype=np.float32)
    if _cv2 is not None:
        # distanceTransform measures distance to the nearest ZERO pixel, so the
        # unsafe set must be the zeros. DIST_MASK_5 is within ~2% of exact L2,
        # which at 0.05 m cells is ~1 mm over a 1 m lookup -- immaterial next to
        # the half-cell correction clearance() already applies.
        src = np.where(unsafe, 0, 1).astype(np.uint8)
        return _cv2.distanceTransform(src, _cv2.DIST_L2, 5).astype(np.float32)
    if _ndimage is not None:                           # pragma: no cover
        return _ndimage.distance_transform_edt(~unsafe).astype(np.float32)
    return _brute_edt(unsafe)                          # pragma: no cover


def _brute_edt(unsafe):                                # pragma: no cover
    """Last-resort EDT if neither cv2 nor scipy is importable."""
    H, W = unsafe.shape
    oy, ox = np.nonzero(unsafe)
    ys, xs = np.mgrid[0:H, 0:W]
    d = np.full((H, W), float(H + W), dtype=np.float32)
    for cy, cx in zip(oy, ox):
        np.minimum(d, np.sqrt((ys - cy) ** 2.0 + (xs - cx) ** 2.0), out=d)
    return d


class PlanarOccupancy(object):
    """Binary unsafe-set over a 2D grid, with a cached distance transform."""

    def __init__(self, unsafe, resolution, origin=(0.0, 0.0), frame_id="map",
                 edt_margin=None):
        """
        unsafe      (H, W) bool. True = occupied, unknown, or otherwise not
                    flyable. Row 0 is the origin row, matching
                    nav_msgs/OccupancyGrid row-major layout.
        resolution  cell size [m]
        origin      world (x, y) of the lower-left corner of cell (0, 0)
        edt_margin  metres subtracted from every clearance reading to absorb the
                    cell-centre-to-cell-centre bias of the distance transform.
                    Defaults to res/2. See the module docstring.
        """
        self.unsafe = np.ascontiguousarray(unsafe, dtype=bool)
        if self.unsafe.ndim != 2:
            raise ValueError("unsafe must be 2D, got %r" % (self.unsafe.shape,))
        self.res = float(resolution)
        self.origin = (float(origin[0]), float(origin[1]))
        self.frame_id = str(frame_id)
        self.edt_margin = (0.5 * self.res if edt_margin is None
                           else float(edt_margin))
        self.H, self.W = self.unsafe.shape
        self._edt = None

    # ------------------------------------------------------------- builders

    @classmethod
    def from_values(cls, values, resolution, origin=(0.0, 0.0),
                    occ_thresh=50, unknown_unsafe=True, frame_id="map",
                    **kw):
        """Build from raw OccupancyGrid cell values (0..100, -1 unknown)."""
        v = np.asarray(values, dtype=np.int16)
        unsafe = v >= int(occ_thresh)
        if unknown_unsafe:
            unsafe |= (v < 0)
        return cls(unsafe, resolution, origin, frame_id=frame_id, **kw)

    @classmethod
    def from_occupancy_grid_msg(cls, msg, occ_thresh=50, unknown_unsafe=True,
                                **kw):
        """Build from anything shaped like a nav_msgs/OccupancyGrid.

        Duck-typed on purpose -- this module never imports nav_msgs, so the same
        code path serves a live subscriber, a rosbag replay and the fake message
        objects the tests use.
        """
        info = msg.info
        v = np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width)
        return cls.from_values(
            v, info.resolution,
            (info.origin.position.x, info.origin.position.y),
            occ_thresh=occ_thresh, unknown_unsafe=unknown_unsafe,
            frame_id=getattr(msg.header, "frame_id", "map"), **kw)

    # ------------------------------------------------------------- geometry

    @property
    def bounds(self):
        """(xmin, ymin, xmax, ymax) of the grid extent in world coordinates."""
        return (self.origin[0], self.origin[1],
                self.origin[0] + self.W * self.res,
                self.origin[1] + self.H * self.res)

    def world_to_cell(self, x, y):
        """Vectorised world -> integer cell index. Returns (ix, iy)."""
        ix = np.floor((np.asarray(x, dtype=np.float64) - self.origin[0])
                      / self.res).astype(np.int64)
        iy = np.floor((np.asarray(y, dtype=np.float64) - self.origin[1])
                      / self.res).astype(np.int64)
        return ix, iy

    def cell_to_world(self, ix, iy):
        """Cell index -> world coordinate of the cell centre."""
        return (self.origin[0] + (np.asarray(ix) + 0.5) * self.res,
                self.origin[1] + (np.asarray(iy) + 0.5) * self.res)

    # ------------------------------------------------------------ clearance

    def _ensure_edt(self):
        if self._edt is None:
            self._edt = _edt_cells(self.unsafe)
        return self._edt

    def clearance(self, x, y):
        """Distance [m] from (x, y) to the nearest unsafe cell.

        x, y may be any matching shape; the result has that shape. Points
        outside the grid return 0.0 -- out-of-map is unsafe, and returning zero
        makes every downstream radius test fail there without a special case.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        edt = self._ensure_edt()

        ix, iy = self.world_to_cell(x, y)
        inside = (ix >= 0) & (ix < self.W) & (iy >= 0) & (iy < self.H)
        d = edt[np.clip(iy, 0, self.H - 1), np.clip(ix, 0, self.W - 1)] * self.res
        d = d.astype(np.float64)

        d = np.maximum(d - self.edt_margin, 0.0)
        return np.where(inside, d, 0.0)

    def is_unsafe_point(self, x, y, radius=0.0):
        """True where a disc of `radius` about (x, y) touches the unsafe set."""
        return self.clearance(x, y) < float(radius)

    # ------------------------------------------------------------ mutation

    def clear_disc(self, cx, cy, radius):
        """Force every cell within `radius` of (cx, cy) to safe.

        Needed for the vehicle's own footprint. With unknown treated as unsafe,
        the cells the vehicle is physically sitting in are frequently unknown --
        the ZED cannot see underneath itself -- so without this the very first
        node of every rollout collides and no trajectory is ever feasible.
        Clearing the footprint asserts the one thing we do know for certain
        about that space: the vehicle is in it, so it is not a wall.
        """
        if radius <= 0.0:
            return
        r_cells = int(np.ceil(radius / self.res))
        gx = int(np.floor((cx - self.origin[0]) / self.res))
        gy = int(np.floor((cy - self.origin[1]) / self.res))
        y0, y1 = max(0, gy - r_cells), min(self.H, gy + r_cells + 1)
        x0, x1 = max(0, gx - r_cells), min(self.W, gx + r_cells + 1)
        if x0 >= x1 or y0 >= y1:
            return                                   # vehicle outside the grid
        ys, xs = np.mgrid[y0:y1, x0:x1]
        wx, wy = self.cell_to_world(xs, ys)
        mask = (wx - cx) ** 2 + (wy - cy) ** 2 <= radius * radius
        block = self.unsafe[y0:y1, x0:x1]
        block[mask] = False
        self.unsafe[y0:y1, x0:x1] = block
        self._edt = None                             # cache is now wrong

    # ------------------------------------------------------------- reporting

    def describe(self):
        n = self.H * self.W
        u = int(self.unsafe.sum())
        xmin, ymin, xmax, ymax = self.bounds
        return ("%dx%d @ %.3f m  extent [%.2f, %.2f] x [%.2f, %.2f]  "
                "unsafe %d/%d (%.1f%%)  frame=%s"
                % (self.W, self.H, self.res, xmin, xmax, ymin, ymax,
                   u, n, 100.0 * u / max(n, 1), self.frame_id))


class FreeSpace(object):
    """Everything is safe. For open-space bench runs only.

    PlanarOccupancy deliberately has no "no map yet" mode that reads as free;
    an empty grid is entirely unknown and therefore entirely unsafe. This class
    exists so that flying without an obstacle set is something a caller has to
    ask for by name, and can be logged as such, rather than something that
    happens by accident when a topic is silent.
    """

    res = 0.05
    frame_id = "none"
    bounds = (-float("inf"), -float("inf"), float("inf"), float("inf"))

    def clearance(self, x, y):
        return np.full(np.asarray(x).shape, 1e3, dtype=np.float64)

    def is_unsafe_point(self, x, y, radius=0.0):
        return np.zeros(np.asarray(x).shape, dtype=bool)

    def clear_disc(self, cx, cy, radius):
        pass

    def describe(self):
        return "FreeSpace (no obstacles -- open-space bench mode)"
