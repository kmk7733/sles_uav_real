#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS-free adapter over an existing 2D occupancy grid.

This module does NOT build maps. It consumes the grid the perception stack
already publishes (`/grid_map`, 0.05 m cells, 0 free / 100 occupied / -1
unknown, produced by depth_to_grid.py) and answers the one question the planner
and the safety validator need:

    clearance(x, y) -> distance in metres from (x, y) to the nearest unsafe cell

Everything else -- inflation radii, what counts as a collision, swept-path
checking -- is built on top of that in planner/safety.py. Keeping the map down
to a single distance query is what lets the same planner run against a recorded
grid, a synthetic arena, or a live topic without changing a line.

THE CLEARANCE PROTOCOL, WHICH IS WHY THIS IS NOT UNDER haa/
`clearance(x, y) -> metres` is the entire interface the planner and the
validator consume, and three different objects implement it:

    PlanarOccupancy                     this file -- a real grid
    FreeSpace                           this file -- open-space bench mode
    perception/occupancy.SplitOccupancy occupied and unknown inflated apart

Nothing downstream may narrow that to a concrete class. `hpa` consumes it too
(through the validator), so it lives at the package root beside `types.py`.

UNSAFE = OCCUPIED or UNKNOWN or OUTSIDE THE MAP
Unknown space is not free space. The stereo pair has not seen it, so nothing is
known about what is in it, and on this vehicle 60-70% of the grid is unobserved
at any moment (see inflate_grid.py). Treating it as free is what lets a planner
confidently fly into a wall it has never looked at. Out-of-map is the same
argument at the grid boundary.

BUT UNKNOWN IS NOT INFLATED, AND OCCUPIED IS
Untraversable and inflated are two different statements, and this class used to
make only the first one. It merged occupied and unknown into a single boolean
before taking one distance transform, so every unobserved cell was pushed back
by the full r_safe -- exactly as if a measured wall stood in it.

r_safe = r_Q + r_perc + r_track + d_clr is what it takes to clear a MEASURED
surface. Unknown space is not a surface, it is a hole in the map: r_perc
describes the error of a range reading that was never taken, and r_track the
tube around a trajectory that has nothing to deviate from. There is nothing
there to be at a distance from, so it earns no margin. The cell itself stays
untraversable -- clearance() returns 0 standing inside it -- and its neighbours
are free.

THIS IS NOT A LOOSENING, IT IS THE SIMULATOR'S RULE.
`planar_sim/perception/occupancy.py:SplitOccupancy` has always done exactly
this, with `mapping.unknown_inflate = 0.0` and the measurement that set it
written beside the key: inflating unknown by even r_quad pushes the frontier
back faster than the camera reveals it, and a run that reached the goal in
10.9 s instead stalled 0.36 m short. The aircraft was flying the other rule
because this file, the one that ships, could not express the sim's. Measured on
flight_20260906_054936: merging blocked 9.6-13.9 percentage points more of the
arena than the rule the simulator validated.

IT CANNOT ACCEPT ANYTHING THE MERGED RULE REJECTED FOR A MEASURED REASON.
d_new is the distance to occupied, d_old the distance to occupied-or-unknown,
so d_new >= d_old everywhere, and inside an unknown cell both are 0. The change
removes a margin around holes in the map and none around surfaces: every plan
the old rule passed still passes.

ONE NUMERICAL CONSEQUENCE, STATED RATHER THAN HIDDEN. edt_cells returns float32
and the old body multiplied by `res` in float32 before widening. This one
widens first, as SplitOccupancy does, which is what makes the two agree to 0.0
rather than to 1e-8. Against the old body the readings differ by up to ~2e-8 m
-- twenty nanometres, four orders below the ~1 mm the cv2/scipy backends already
disagree by (see edt_backend above).

WHAT THIS DOES NOT CHANGE. `haa/cost.py` and `haa/geodesic.py` read
`getattr(occ, "occupied", occ.unsafe)` and `getattr(occ, "unknown", zeros)`,
and on the aircraft they were already getting the right two masks:
`planar_planner_node.py:_grid_cb` recovered the split from the raw message and
assigned it onto the instance, because this class did not carry it. So the
frontier term's UNKNOWN class and the geodesic's optimism were working, and
only `clearance()` -- the validator and the MPPI obstacle cost -- saw the merged
set. Carrying the split here makes that node-side patch redundant rather than
necessary, which is the point: the two files that already asked for `occupied`
and `unknown` now get them from the object itself.

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

THE EDT BACKEND IS PART OF THE ANSWER, SO IT IS REPORTED
`edt_backend()` names which of the three implementations is live, because they
do not agree to the bit and the planner's decisions are downstream of the
difference:

    cv2      DIST_L2 / DIST_MASK_5, within ~2% of exact  <- APPROXIMATE
    scipy    ndimage.distance_transform_edt, exact
    brute    exact, and O(unsafe cells x grid); a fallback, not a choice

A run under an interpreter with cv2 and a run under one with only scipy can
therefore return different clearances for the same grid -- ~1 mm over a 1 m
lookup, immaterial next to the half-cell correction, but not zero, and enough
that "identical seed, identical result" only holds within one backend. The
vendored original had the same property silently; `describe()` now says which
one answered.

THIS IS THE PLANNER'S OWN FILE NOW
It was `vendor/planner/planar_map.py`. The extraction that sat here unused was
AHEAD of it rather than behind: two changes, neither numerical -- `_edt_cells`
is public as `edt_cells`, and the brute-force fallback warns once instead of
being silently slow. So the de-vendoring kept this file and dropped the other.
`planar_sim/perception/occupancy.py` still spells the import
`edt_cells as _edt_cells`, which is the one visible trace of the rename.
"""

import warnings

import numpy as np

try:
    import cv2 as _cv2
except ImportError:                                    # pragma: no cover
    _cv2 = None

try:
    from scipy import ndimage as _ndimage
except ImportError:                                    # pragma: no cover
    _ndimage = None


def edt_backend():
    """Which distance-transform implementation will answer. See the docstring.

    Public because it is a property of the ANSWER, not of the machine: a
    clearance quoted without it is reproducible only by accident.
    """
    if _cv2 is not None:
        return "cv2"
    if _ndimage is not None:
        return "scipy"
    return "brute"


def edt_cells(unsafe):
    """Distance in CELLS from every cell to the nearest unsafe cell.

    PUBLIC, unlike the vendored `planar_map._edt_cells` it was copied from.
    `perception/occupancy.py:SplitOccupancy` runs two of these -- one over
    occupied, one over unknown -- so it has always needed this entry point, and
    reached through the underscore to get it. A leading underscore that the
    package's own sibling has to ignore is not encapsulation, it is a mislabel.
    """
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
    if _ndimage is not None:
        return _ndimage.distance_transform_edt(~unsafe).astype(np.float32)
    return _brute_edt(unsafe)


_warned_brute = False


def _brute_edt(unsafe):
    """Last-resort EDT if neither cv2 nor scipy is importable.

    WARNS ONCE. This path is O(unsafe cells x H x W) in pure Python and the
    vendored original took it in silence, which is how an interpreter with
    neither backend reads as "the planner got slow" rather than "the planner is
    running its fallback distance transform". On this machine the `sles_bc` env
    has neither, so the fallback is reachable in practice, not hypothetical.
    """
    global _warned_brute
    if not _warned_brute:
        _warned_brute = True
        warnings.warn(
            "planner/grid.py: neither cv2 nor scipy is importable, so the "
            "distance transform is running the brute-force fallback "
            "(O(unsafe cells x grid) in Python). Every clearance() query on a "
            "fresh grid pays it. Install either, or use an interpreter that "
            "has one -- see planar_sim/paths.py.",
            RuntimeWarning, stacklevel=3)
    H, W = unsafe.shape
    oy, ox = np.nonzero(unsafe)
    ys, xs = np.mgrid[0:H, 0:W]
    d = np.full((H, W), float(H + W), dtype=np.float32)
    for cy, cx in zip(oy, ox):
        np.minimum(d, np.sqrt((ys - cy) ** 2.0 + (xs - cx) ** 2.0), out=d)
    return d


class PlanarOccupancy(object):
    """Unsafe-set over a 2D grid, occupied and unknown kept apart."""

    def __init__(self, unsafe, resolution, origin=(0.0, 0.0), frame_id="map",
                 edt_margin=None, unknown=None, unknown_inflate=0.0,
                 r_safe=None):
        """
        unsafe      (H, W) bool. True = occupied, unknown, or otherwise not
                    flyable. Row 0 is the origin row, matching
                    nav_msgs/OccupancyGrid row-major layout.
        resolution  cell size [m]
        origin      world (x, y) of the lower-left corner of cell (0, 0)
        edt_margin  metres subtracted from every clearance reading to absorb the
                    cell-centre-to-cell-centre bias of the distance transform.
                    Defaults to res/2. See the module docstring.
        unknown     (H, W) bool, the subset of `unsafe` that is unsafe because
                    it was never observed rather than because something was
                    measured in it. None (the default) means "nothing here is
                    unknown", which is the ground-truth case and reproduces this
                    class's pre-split behaviour BIT FOR BIT: with an all-false
                    mask `occupied` is `unsafe`, the unknown branch of
                    clearance() is dead, and the same single EDT is taken.
        unknown_inflate
                    metres of margin around the unknown boundary. 0.0 -- only
                    measured surfaces are inflated. See the module docstring for
                    the measurement behind the zero.
        r_safe      the level a caller compares clearance() against. Needed ONLY
                    when unknown_inflate > 0, because that margin is folded into
                    the same scalar: it is what makes
                    `clearance >= r_safe` mean `d_occ >= r_safe AND
                    d_unknown >= unknown_inflate`. Passing the margin without
                    the level it is measured against is an error, not a default.
        """
        self.unsafe = np.ascontiguousarray(unsafe, dtype=bool)
        if self.unsafe.ndim != 2:
            raise ValueError("unsafe must be 2D, got %r" % (self.unsafe.shape,))
        if unknown is None:
            self.unknown = np.zeros_like(self.unsafe)
        else:
            u = np.ascontiguousarray(unknown, dtype=bool)
            if u.shape != self.unsafe.shape:
                raise ValueError("unknown %r does not match unsafe %r"
                                 % (u.shape, self.unsafe.shape))
            # Not in-place: `ascontiguousarray` hands back the caller's own
            # array when it is already contiguous bool, and this class has no
            # business editing it.
            self.unknown = u & self.unsafe
        self.occupied = self.unsafe & ~self.unknown
        self.unknown_inflate = float(unknown_inflate)
        self.r_safe = None if r_safe is None else float(r_safe)
        if self.unknown_inflate > 0.0 and self.r_safe is None:
            raise ValueError(
                "unknown_inflate=%.3f needs r_safe: the margin is expressed "
                "against the level clearance() is compared to. See the "
                "docstring and perception/occupancy.py:SplitOccupancy."
                % self.unknown_inflate)
        self.res = float(resolution)
        self.origin = (float(origin[0]), float(origin[1]))
        self.frame_id = str(frame_id)
        self.edt_margin = (0.5 * self.res if edt_margin is None
                           else float(edt_margin))
        self.H, self.W = self.unsafe.shape
        self._edt = None
        self._edt_unk = None

    def _resplit(self):
        """Recompute `occupied` after `unsafe` or `unknown` was written.

        STORED, NOT A PROPERTY, and that is a compatibility decision rather than
        a stylistic one. `planar_planner_node.py:_grid_cb` assigns
        `occ.occupied = ...` and `occ.unknown = ...` straight onto the instance
        -- it recovered the split from the raw message back when this class did
        not carry it -- and a read-only property turns that line into an
        AttributeError swallowed by the node's own `except`, which shows up as
        "grid parse failed" and a planner flying on the last map it managed to
        parse. An attribute keeps that code working while it is retired.
        """
        self.occupied = self.unsafe & ~self.unknown
        self._edt = None
        self._edt_unk = None

    # ------------------------------------------------------------- builders

    @classmethod
    def from_values(cls, values, resolution, origin=(0.0, 0.0),
                    occ_thresh=50, unknown_unsafe=True, frame_id="map",
                    **kw):
        """Build from raw OccupancyGrid cell values (0..100, -1 unknown).

        The -1 cells are handed on as the `unknown` mask, not just folded into
        `unsafe`, so the object downstream can tell "never looked" from
        "measured a surface". With `unknown_unsafe=False` they are free space
        and no mask is passed: an unknown cell that is not unsafe must not be
        given clearance 0 by the unknown branch.
        """
        v = np.asarray(values, dtype=np.int16)
        unsafe = v >= int(occ_thresh)
        unk = None
        if unknown_unsafe:
            unk = v < 0
            unsafe = unsafe | unk
        return cls(unsafe, resolution, origin, frame_id=frame_id, unknown=unk,
                   **kw)

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
            self._edt = edt_cells(self.occupied)
        return self._edt

    def _ensure_edt_unk(self):
        if self._edt_unk is None:
            self._edt_unk = edt_cells(self.unknown)
        return self._edt_unk

    def clearance(self, x, y):
        """Distance [m] from (x, y) to the nearest OCCUPIED cell; 0 in unknown.

        x, y may be any matching shape; the result has that shape. Points
        outside the grid return 0.0 -- out-of-map is unsafe, and returning zero
        makes every downstream radius test fail there without a special case.
        Standing in an unobserved cell returns 0.0 for the same reason, so
        unknown space is untraversable at any r_safe; what it does NOT do is
        push the boundary back by r_safe, because there is no measured surface
        there to be at a distance from. See the module docstring.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        edt = self._ensure_edt()

        ix, iy = self.world_to_cell(x, y)
        inside = (ix >= 0) & (ix < self.W) & (iy >= 0) & (iy < self.H)
        jy = np.clip(iy, 0, self.H - 1)
        jx = np.clip(ix, 0, self.W - 1)
        d = edt[jy, jx].astype(np.float64) * self.res
        d = np.maximum(d - self.edt_margin, 0.0)

        if self.unknown_inflate > 0.0:
            du = self._ensure_edt_unk()[jy, jx].astype(np.float64) * self.res
            du = np.maximum(du - self.edt_margin, 0.0)
            # clearance >= r_safe  <=>  d >= r_safe AND du >= unknown_inflate
            d = np.minimum(d, du - self.unknown_inflate + self.r_safe)

        d = np.where(self.unknown[jy, jx], 0.0, d)
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
        for arr in (self.unsafe, self.unknown):
            block = arr[y0:y1, x0:x1]
            block[mask] = False
            arr[y0:y1, x0:x1] = block
        self._resplit()                    # occupied, and both caches, restated

    # ------------------------------------------------------------- reporting

    def describe(self):
        n = self.H * self.W
        u = int(self.unsafe.sum())
        xmin, ymin, xmax, ymax = self.bounds
        k = int(self.unknown.sum())
        return ("%dx%d @ %.3f m  extent [%.2f, %.2f] x [%.2f, %.2f]  "
                "unsafe %d/%d (%.1f%%) of which unknown %d (%.1f%%), "
                "unknown_inflate %.2f m  frame=%s  edt=%s"
                % (self.W, self.H, self.res, xmin, xmax, ymin, ymax,
                   u, n, 100.0 * u / max(n, 1), k, 100.0 * k / max(n, 1),
                   self.unknown_inflate, self.frame_id, edt_backend()))


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
