#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Inflated obstacle set for HAA collision checking.

Consumes a nav_msgs/OccupancyGrid (from depth_to_grid.py, resolution 0.05,
values 0 free / 100 occupied / -1 unknown) and turns it into a binary grid
that has already been dilated, so a collision test is a single array lookup
per rollout point instead of a per-point disk check.

Inflation radius, DeSimplex paper eq (79):

    r_eff = r_Q + r_track + r_perc

  r_Q      quadrotor disk footprint radius
  r_track  low-level tracking error -- how far the true state can sit from
           the nominal one. With PX4 doing the tracking this is PX4's position
           hold error, and it is exactly the role of the robust invariant set
           Z in eq (14): O_inflated = O (+) Z. Measure it and set it; the
           default here is the project's current 5 cm working value.
  r_perc   stereo reconstruction / occupancy-grid uncertainty

Because eq (14)-(15) inflate by Z, avoiding the inflated set with the NOMINAL
trajectory implies the TRUE trajectory avoids the real obstacles. That is why
r_track belongs in this radius and not in the cost function.
"""

import numpy as np

try:
    import cv2
    _HAVE_CV2 = True
except ImportError:                                   # pragma: no cover
    _HAVE_CV2 = False
    from scipy import ndimage


class InflatedGrid(object):
    """Binary inflated occupancy grid with world <-> cell lookup."""

    def __init__(self, occ_thresh=50, unknown_is_obstacle=True):
        """
        occ_thresh           OccupancyGrid value at/above which a cell counts
                             as occupied
        unknown_is_obstacle  treat -1 (never observed) as blocked. True keeps
                             rollouts inside observed free space, which is the
                             conservative choice HAA wants -- the vehicle does
                             not fly into space the stereo pair has not seen.
        """
        self.occ_thresh = int(occ_thresh)
        self.unknown_is_obstacle = bool(unknown_is_obstacle)

        self.ready = False
        self.frame_id = ""
        self.stamp = None
        self.res = None
        self.origin = None          # (ox, oy) world coords of cell (0,0)
        self.nx = self.ny = 0
        self.blocked = None         # (ny, nx) bool, already inflated
        self.r_eff = 0.0

    # ------------------------------------------------------------- ingestion

    def update(self, msg, r_eff):
        """Ingest a nav_msgs/OccupancyGrid and inflate by r_eff metres."""
        self.frame_id = msg.header.frame_id
        self.stamp = msg.header.stamp
        self.res = msg.info.resolution
        self.nx = msg.info.width
        self.ny = msg.info.height
        self.origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.r_eff = float(r_eff)

        g = np.asarray(msg.data, dtype=np.int16).reshape(self.ny, self.nx)

        occ = g >= self.occ_thresh
        if self.unknown_is_obstacle:
            occ |= (g < 0)

        self.blocked = self._dilate(occ, self.r_eff, self.res)
        self.ready = True

    @staticmethod
    def _dilate(occ, radius_m, res):
        """Grow the occupied set by radius_m, using a disk structuring element.

        A disk (not a square) matters: a square element would over-inflate the
        diagonals by up to sqrt(2), which for a 0.36 m radius is an extra
        0.15 m of phantom obstacle on the diagonal directions.
        """
        if radius_m <= 0.0:
            return occ.copy()

        r = int(np.ceil(radius_m / res))
        if r < 1:
            return occ.copy()

        d = 2 * r + 1
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        disk = (xx * xx + yy * yy) <= (radius_m / res) ** 2

        src = occ.astype(np.uint8)
        if _HAVE_CV2:
            out = cv2.dilate(src, disk.astype(np.uint8),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0)
        else:                                          # pragma: no cover
            out = ndimage.binary_dilation(src, structure=disk).astype(np.uint8)
        return out.astype(bool)

    # ---------------------------------------------------------------- lookup

    def world_to_cell(self, x, y):
        """Vectorised world -> integer cell index. Returns (ix, iy)."""
        ix = np.floor((np.asarray(x) - self.origin[0]) / self.res).astype(np.int32)
        iy = np.floor((np.asarray(y) - self.origin[1]) / self.res).astype(np.int32)
        return ix, iy

    def collides(self, x, y, outside_is_blocked=True):
        """Vectorised collision test against the inflated set.

        x, y may be any matching shape (e.g. (K, N) for MPPI rollouts).
        Returns a bool array of that shape.

        outside_is_blocked: points beyond the grid extent count as blocked.
        The grid is a bounded window around the arena, so leaving it means
        leaving the region HAA can certify -- treating that as free would let
        rollouts escape through the map edge.
        """
        x = np.asarray(x)
        if not self.ready:
            # No map yet: report everything blocked, so the caller's "all
            # samples infeasible" path fires instead of silently flying blind.
            return np.ones(x.shape, dtype=bool)

        ix, iy = self.world_to_cell(x, y)
        inside = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)

        hit = np.empty(x.shape, dtype=bool)
        hit.fill(bool(outside_is_blocked))
        if inside.any():
            hit[inside] = self.blocked[iy[inside], ix[inside]]
        return hit

    def distance_to_blocked(self, x, y):
        """Approximate clearance [m] from (x, y) to the nearest blocked cell.

        Used by the safety-margin check (paper eq 62), not by the planner cost.
        Computed lazily with a distance transform over the inflated grid.
        """
        if not self.ready:
            return 0.0
        if getattr(self, "_edt", None) is None or self._edt_stamp != self.stamp:
            free = (~self.blocked).astype(np.uint8)
            if _HAVE_CV2:
                edt = cv2.distanceTransform(free, cv2.DIST_L2, 5)
            else:                                      # pragma: no cover
                edt = ndimage.distance_transform_edt(free)
            self._edt = edt * self.res
            self._edt_stamp = self.stamp

        ix, iy = self.world_to_cell(x, y)
        if not (0 <= ix < self.nx and 0 <= iy < self.ny):
            return 0.0
        return float(self._edt[iy, ix])


def effective_radius(r_quad, r_track, r_perc):
    """DeSimplex eq (79): r_eff = r_Q + r_track + r_perc."""
    return float(r_quad) + float(r_track) + float(r_perc)
