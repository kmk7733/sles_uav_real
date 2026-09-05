# -*- coding: utf-8 -*-
"""The log-odds accumulator, vehicle side.

`AndertMapper` (simulator, `planar_sim/perception/andert.py`) could not be
vendored -- it drags in scipy's Rotation, the simulated stereo renderer,
SplitOccupancy and the YAML config layer, none of which exist or belong here.
What it actually does that this node needs is small, and it is reproduced here
with the source cited line by line so the two can be diffed by eye.

THE THREE THINGS THAT ARE NOT OBVIOUS, all of them inherited:

1. `seen` IS RECORDED, NEVER INFERRED FROM `L`.  andert.py:128-138
   Inferring it as `L == 0.0` looks equivalent and is not: a cell seen as free
   accumulates negative log-odds and a later opposite contribution can cancel
   it back to exactly 0.0, at which point a cell that HAS been observed becomes
   indistinguishable from one that never was. Measured in the simulator before
   this was split out: 26, 73 and 143 cells per frame reverting known ->
   unknown, i.e. the map SHRINKING as the vehicle flew. This mask only ever
   gains bits, so the observed set is monotone by construction.

2. UNKNOWN IS `~seen`, NOT A LOG-ODDS BAND.  andert.py:361-363
   The node this replaces published `-1` for everything in (-0.4, +0.4), which
   folded "observed and ambiguous" into "never observed". Downstream
   (`planar_map.py`, `unknown_unsafe: True`) both read as obstacle, so the
   distinction is not academic: it decides whether ambiguity is flyable.

3. THE FREE DISC IS ASSERTED, AND ASSERTING COUNTS AS OBSERVING.
   andert.py:174-201. Setting `L` without setting `seen` publishes `-1`, which
   is unsafe, which is exactly the deadlock the disc exists to prevent. The
   claim being made is "the vehicle is physically here, so this is not a wall",
   and it must not evaporate the first time a measurement cancels the log-odds.
"""

import numpy as np


def logit(p):
    """andert.py:85-87."""
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))


class AndertGrid(object):
    """Accumulates Andert log-odds frames into one persistent grid.

    Row index is y, column index is x -- the same convention as
    `build_frame_grid`'s (grid_h, grid_w) return and as nav_msgs/OccupancyGrid's
    row-major data, so no transpose happens anywhere in this package.
    """

    def __init__(self, nx, ny, res, min_x, min_y,
                 log_odds_clip=10.0, p_occ=0.5):
        self.nx = int(nx)
        self.ny = int(ny)
        self.res = float(res)
        self.min_x = float(min_x)
        self.min_y = float(min_y)
        self.l_clip = float(log_odds_clip)
        self.occ_thresh = logit(p_occ)          # andert.py:119
        self.L = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.seen = np.zeros((self.ny, self.nx), dtype=bool)
        self.n_frames = 0
        self.seeded = False

    # ---------------------------------------------------------------- setup

    def seed_free_disc(self, cx, cy, radius):
        """Assert a disc around (cx, cy) free, before any observation.

        andert.py:174-201. The camera cannot see underneath or immediately
        around itself -- the frustum has a near plane and the field of view is
        a wedge -- so the vehicle's own surroundings stay unknown however long
        it looks. With unknown-is-unsafe that is a deadlock: the first solve
        fails, the vehicle cannot move, and it therefore never observes
        anything new.

        Returns the number of cells asserted, so a caller can tell "seeded"
        from "seeded entirely outside the grid".
        """
        r = float(radius)
        if r <= 0.0:
            return 0
        xs = self.min_x + (np.arange(self.nx) + 0.5) * self.res
        ys = self.min_y + (np.arange(self.ny) + 0.5) * self.res
        gx, gy = np.meshgrid(xs, ys)
        disc = (gx - float(cx)) ** 2 + (gy - float(cy)) ** 2 <= r * r
        # A definite but NOT saturated free belief, so a later real measurement
        # can still override it rather than being outvoted by an assumption.
        self.L[disc] = np.minimum(self.L[disc], -2.0)
        self.seen |= disc                       # see note 3 in the module head
        self.seeded = True
        return int(disc.sum())

    def assert_seen(self, mask):
        """Mark cells observed without claiming anything about occupancy.

        Used for the border wall, whose cells are asserted occupied rather than
        measured and would otherwise publish as unknown.
        """
        self.seen |= mask

    # --------------------------------------------------------------- update

    def integrate(self, frame, touched):
        """Fold one `build_frame_grid` result in. andert.py:306-308.

        `touched` is RETURNED by the walk, not inferred from `frame != 0`: a
        cell whose interval maximum lands exactly on P = 0.5 contributes 0.0
        log-odds and WAS seen, yet would read as never observed.
        """
        self.seen |= touched
        self.L += frame
        np.clip(self.L, -self.l_clip, self.l_clip, out=self.L)
        self.n_frames += 1

    # -------------------------------------------------------------- readout

    def to_int8(self):
        """The belief as nav_msgs/OccupancyGrid data. andert.py:361-363.

            100  occupied   L > logit(p_occ)
              0  free       observed and not occupied
             -1  unknown    never observed  --  and nothing else
        """
        out = np.full((self.ny, self.nx), -1, dtype=np.int8)
        occupied = self.L > self.occ_thresh
        out[self.seen] = 0
        out[occupied] = 100
        return out

    def coverage(self):
        """Fractions, for the log line that says whether the map is growing."""
        n = float(self.L.size)
        seen = float(self.seen.sum())
        occ = float((self.L > self.occ_thresh).sum())
        return {"seen_pct": 100.0 * seen / n,
                "occ_pct": 100.0 * occ / n,
                "free_pct": 100.0 * (seen - occ) / n,
                "frames": self.n_frames}
