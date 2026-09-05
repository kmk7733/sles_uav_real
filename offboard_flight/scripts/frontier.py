#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PA-MPPI's frontier term, ported to the planar planner.

    Zhai et al., "PA-MPPI: Perception-Aware Model Predictive Path Integral
    Control for Quadrotor Navigation in Unknown Environments", RA-L 2026.
    arXiv:2509.14978

THE PROBLEM IT SOLVES
Our cost has no reason to look at unexplored space. Unknown is unsafe, so the
planner will not enter it; and nothing rewards moving somewhere that would
REVEAL it. The result is a planner that can only exploit the free space it
already has, and stalls the moment the goal is behind something it has not
looked at. That is the deadlock we hit with an all-unknown starting map.

THE TERM
Trace a ray from the trajectory's terminal position to the goal and ask what it
hits FIRST:

    clear line of sight   ->  0            (the term switches itself off)
    an OCCUPIED cell      ->  +c_occupied  (the goal is walled off from here)
    an UNKNOWN cell       ->  -c_unknown   (a REWARD -- there might be a way
                                            through, go and look)

So among positions that cannot see the goal, the planner prefers the ones whose
blockage is merely unexplored over the ones that are genuinely walled off. The
reward is negative cost, which is the part that makes it exploration rather than
mere preference.

WHAT IS DELIBERATELY NOT PORTED
PA-MPPI also carries a point-of-interest term that aligns the camera axis with
the goal direction, and switches its whole cost function between an "occluded"
and a "visible" phase with c_goal changing by a factor of 40. Those are separate
changes with separate consequences; this module is the frontier term alone, so
that its effect can be attributed.

DIFFERENCES FROM THE PAPER, AND WHY
* 2D, not 3D. Our occupancy is a horizontal slice, so the ray march is 2D. The
  paper uses a 3D DDA; here a fixed-step march at half a cell is simpler and the
  grid is small enough that it costs nothing.
* Weights are not the paper's numbers. PA-MPPI's costs are O(1-15); ours are
  O(100) and MPPI weights on the SPREAD of the cost across samples, not its
  magnitude. Copying c_unknown = -4.0 into this cost would be lost in the noise,
  so the paper's ratio (occupied : unknown = 2 : -4) is kept and the overall
  scale is exposed as `w_frontier` to be tuned against our own terms.
* Evaluated at the terminal node only, as in the paper (k = H-1), for cost.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

from planar_mppi import PlanarMPPI

CLEAR, OCCUPIED, UNKNOWN = 0, 1, -1


def first_blocking_class(occ, P, goal, step=None, max_samples=256,
                         r_pass=0.0):
    """What does the ray from each point to the goal hit first?

    P (K, 2) positions, goal (2,). Returns (K,) of CLEAR / OCCUPIED / UNKNOWN.

    The ray is parameterised t in [0, 1] from the point to the goal, exactly as
    in the paper, so its length is the distance to the goal and a point already
    at the goal trivially has line of sight.
    """
    P = np.asarray(P, dtype=np.float64).reshape(-1, 2)
    g = np.asarray(goal, dtype=np.float64).reshape(2)
    # Works against a SplitOccupancy (occupied/unknown separate) and against a
    # plain PlanarOccupancy (ground truth, nothing unknown) so the same ray can
    # be used to describe a scenario as to plan in it.
    occupied = np.asarray(getattr(occ, "occupied", occ.unsafe))
    unknown = np.asarray(getattr(occ, "unknown",
                                 np.zeros_like(occupied, dtype=bool)))
    # A RAY HAS NO WIDTH; THE VEHICLE DOES.
    # Testing bare cell occupancy asks "can a photon get through", which is a
    # different question from "can a disc of r_pass get through" and answers
    # CLEAR too early. Measured on `detour`: standing in front of the gap at
    # (1.33, -0.50) the line of sight to the goal threads the opening at
    # x = 1.11, so the term reported CLEAR and switched itself off -- at the
    # one place guidance was needed. With r_pass = r_safe the same ray is
    # blocked until the vehicle can actually fit through.
    if r_pass > 0.0:
        passable = distance_transform_edt(~occupied) * float(occ.res) >= r_pass
        occupied = occupied | (~passable & ~unknown)
    res = float(occ.res)
    ox, oy = occ.origin
    H, W = occupied.shape

    d = g[None, :] - P                                    # (K, 2)
    dist = np.linalg.norm(d, axis=1)                      # (K,)
    step = 0.5 * res if step is None else float(step)
    m = int(np.ceil(float(dist.max()) / step)) + 1 if dist.size else 2
    m = int(np.clip(m, 2, max_samples))

    t = np.linspace(0.0, 1.0, m)[None, :]                 # (1, m)
    xs = P[:, 0:1] + t * d[:, 0:1]
    ys = P[:, 1:2] + t * d[:, 1:2]

    ix = np.floor((xs - ox) / res).astype(np.int64)
    iy = np.floor((ys - oy) / res).astype(np.int64)
    inb = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    jx = np.clip(ix, 0, W - 1)
    jy = np.clip(iy, 0, H - 1)

    occ_hit = occupied[jy, jx] & inb
    unk_hit = unknown[jy, jx] & inb
    # Outside the grid is not free either; treat it as occupied so a ray that
    # leaves the arena is not rewarded as if it were unexplored.
    occ_hit |= ~inb
    blocked = occ_hit | unk_hit

    any_block = blocked.any(axis=1)
    first = np.argmax(blocked, axis=1)                    # 0 when none blocked
    rows = np.arange(P.shape[0])
    cls = np.where(occ_hit[rows, first], OCCUPIED, UNKNOWN)
    return np.where(any_block, cls, CLEAR)


class FrontierMPPI(PlanarMPPI):
    """PlanarMPPI plus PA-MPPI's frontier term on the terminal node.

    The vendored planner is not modified; only `_cost` is extended, so a diff
    against sles_uav_real stays meaningful and the term can be switched off by
    constructing the base class instead.
    """

    def __init__(self, *a, **kw):
        self.w_frontier = float(kw.pop("w_frontier", 0.0))
        self.c_occupied = float(kw.pop("c_occupied", 2.0))
        self.c_unknown = float(kw.pop("c_unknown", -4.0))
        self.use_geodesic = bool(kw.pop("use_geodesic", False))
        self.unknown_free = bool(kw.pop("geodesic_unknown_free", True))
        super(FrontierMPPI, self).__init__(*a, **kw)
        self.last_frontier = None          # class per sample, for diagnostics
        self.ctg = None                    # CostToGo for this solve

    def frontier_cost(self, X, goal):
        """(K, n, 6) -> (K,) frontier cost, evaluated at the terminal node."""
        occ = self.validator.occ
        if self.w_frontier == 0.0 or occ is None:
            return np.zeros(X.shape[0])
        if not hasattr(occ, "unsafe"):        # FreeSpace: nothing to reason about
            return np.zeros(X.shape[0])
        cls = first_blocking_class(occ, X[:, -1, 0:2], goal,
                                   r_pass=self.validator.r_safe)
        self.last_frontier = cls
        c = np.zeros(X.shape[0])
        c[cls == OCCUPIED] = self.c_occupied
        c[cls == UNKNOWN] = self.c_unknown
        return self.w_frontier * c

    def geodesic_swap(self, X, goal):
        """Correction that turns the Euclidean goal terms into geodesic ones.

        The base cost is not reimplemented -- it is corrected. Subtracting the
        two Euclidean goal terms and adding their geodesic counterparts leaves
        every other term (obstacle, slew, yaw) exactly as the vendored planner
        computed it, so this cannot silently drift from it.
        """
        if not self.use_geodesic or self.ctg is None:
            return 0.0
        w = self.w
        P = X[..., 0:2]
        eu = np.linalg.norm(P - np.asarray(goal)[None, None, :], axis=-1)
        ge = self.ctg.query(P)
        return (w.w_goal * (ge[:, :-1] - eu[:, :-1]).sum(axis=1)
                + w.w_term_pos * (ge[:, -1] - eu[:, -1]))

    def _cost(self, X, U, goal, a_prev):
        return (super(FrontierMPPI, self)._cost(X, U, goal, a_prev)
                + self.frontier_cost(X, goal)
                + self.geodesic_swap(X, goal))

    def plan(self, state, goal, **kw):
        # The map changes every tick, so the field is rebuilt every solve.
        if self.use_geodesic and getattr(self.validator, "occ", None) is not None:
            occ = self.validator.occ
            if hasattr(occ, "unsafe"):
                from geodesic import CostToGo
                self.ctg = CostToGo(occ, goal, self.validator.r_safe,
                                    unknown_free=self.unknown_free)
        return super(FrontierMPPI, self).plan(state, goal, **kw)

    def describe(self):
        base = super(FrontierMPPI, self).describe()
        if self.w_frontier == 0.0:
            return base + "\n  frontier term OFF"
        return (base + "\n  frontier (PA-MPPI): w %.1f  occupied %+.1f  "
                       "unknown %+.1f  -> terminal-node ray to goal"
                % (self.w_frontier, self.c_occupied, self.c_unknown))
