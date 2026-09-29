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

from planner.mppi import PlanarMPPI
from planner.types import IPSI, NNU, S_POS, S_VEL, U_ACC

# Hoisted from inside `plan`, where it was a deferred import. It never guarded
# a cycle -- geodesic.py imports numpy and scipy and nothing from this package
# -- it was only ever there because the two files lived in different packages.
from planner.haa.geodesic import CostToGo, MapCostCache

CLEAR, OCCUPIED, UNKNOWN = 0, 1, -1


def _ray_cells(P, d, t, res, ox, oy, H, W):
    """Return the cells from the original shared ray sampling coordinates."""
    xs = P[:, 0:1] + t * d[:, 0:1]
    ys = P[:, 1:2] + t * d[:, 1:2]
    ix = np.floor((xs - ox) / res).astype(np.int64)
    iy = np.floor((ys - oy) / res).astype(np.int64)
    inb = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    return np.clip(ix, 0, W - 1), np.clip(iy, 0, H - 1), inb


def _ray_cell_classes(classes, P, d, t, res, ox, oy):
    """Gather one classification grid instead of two separate masks."""
    H, W = classes.shape
    ix, iy, inb = _ray_cells(P, d, t, res, ox, oy, H, W)
    sampled = classes[iy, ix]
    sampled[~inb] = OCCUPIED
    return sampled


def first_blocking_class(occ, P, goal, step=None, max_samples=256,
                         r_pass=0.0, cache=None):
    """What does the ray from each point to the goal hit first?

    P (K, 2) positions, goal (2,). Returns (K,) of CLEAR / OCCUPIED / UNKNOWN.

    The ray is parameterised t in [0, 1] from the point to the goal, exactly as
    in the paper, so its length is the distance to the goal and a point already
    at the goal trivially has line of sight. An optional planner-owned cache
    shares the exact occupied EDT with geodesic guidance and other ray calls.
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
        if cache is None:
            passable = distance_transform_edt(~occupied) * float(occ.res) >= r_pass
        else:
            passable = cache.passable(occupied, occ.res, r_pass)
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
    if P.shape[0] == 1:
        # Acceptance evaluates one ray. Building a whole classification grid
        # costs more than its two small gathers, especially on mixed maps.
        ix, iy, inb = _ray_cells(P, d, t, res, ox, oy, H, W)
        occ_hit = occupied[iy, ix] | ~inb
        unk_hit = unknown[iy, ix]
        first = int(np.argmax(occ_hit | unk_hit, axis=1)[0])
        cls = OCCUPIED if occ_hit[0, first] else (
            UNKNOWN if unk_hit[0, first] else CLEAR)
        return np.asarray([cls], dtype=np.int64)

    # Resolve overlapping masks with the original occupied-first precedence.
    # int8 keeps the sampled K x m array small; the public result stays int64.
    classes = np.zeros((H, W), dtype=np.int8)
    classes[unknown] = UNKNOWN
    classes[occupied] = OCCUPIED

    # The sampling density belongs to the WHOLE input batch. Compute m and t
    # above before removing any rays, including rays already blocked at t=0.
    if m > 0:
        result = _ray_cell_classes(classes, P, d, t[:, :1], res, ox, oy)[:, 0]
        clear = result == CLEAR
        if not clear.any():
            return result.astype(np.int64)
        sample = _ray_cell_classes(classes, P[clear], d[clear], t, res, ox, oy)
        first = np.argmax(sample != CLEAR, axis=1)
        # An entirely clear row selects index zero, which is itself CLEAR;
        # an additional any-block reduction and final np.where are unnecessary.
        result[clear] = sample[np.arange(sample.shape[0]), first]
        return result.astype(np.int64)

    # Retain the original error behavior for a zero-length sampling axis.
    sample = _ray_cell_classes(classes, P, d, t, res, ox, oy)
    first = np.argmax(sample != CLEAR, axis=1)
    return sample[np.arange(P.shape[0]), first].astype(np.int64)


# Half-angle [rad] of the heading cone inside which yaw_mode 'goal_in_view'
# costs nothing: the ZED 2i's 84.9 deg horizontal FOV is +-42.4 deg; 40 deg
# keeps the goal a little inside the image edge.
GOAL_VIEW_HALF_ANGLE = np.radians(40.0)


class FrontierMPPI(PlanarMPPI):
    """PlanarMPPI plus PA-MPPI's frontier term on the terminal node.

    `mppi.py` is not modified; only `_cost` is extended, so a diff
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
        self._map_cost_cache = MapCostCache()
        # Below this speed there is no travel direction to align the camera
        # to, so the yaw term aims at the goal instead. `dyn.lim.v_max` is the
        # planner's own ceiling, so this tracks it rather than a constant.
        self._v_stopped = 0.15 * float(self.dyn.lim.v_max)

    def frontier_cost(self, X, goal):
        """(K, n, 6) -> (K,) frontier cost, evaluated at the terminal node."""
        occ = self.validator.occ
        if self.w_frontier == 0.0 or occ is None:
            return np.zeros(X.shape[0])
        if not hasattr(occ, "unsafe"):        # FreeSpace: nothing to reason about
            return np.zeros(X.shape[0])
        cls = first_blocking_class(occ, X[:, -1, 0:2], goal,
                                   r_pass=self.validator.r_safe,
                                   cache=self._map_cost_cache)
        self.last_frontier = cls
        c = np.zeros(X.shape[0])
        c[cls == OCCUPIED] = self.c_occupied
        c[cls == UNKNOWN] = self.c_unknown
        return self.w_frontier * c

    def _cost(self, X, U, goal, a_prev):
        """THE cost, owned here in full.

        Until 2026-08-13 this called `PlanarMPPI._cost` and
        applied a geodesic CORRECTION on top (subtract the Euclidean goal
        terms, add geodesic ones). Rewritten to state the cost directly, so
        the whole objective is readable in one place and terms this project
        decided against are visibly absent. `mppi.py` is unchanged; this
        subclass simply no longer defers to it.

            J = w_goal      * sum_{k<N} d_k          d = D_geo (or Euclidean
              + w_term_pos  * d_N                        when no field is up)
              + w_term_vel  * |v_N|^2
              + w_obs       * sum max(0, d_infl - clearance)^2
              + sum (nu_k - nu_{k-1})^T R_dnu (nu_k - nu_{k-1})
              + w_frontier  * frontier_cost(terminal)

        REMOVED, deliberately (see docs/HAA_MPPI.md):
        * yaw alignment  w_yaw * sum wrap(psi_k - psi_ref)^2.  Heading is not
          a safety quantity here -- unknown space is rejected by the hard
          gate, obstacles are static, so a badly aimed camera stalls progress
          rather than causing collisions -- and at w_yaw = 0 in every run this
          project ever flew, the term was dead code with a live tuning knob.
          If it is ever revived, restore `_yaw_reference` from `mppi.py`
          (modes goal / velocity / hold) rather than reinventing it.
        """
        P = X[..., S_POS]
        # Goal terms. Geodesic distance-to-go when the Dijkstra field is up
        # (`plan` refreshes it each solve); Euclidean otherwise. This replaces
        # the old swap-correction with the same value, stated directly.
        if self.use_geodesic and self.ctg is not None:
            d = self.ctg.query(P)
        else:
            d = np.linalg.norm(P - np.asarray(goal)[None, None, :], axis=-1)
        cl = self.validator.clearance(P) if self.w.w_obs > 0.0 else None
        return self._cost_from_parts(X, U, goal, a_prev, dict(d=d, clearance=cl, frontier=None))

    def _cost_from_parts(self, X, U, goal, a_prev, parts):
        """`_cost` given the per-node goal distance d (K,n), clearance (K,n)
        and optionally the terminal frontier class (K,). Sums and their order
        are exactly `_cost`'s; only where d / clearance / class were looked up
        differs (NumPy in `_cost`, or planner/haa/cuda_batch.py)."""
        w = self.w
        P = X[..., S_POS]
        V = X[..., S_VEL]
        d = parts["d"]
        J = w.w_goal * d[:, :-1].sum(axis=1)
        J += w.w_term_pos * d[:, -1]
        J += w.w_term_vel * np.square(V[:, -1, :]).sum(axis=1)

        # Obstacle preference (the hard gate is the validator, not this).
        if w.w_obs > 0.0:
            cl = parts["clearance"]
            J += w.w_obs * np.square(
                np.maximum(0.0, self.d_influence - cl)).sum(axis=1)

        # Slew: |nu_k - nu_{k-1}|^2 chained back to the acceleration actually
        # being flown, so the first node is anchored to a_prev.
        if np.any(w.R_dnu > 0.0):
            nu_prev = np.zeros(NNU)
            if a_prev is not None:
                nu_prev[U_ACC] = np.asarray(
                    a_prev, dtype=np.float64).reshape(-1)[:2]
            chain = np.concatenate(
                [np.broadcast_to(nu_prev, (U.shape[0], 1, NNU)), U], axis=1)
            dU = np.diff(chain, axis=1)
            J += (w.R_dnu[None, None, :] * np.square(dU)).sum(axis=(1, 2))

        # Yaw alignment. BACK, and only because the map is a BELIEF.
        #
        # It was removed on the argument that "heading is not a safety quantity
        # -- unknown space is rejected by the hard gate, obstacles are static,
        # so a badly aimed camera stalls progress rather than causing
        # collisions". Every clause of that is true and the conclusion does not
        # follow on a fused map, because THE STALL IS THE FAILURE. The camera
        # is body-fixed: heading decides what gets observed, unobserved space
        # is untraversable, and a vehicle pointing away from where it is going
        # cannot clear the space it is entering. Measured on clutter/2000 with
        # w_yaw = 0, HAA alone: unknown falls 3448 -> 1350 cells in the first
        # 4 s and then essentially stops (1291 at t=5, 985 at t=13) while the
        # speed collapses to 0.03 m/s. The sample pool was healthy throughout
        # -- 116-192 of 192 valid -- so this was never rejection. The vehicle
        # simply stopped looking anywhere new, and therefore stopped having
        # anywhere new to go.
        #
        # `velocity` mode points the camera where the vehicle is going, which
        # is the direction whose free space it actually needs. Below
        # `v_stopped` there is no travel direction to align to, so the
        # reference falls back to the GOAL bearing -- the vehicle turns to look
        # at where it wants to be, which is what breaks the deadlock.
        if getattr(w, 'w_yaw', 0.0) > 0.0:
            psi = X[..., IPSI]
            mode = getattr(w, 'yaw_mode', 'velocity')
            if mode in ('goal', 'goal_in_view'):
                ref = np.arctan2(np.asarray(goal)[1] - P[..., 1],
                                 np.asarray(goal)[0] - P[..., 0])
            else:
                sp = np.hypot(V[..., 0], V[..., 1])
                moving = sp > self._v_stopped
                ref = np.where(
                    moving, np.arctan2(V[..., 1], V[..., 0]),
                    np.arctan2(np.asarray(goal)[1] - P[..., 1],
                               np.asarray(goal)[0] - P[..., 0]))
            e = np.arctan2(np.sin(psi - ref), np.cos(psi - ref))
            if mode == 'goal_in_view':
                # Penalise only a goal OUTSIDE the camera view: no cost while
                # the goal bearing is within GOAL_VIEW_HALF_ANGLE of the
                # heading, the excess angle squared beyond it. Keeps the goal
                # region observed without pinning the heading to it.
                e = np.maximum(0.0, np.abs(e) - GOAL_VIEW_HALF_ANGLE)
            J += w.w_yaw * np.square(e).sum(axis=1)

        if parts.get("frontier") is None:
            return J + self.frontier_cost(X, goal)
        return J + self._frontier_from_classes(parts["frontier"], X.shape[0])

    def _frontier_from_classes(self, cls, K):
        """frontier_cost's weighting of an already-computed class per sample."""
        occ = self.validator.occ
        if self.w_frontier == 0.0 or occ is None or not hasattr(occ, "unsafe"):
            return np.zeros(K)
        self.last_frontier = cls
        c = np.zeros(K)
        c[cls == OCCUPIED] = self.c_occupied
        c[cls == UNKNOWN] = self.c_unknown
        return self.w_frontier * c

    def _project_start(self, xi):
        """Poor-man's tube consistency (paper eq. 16) for the start state.

        `states_ok` checks node 0, so a start state even 1 cm/s
        over the planner's cap kills EVERY sample before scoring -- under a
        real plant (RotorPy, hardware) a tracking overshoot then reads as
        spurious HAA-infeasibility. The braking fallback shares node 0 and dies
        with them, so the planner ends up refusing the manoeuvre that exists to
        restore the limit. The paper makes the nominal initial state a decision
        variable constrained by `x_k in x0_bar (+) Z` (eq. 16); this picks the
        CLOSEST admissible point rather than optimising over it, which
        under-approximates the paper's feasible set -- the safe direction.

        UNCONDITIONAL, AND THAT IS THE CHANGE. It used to clip only within
        `X0_PROJ_BAND` = 15% of the bound and leave anything past that to be
        rejected, so that a 0.5 m/s hand-over to a 0.35 m/s HAA stayed
        infeasible and the supervisor's R_Nr / braking still ran. That made ONE
        constant answer two unrelated questions -- "can this planner produce a
        tick" and "is this state in S_HAA" -- and it answered the second in the
        wrong place: `in_s_haa` has no state gate of its own, so the band was
        the only thing deciding envelope membership on the velocity channel.
        It now gates on `x in X` itself (eq. 33's first condition), which
        leaves this function nothing to protect.

        The 0.15 was never measured. It was a separator between an 11.1%
        tracking overshoot (sup 0.039 against v_max 0.35) and a 43% hand-over
        (0.5 into 0.35), both quoted when `z_vel` was 0 so `lim.v_max` WAS
        0.35. When z_vel became 0.04 the constant kept multiplying `lim.v_max`,
        which is now X_bar = 0.31, so it silently went from 1.35x the overshoot
        it covered to 3.75x -- and started reaching 0.0065 m/s past X itself,
        which is the one place eq. 16 cannot go. See `Config.limits_flown`; this
        was a fourth instance of the failure its docstring lists three of.

        Projection lands EXACTLY on the bound, not inside it. `states_ok`
        compares against `+_TOL` (1e-6), so the boundary passes; an inward
        epsilon would move every clipped start state and buy nothing.

        No-op under IdealPlant, where the state IS the reference and never
        exceeds the bounds.
        """
        from planner.types import PlanarState
        if isinstance(xi, PlanarState):
            xi = xi.to_array()
        xi = np.asarray(xi, dtype=np.float64).reshape(-1).copy()
        lim = self.dyn.lim
        sp = float(np.hypot(xi[2], xi[3]))
        if sp > lim.v_max:
            xi[2:4] *= lim.v_max / sp
        om = float(xi[5])
        if abs(om) > lim.omega_max:
            xi[5] = np.sign(om) * lim.omega_max
        return xi

    def plan(self, state, goal, **kw):
        state = self._project_start(state)
        # Check current map contents every solve. A new field is necessary
        # only when traversability, edge weights or the effective source change.
        if self.use_geodesic and getattr(self.validator, "occ", None) is not None:
            occ = self.validator.occ
            if hasattr(occ, "unsafe"):
                self.ctg = CostToGo(occ, goal, self.validator.r_safe,
                                    unknown_free=self.unknown_free,
                                    cache=self._map_cost_cache)
        return super(FrontierMPPI, self).plan(state, goal, **kw)

    def describe(self):
        base = super(FrontierMPPI, self).describe()
        if self.w_frontier == 0.0:
            return base + "\n  frontier term OFF"
        return (base + "\n  frontier (PA-MPPI): w %.1f  occupied %+.1f  "
                       "unknown %+.1f  -> terminal-node ray to goal"
                % (self.w_frontier, self.c_occupied, self.c_unknown))
