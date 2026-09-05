#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The final safety validator. Independent of the MPPI cost, by construction.

NOT UNDER haa/, AND THAT IS THE POINT
The validator gates whatever flies, not whatever MPPI produced. `hpa` emits an
unverified learned plan and `planner/supervisor.py` gates it through this
same object; if this lived under `haa/` the learned producer would import the
classical one just to be checked. Safety is common ground, so it sits beside
`types.py` at the package root.

THIS IS THE PLANNER'S OWN FILE NOW
It was `vendor/planner/planar_safety.py`; the extraction that sat here unused
differed from it by its own import line and nothing else.

    r_safe = r_Q + r_perc + r_track + d_clr

  r_Q      quadrotor disc footprint radius
  r_perc   perception / reconstruction uncertainty. 0.18 m on this vehicle,
           measured over 41 pillar samples at 0.66-3.17 m: radial |error| p95
           0.157, p99 0.182 (see inflate_grid.py).
  r_track  low-level tracking error -- how far the true state may sit from the
           nominal one while PX4 flies it. This is the robust invariant tube Z:
           because the nominal trajectory clears the set inflated by Z, the true
           trajectory clears the real obstacles.
  d_clr    discretionary clearance on top. Also where the grid's discretisation
           error is bought back (see the note on sweep sampling below).

WHY THIS IS A SEPARATE MODULE FROM THE COST
planar_mppi has an obstacle-clearance term in its cost. That term is a soft
preference: it shapes where samples go, it is traded off against goal progress,
and a large enough goal reward will happily outvote it. This module shares none
of that. It answers one boolean question -- is every point of this swept path at
least r_safe from anything unsafe -- and it is the only thing standing between a
plan and the vehicle. If the two ever disagree, this one wins.

The separation is not stylistic. A validator that reused the cost's notion of
"near an obstacle" would inherit the cost's tuning, and re-tuning w_obs for
better goal-seeking would silently move the safety boundary.

WHY SWEPT, NOT NODAL
At dt = 0.1 s and v_max = 1.5 m/s a node spacing is up to 0.15 m -- three cells.
A trajectory can put both endpoints of a segment in free space with the segment
itself passing straight through a 0.10 m obstacle. Checking nodes only is
checking that a fence has no posts where you looked.
"""

import numpy as np

from planner.types import S_POS


def safe_radius(r_quad=0.31, r_perc=0.18, r_track=0.05, d_clr=0.05):
    """r_safe = r_Q + r_perc + r_track + d_clr.

    Defaults are this vehicle's measured values: r_quad from the existing
    ~robot_radius, r_perc from the pillar campaign, r_track the 5 cm working
    margin, d_clr covering grid discretisation. They sum to 0.59 m, which is
    large relative to a 7.2 x 5.2 m arena -- that is a real property of a 0.31 m
    quadrotor with 0.18 m of depth uncertainty, not a tuning mistake, and it is
    better seen in the number than discovered in a wall.
    """
    return float(r_quad) + float(r_perc) + float(r_track) + float(d_clr)


class PlanarSafetyValidator(object):
    """Swept-path collision validation against an unsafe set."""

    def __init__(self, occ, r_safe, sweep_step=None):
        """
        occ         PlanarOccupancy (or FreeSpace) exposing clearance(x, y)
        r_safe      inflation radius [m], from safe_radius()
        sweep_step  spacing [m] at which segments between consecutive nodes are
                    sampled. Defaults to half the cell size.

                    Half a cell is dense enough that no cell the segment passes
                    through can be stepped over. It is not a proof against a
                    segment clipping the extreme corner of a cell -- for that
                    the residual is bounded by the cell diagonal, which is what
                    d_clr in r_safe is there to absorb. At r_safe = 0.59 m
                    against 0.05 m cells the clearance field varies slowly over
                    a sweep step, so this is comfortable rather than marginal.
        """
        self.occ = occ
        self.r_safe = float(r_safe)
        res = float(getattr(occ, "res", 0.05))
        self.sweep_step = float(sweep_step) if sweep_step else 0.5 * res

    # ------------------------------------------------------------- clearance

    def clearance(self, P):
        """Clearance [m] at each point. P (..., 2) -> (...)."""
        P = np.asarray(P, dtype=np.float64)
        return self.occ.clearance(P[..., 0], P[..., 1])

    def margin(self, P):
        """Signed margin: clearance - r_safe. Negative means unsafe."""
        return self.clearance(P) - self.r_safe

    # ----------------------------------------------------------- node checks

    def nodes_safe(self, P):
        """Per-trajectory nodal check. P (K, n, 2) -> (K,) bool.

        Cheap, and used to reject SAMPLES -- there are hundreds of them and they
        are only candidates. The accepted trajectory always goes through
        path_safe() as well. Never use this as the final gate.
        """
        P = np.asarray(P, dtype=np.float64)
        was_2d = P.ndim == 2
        if was_2d:
            P = P[None, :, :]
        ok = (self.clearance(P) >= self.r_safe).all(axis=1)
        return bool(ok[0]) if was_2d else ok

    # ---------------------------------------------------------- swept checks

    def _sweep_points(self, P):
        """Densify P (K, n, 2) into (K, n-1, m, 2) points along each segment."""
        d = P[:, 1:, :] - P[:, :-1, :]                       # (K, n-1, 2)
        seg = np.linalg.norm(d, axis=-1)
        longest = float(seg.max()) if seg.size else 0.0
        m = int(np.ceil(longest / self.sweep_step)) + 1
        m = max(m, 2)
        s = np.linspace(0.0, 1.0, m)                         # (m,)
        return P[:, :-1, None, :] + s[None, None, :, None] * d[:, :, None, :]

    def paths_safe(self, P):
        """Full swept check. P (K, n, 2) -> (K,) bool, or scalar for (n, 2).

        Every point of every segment between consecutive nodes must have
        clearance >= r_safe. This is the gate.
        """
        P = np.asarray(P, dtype=np.float64)
        was_2d = P.ndim == 2
        if was_2d:
            P = P[None, :, :]
        if P.shape[1] < 2:
            ok = self.clearance(P) >= self.r_safe
            ok = ok.all(axis=1)
            return bool(ok[0]) if was_2d else ok

        pts = self._sweep_points(P)                          # (K, n-1, m, 2)
        ok = (self.clearance(pts) >= self.r_safe).all(axis=(1, 2))
        return bool(ok[0]) if was_2d else ok

    def path_safe(self, P):
        """Single trajectory. P (n, 2) or (n, >=2) -> bool."""
        P = np.asarray(P, dtype=np.float64)
        if P.ndim != 2:
            raise ValueError("path_safe expects (n, 2), got %r" % (P.shape,))
        return bool(self.paths_safe(P[:, :2]))

    def validate_states(self, X):
        """Convenience: swept-validate a (n, 6) planar state trajectory."""
        return self.path_safe(np.asarray(X, dtype=np.float64)[:, S_POS])

    # ----------------------------------------------------------- diagnostics

    def first_violation(self, P):
        """Index of the first unsafe segment of P (n, 2), or -1 if all clear.

        For logging why a candidate was rejected. Returns (segment_index,
        min_clearance) so a caller can say how badly it failed, not just that it
        did.
        """
        P = np.asarray(P, dtype=np.float64)[:, :2]
        if P.shape[0] < 2:
            c = self.clearance(P)
            return (-1, float(c.min())) if (c >= self.r_safe).all() else (0, float(c.min()))
        pts = self._sweep_points(P[None, :, :])[0]           # (n-1, m, 2)
        c = self.clearance(pts)                              # (n-1, m)
        seg_min = c.min(axis=1)
        bad = np.nonzero(seg_min < self.r_safe)[0]
        if bad.size == 0:
            return (-1, float(seg_min.min()))
        return (int(bad[0]), float(seg_min[bad[0]]))

    def describe(self):
        return ("r_safe=%.3f m  sweep_step=%.3f m  map: %s"
                % (self.r_safe, self.sweep_step,
                   self.occ.describe() if hasattr(self.occ, "describe")
                   else type(self.occ).__name__))
