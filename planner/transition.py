#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A closed-form bridge between two reference trajectories, and its gate.

    bridge = bridge_reference(src_node, dst_ref, steps)
    if bridge_ok(bridge, steps, validator, dyn, a_prev):
        fly it, then hand over to whoever produced dst_ref

WHY A BRIDGE AND NOT A BLEND
Switching authority mid-flight means switching the reference the geometric
controller is tracking, and two independently planned trajectories agree about
nothing -- not position, not velocity, not acceleration. Handing the controller
a step in any of those asks it to reject a disturbance the planner invented.
Averaging the two trajectories is not a fix either: the average of two
collision-free paths around opposite sides of a pillar goes through the pillar.

So: keep both trajectories exactly as their planners produced them, and spend a
few steps travelling from one to the other along a curve that MATCHES BOTH ENDS
IN EVERY DERIVATIVE THE REFERENCE CARRIES. The result is a single reference the
controller cannot tell was spliced.

WHICH POLYNOMIAL, AND WHY IT IS DECIDED BY THE DATA
`PlanarReferenceSequence` carries p, v, a for translation and psi, psi_dot for
heading (planar_types.py). Six boundary conditions on translation -> the unique
minimum-jerk quintic; four on heading -> the unique cubic. Neither is a choice
so much as a count: a higher order would need a boundary condition the
reference does not have, and a lower one would drop a continuity the controller
can feel. `SE3Control` differentiates nothing beyond acceleration, so C2 in
translation is exactly enough.

ALTITUDE AND YAW ARE NOT SPECIAL-CASED. This is a fixed-altitude planner: `z`
is added by `PlanarReferenceSequence.lift` after the fact and is constant, so
there is nothing vertical to bridge. Heading is bridged in the same call rather
than by a second planner, on the unwrapped angle so a bridge across +-pi takes
the short way round.
"""

import numpy as np

from planner.types import (IAL, IOM, IPSI, NXI, S_POS, S_VEL, U_ACC,
                          PlanarReferenceSequence, wrap_angle)

__all__ = ["quintic", "cubic", "bridge_reference", "bridge_ok", "drop_first"]


def drop_first(ref):
    """`ref` without its first node. NOT `PlanarReferenceSequence.shifted`.

    `shifted` pads the tail by repeating the last node with velocity and
    acceleration ZEROED, which is right for continuing a stale plan -- it
    coasts to a stop rather than running off the horizon -- and wrong here: the
    tail past a bridge is the destination planner's live trajectory, and
    zeroing its derivatives would put back the discontinuity the bridge just
    removed, one node further on. This shortens instead of padding.
    """
    if ref.n_nodes <= 1:
        return ref
    return PlanarReferenceSequence(ref.p[1:], ref.v[1:], ref.a[1:],
                                   ref.psi[1:], ref.psi_dot[1:], ref.dt)


def quintic(x0, v0, a0, x1, v1, a1, T, n):
    """Sample the quintic with those six boundary conditions at n+1 nodes.

    Returns (x, v, a), each (n+1, dim). `T` is the total duration; node k sits
    at t = k*T/n, so node 0 reproduces the start triple and node n the end
    triple to machine precision -- which is what `test_switching.py` asserts
    rather than trusts.
    """
    x0, v0, a0, x1, v1, a1 = (np.atleast_1d(np.asarray(q, dtype=np.float64))
                              for q in (x0, v0, a0, x1, v1, a1))
    T = float(T)
    if T <= 0.0:
        raise ValueError("transition duration must be positive, got %r" % T)

    # Standard minimum-jerk coefficients. c0..c2 come straight from the start
    # triple; c3..c5 solve the 3x3 for the end triple.
    c0, c1, c2 = x0, v0, 0.5 * a0
    d = x1 - (c0 + c1 * T + c2 * T ** 2)
    dv = v1 - (c1 + 2.0 * c2 * T)
    da = a1 - 2.0 * c2
    T3, T4, T5 = T ** 3, T ** 4, T ** 5
    c3 = (10.0 * d / T3) - (4.0 * dv / (T * T)) + (0.5 * da / T)
    c4 = (-15.0 * d / T4) + (7.0 * dv / T3) - (1.0 * da / (T * T))
    c5 = (6.0 * d / T5) - (3.0 * dv / T4) + (0.5 * da / T3)

    t = np.linspace(0.0, T, int(n) + 1)[:, None]
    x = c0 + c1 * t + c2 * t ** 2 + c3 * t ** 3 + c4 * t ** 4 + c5 * t ** 5
    v = c1 + 2 * c2 * t + 3 * c3 * t ** 2 + 4 * c4 * t ** 3 + 5 * c5 * t ** 4
    a = 2 * c2 + 6 * c3 * t + 12 * c4 * t ** 2 + 20 * c5 * t ** 3
    return x, v, a


def cubic(x0, d0, x1, d1, T, n):
    """Sample the cubic through (value, derivative) at both ends. (x, xdot)."""
    x0, d0, x1, d1 = (np.atleast_1d(np.asarray(q, dtype=np.float64))
                      for q in (x0, d0, x1, d1))
    T = float(T)
    if T <= 0.0:
        raise ValueError("transition duration must be positive, got %r" % T)
    c0, c1 = x0, d0
    c2 = (3.0 * (x1 - x0) / (T * T)) - ((2.0 * d0 + d1) / T)
    c3 = (-2.0 * (x1 - x0) / (T ** 3)) + ((d0 + d1) / (T * T))
    t = np.linspace(0.0, T, int(n) + 1)[:, None]
    return (c0 + c1 * t + c2 * t ** 2 + c3 * t ** 3,
            c1 + 2 * c2 * t + 3 * c3 * t ** 2)


def bridge_reference(src, dst, steps):
    """Splice a `steps`-step bridge from `src` onto `dst`. -> reference, or None.

    `src` is the state to leave from, as the five fields a reference node has:
    `(p, v, a, psi, psi_dot)`. The caller supplies the CURRENTLY COMMANDED
    reference node when it has one and the measured state otherwise -- bridging
    from the command rather than from the estimate is what keeps the join free
    of the tracking error, which is a disturbance the controller is already
    rejecting and must not be asked to reject twice.

    THE ENDPOINT IS TIME-ALIGNED, not nearest. `dst` node `steps` is where the
    destination planner intended the vehicle to be `steps` ticks from now, so
    arriving there means arriving on schedule; joining at node 0 instead would
    ask the bridge to undo the destination's own head start, and joining at the
    nearest node would leave a jump in time that shows up as a jump in
    velocity.

    Returns a reference with the same node count as `dst`: bridge nodes
    0..steps, then `dst` from node steps+1 on. Continuity at the join is exact
    by construction, so the whole thing is one C2 trajectory. None when `dst`
    is too short to give an endpoint, which the caller must treat as "do not
    switch this tick".
    """
    steps = int(steps)
    if steps < 1:
        raise ValueError("transition needs at least one step, got %d" % steps)
    if dst is None or dst.n_nodes < steps + 1:
        return None

    p0, v0, a0, psi0, om0 = src
    T = steps * float(dst.dt)

    p, v, a = quintic(p0, v0, a0,
                      dst.p[steps], dst.v[steps], dst.a[steps], T, steps)
    # Unwrapped target, so a bridge across the +-pi cut takes the short way.
    psi_end = float(psi0) + float(wrap_angle(dst.psi[steps] - psi0))
    psi, om = cubic(psi0, om0, psi_end, dst.psi_dot[steps], T, steps)

    return PlanarReferenceSequence(
        np.concatenate([p, dst.p[steps + 1:]], axis=0),
        np.concatenate([v, dst.v[steps + 1:]], axis=0),
        np.concatenate([a, dst.a[steps + 1:]], axis=0),
        np.concatenate([wrap_angle(psi[:, 0]), dst.psi[steps + 1:]], axis=0),
        np.concatenate([om[:, 0], dst.psi_dot[steps + 1:]], axis=0),
        dst.dt)


def bridge_ok(ref, steps, validator, dyn, a_prev=None):
    """Is the bridge portion of `ref` safe and admissible? -> bool.

    Checked with the SAME utilities the planners are gated by, so a bridge
    cannot be accepted under a weaker rule than the trajectory it replaces:

      * `validator.path_safe` -- the swept obstacle check against the inflated
        map. `r_safe` already carries the body radius, the perception margin
        and the tracking tube, so nothing is re-added here.
      * `dyn.states_ok`  -- v_max and omega_max (C4, C5)
      * `dyn.inputs_ok`  -- a_max_eff, alpha_max and the jerk chain (C1-C3),
        anchored on `a_prev` so the first bridge step is slew-legal against the
        acceleration actually being flown.

    Only nodes 0..steps are checked. The tail past the join belongs to the
    destination planner and was validated when IT was produced; re-checking it
    here would reject a bridge for a fault in someone else's trajectory.
    """
    steps = int(steps)
    if ref is None or ref.n_nodes < steps + 1:
        return False
    p, v = ref.p[:steps + 1], ref.v[:steps + 1]
    a, psi, om = ref.a[:steps + 1], ref.psi[:steps + 1], ref.psi_dot[:steps + 1]
    if not np.isfinite(p).all() or not np.isfinite(v).all():
        return False

    if not bool(validator.path_safe(p)):
        return False

    X = np.zeros((steps + 1, NXI), dtype=np.float64)
    X[:, S_POS], X[:, S_VEL] = p, v
    X[:, IPSI], X[:, IOM] = psi, om
    if not bool(dyn.states_ok(X)):
        return False

    # The input that produces the bridge: its acceleration, and the angular
    # acceleration implied by the heading-rate profile.
    U = np.zeros((steps, 3), dtype=np.float64)
    U[:, U_ACC] = a[:steps]
    U[:, IAL] = np.diff(om) / float(ref.dt)
    return bool(dyn.inputs_ok(U, a_prev=a_prev))
