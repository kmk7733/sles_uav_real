#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Planar double-integrator model with heading, for fixed-altitude planning.

    p_{k+1}     = p_k + dt v_k + 0.5 dt^2 a_k
    v_{k+1}     = v_k + dt a_k
    psi_{k+1}   = wrap(psi_k + dt omega_k + 0.5 dt^2 alpha_k)
    omega_{k+1} = omega_k + dt alpha_k

State xi = [x y vx vy psi omega], input nu = [ax ay alpha], both world-frame.

WHY THIS MODEL
The vehicle is a quadrotor, but at fixed altitude with a working inner loop the
planner does not need roll and pitch as decision variables -- they are the
low-level controller's response to the commanded acceleration, related to it by
a_xy = g tan(theta). Planning in acceleration and enforcing

    |a|_2 <= g tan(theta_max)

is therefore the same constraint set expressed one level up, and it removes the
attitude random-walk that makes moment-sampled rollouts useless (see the note in
haa_dynamics.AttitudeStabilizedQuadrotor: 0.9% of samples survived a 4 s
horizon). It also makes the state trajectory an AFFINE function of the input
sequence, which is what lets the damped update in planar_mppi work -- see
`is_convex_in_input` below.

The half-dt^2 terms are not cosmetic. Explicit Euler on a double integrator
lags position by 0.5 dt^2 a per step, which over a 20-step horizon at dt = 0.1
and 2 m/s^2 is 0.2 m of accumulated error -- comparable to the safety radius the
whole planner is built around.
"""

import numpy as np

from planner.types import (NXI, NNU, IPSI, IOM, IAL, S_POS, S_VEL, U_ACC,
                          wrap_angle)

G = 9.80665

# Slack for limit comparisons. clip_inputs lands values exactly on the boundary
# and float rounding can leave them a few ulp outside; 1e-6 m/s^2 is far below
# anything physical.
_TOL = 1e-6


class PlanarLimits(object):
    """Hard limits. Everything here is a constraint, not a cost term."""

    __slots__ = ("v_max", "a_max", "omega_max", "alpha_max", "tilt_max",
                 "j_max")

    def __init__(self, v_max=1.5, a_max=2.5, omega_max=1.5, alpha_max=3.0,
                 tilt_max=0.5236, j_max=8.0):
        """
        v_max      max planar speed |v|_2 [m/s]. Matches the 1.5 already used
                   by QuadrotorDynamics.
        a_max      max planar acceleration |a|_2 [m/s^2]
        omega_max  max heading rate |omega| [rad/s]
        alpha_max  max heading acceleration |alpha| [rad/s^2]
        tilt_max   max roll/pitch the low-level controller may be asked for
                   [rad]. 0.5236 = 30 deg, as in QuadrotorDynamics.
        j_max      max acceleration slew |a_k - a_{k-1}| / dt [m/s^3]. This is
                   the jerk bound; it is what stops MPPI handing PX4 a step
                   change in attitude demand between consecutive nodes.
        """
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.omega_max = float(omega_max)
        self.alpha_max = float(alpha_max)
        self.tilt_max = float(tilt_max)
        self.j_max = float(j_max)

    @property
    def a_max_tilt(self):
        """Acceleration bound implied by the tilt limit: g tan(theta_max)."""
        return G * np.tan(self.tilt_max)

    @property
    def a_max_eff(self):
        """The binding acceleration bound: min(a_max, g tan(theta_max)).

        Both are real constraints and either can bind. At 30 deg the tilt bound
        is 5.66 m/s^2, so with the default a_max = 2.5 it is a_max that binds --
        but raise a_max and the tilt bound takes over, which is the point of
        computing the min rather than picking one.
        """
        return min(self.a_max, self.a_max_tilt)

    def describe(self):
        return ("v<=%.2f m/s  a<=%.2f m/s^2 (a_max %.2f, g tan %.1fdeg = %.2f)  "
                "omega<=%.2f rad/s  alpha<=%.2f rad/s^2  jerk<=%.1f m/s^3"
                % (self.v_max, self.a_max_eff, self.a_max,
                   np.degrees(self.tilt_max), self.a_max_tilt,
                   self.omega_max, self.alpha_max, self.j_max))


def _as_batch(U):
    """(N,3) -> (1,N,3) so the sequential passes have one code path."""
    U = np.asarray(U, dtype=np.float64)
    if U.ndim == 2:
        return U[None, :, :], True
    if U.ndim != 3:
        raise ValueError("input sequence must be (N,3) or (K,N,3), got %r"
                         % (U.shape,))
    return U, False


class PlanarDynamics(object):
    """Vectorised rollout, input clipping and constraint checking."""

    def __init__(self, limits=None, dt=0.1):
        self.lim = limits if limits is not None else PlanarLimits()
        self.dt = float(dt)

    # ------------------------------------------------------------- integrate

    def step(self, XI, NU):
        """One step. XI (...,6), NU (...,3) -> (...,6)."""
        XI = np.asarray(XI, dtype=np.float64)
        NU = np.asarray(NU, dtype=np.float64)
        dt = self.dt
        half = 0.5 * dt * dt

        p = XI[..., S_POS]
        v = XI[..., S_VEL]
        psi = XI[..., IPSI]
        om = XI[..., IOM]

        a = NU[..., U_ACC]
        al = NU[..., IAL]

        p_new = p + dt * v + half * a
        v_new = v + dt * a
        psi_new = wrap_angle(psi + dt * om + half * al)
        om_new = om + dt * al

        return np.concatenate([p_new, v_new,
                               psi_new[..., None], om_new[..., None]], axis=-1)

    def rollout(self, xi0, U):
        """Roll K input sequences forward from a shared initial state.

        xi0 (6,) or (K,6), U (K,N,3) or (N,3) -> X (K,N+1,6).
        """
        U, was_2d = _as_batch(U)
        K, N = U.shape[0], U.shape[1]
        X = np.empty((K, N + 1, NXI), dtype=np.float64)
        X[:, 0, :] = np.asarray(xi0, dtype=np.float64)
        for k in range(N):
            X[:, k + 1, :] = self.step(X[:, k, :], U[:, k, :])
        return X

    # ------------------------------------------------------------- clipping

    def clip_inputs(self, U, a_prev=None):
        """Project an input sequence onto the input constraint set.

        Applies, in this order:
          1. |alpha| <= alpha_max                (box)
          2. |a|_2   <= a_max_eff                (disc)
          3. |a_k - a_{k-1}|_2 <= j_max dt       (chain of discs)

        The order matters and is not arbitrary. Step 3 projects a_k onto a disc
        centred on the already-projected a_{k-1}; because that centre is itself
        inside the a_max_eff disc and the result lies on the segment between two
        points of that disc, step 3 cannot undo step 2. Doing it the other way
        round would break the slew bound, and an iterate-to-convergence scheme
        would be needed instead.

        a_prev is the acceleration currently being flown (the previous plan's
        first input). It anchors the first slew constraint; without it MPPI is
        free to demand a step change on the very node the controller executes
        next. Defaults to zero.
        """
        U, was_2d = _as_batch(U)
        U = U.copy()
        K, N = U.shape[0], U.shape[1]

        lim = self.lim
        U[..., IAL] = np.clip(U[..., IAL], -lim.alpha_max, lim.alpha_max)

        a = U[..., U_ACC]
        n = np.linalg.norm(a, axis=-1, keepdims=True)
        a *= np.minimum(1.0, lim.a_max_eff / np.maximum(n, 1e-12))

        if a_prev is None:
            prev = np.zeros((K, 2), dtype=np.float64)
        else:
            prev = np.broadcast_to(
                np.asarray(a_prev, dtype=np.float64).reshape(-1)[:2],
                (K, 2)).copy()
            pn = np.linalg.norm(prev, axis=-1, keepdims=True)
            prev *= np.minimum(1.0, lim.a_max_eff / np.maximum(pn, 1e-12))

        dmax = lim.j_max * self.dt
        for k in range(N):
            d = U[:, k, U_ACC] - prev
            dn = np.linalg.norm(d, axis=-1, keepdims=True)
            scale = np.minimum(1.0, dmax / np.maximum(dn, 1e-12))
            U[:, k, U_ACC] = prev + d * scale
            prev = U[:, k, U_ACC]

        return U[0] if was_2d else U

    # ---------------------------------------------------------- feasibility

    def inputs_ok(self, U, a_prev=None):
        """Per-sample input-constraint check. U (K,N,3) -> (K,) bool."""
        U, was_2d = _as_batch(U)
        K, N = U.shape[0], U.shape[1]
        lim = self.lim

        ok = np.isfinite(U).all(axis=(1, 2))
        ok &= (np.abs(U[..., IAL]) <= lim.alpha_max + _TOL).all(axis=1)

        an = np.linalg.norm(U[..., U_ACC], axis=-1)
        ok &= (an <= lim.a_max_eff + _TOL).all(axis=1)

        if a_prev is None:
            prev = np.zeros((K, 1, 2), dtype=np.float64)
        else:
            prev = np.broadcast_to(
                np.asarray(a_prev, dtype=np.float64).reshape(-1)[:2],
                (K, 2))[:, None, :]
        chain = np.concatenate([prev, U[..., U_ACC]], axis=1)      # (K, N+1, 2)
        slew = np.linalg.norm(np.diff(chain, axis=1), axis=-1)     # (K, N)
        ok &= (slew <= lim.j_max * self.dt + _TOL).all(axis=1)

        return ok[0] if was_2d else ok

    def states_ok(self, X):
        """Per-sample state-constraint check. X (K,N+1,6) -> (K,) bool.

        Velocity and heading rate are STATES, so they cannot be clipped at
        sample time the way inputs can -- they are enforced by rejection, which
        is what the architecture calls for.

        KNOWN DEFECT: NODE 0 IS CHECKED, AND NODE 0 IS NOT A PLAN.
        `.all(axis=1)` runs over every node including X[:, 0], which is the
        MEASURED CURRENT STATE -- where the vehicle actually is, not something
        this planner proposed. Every sample shares it, and so does the braking
        fallback, so a true speed a hair over `v_max` rejects the entire batch
        and `plan()` returns FAILED on an empty map. The planner refuses the
        manoeuvre that exists to restore the limit, on the grounds that the
        limit is violated. (`supervisor.recovery` declines to check state
        bounds for exactly this reason; see its docstring.)

        NOT FIXED HERE, AND MITIGATED INSTEAD, because the fix is not obviously
        `[1:]`: node 0 being out of bounds is real information, and silently
        accepting it would let a plan start from a state the rest of the
        argument assumes cannot occur.

        WHAT THIS REJECTS AT IS `self.lim`, WHICH FOR THE HAA IS X_bar. An
        earlier version of this note said the gate was at 0.35, the physical
        v_max; it is not. `build_planner` hands the planner `limits_flown`, so
        with `safety.z_vel` = 0.04 the rejection is at 0.31 and the measured
        worst true speed of 0.3225 lands OUTSIDE it. That is not a defect in
        the tightening -- it is why `FrontierMPPI._project_start` exists, and
        the 0.04 is the room the true state has before it leaves X itself.

        MEASURED, 31 held-out episodes, the HAA arm's true speed:

            planned cap                  0.310 m/s
            worst true speed             0.3225        (over cap by 0.0125)
            headroom to the 0.35 gate    0.0275        (8%)
            episodes rejected            0 / 31

        THE YAW CHANNEL IS NO LONGER THE THIN ONE. An earlier version of this
        note said "worst 0.4488 against omega_max 0.471, and `safety.z_omega`
        is 0"; both of those numbers have moved. `limits.omega_max` is 0.5236
        (30 deg/s, the autopilot's own MPC_YAWRAUTO_MAX) and `z_omega` is
        0.0349, so this rejects at 0.4887 -- 28.0 deg/s, with 2.0 deg/s of tube
        behind it, which is 2.3x the measured yaw-rate tracking error sup of
        0.0151 rad/s. Measured at the OLD 27 deg/s envelope the HAA arm's true
        rate peaked at 20.2 deg/s with a p99 of 13.8, so this is not a cap the
        planner approaches. What hit the old bound was the mode SWITCH: 28 of
        178 785 DeSimplex samples, all within 0.14 s of one.
        """
        X = np.asarray(X, dtype=np.float64)
        was_2d = X.ndim == 2
        if was_2d:
            X = X[None, :, :]
        lim = self.lim

        ok = np.isfinite(X).all(axis=(1, 2))
        sp = np.linalg.norm(X[..., S_VEL], axis=-1)
        ok &= (sp <= lim.v_max + _TOL).all(axis=1)
        ok &= (np.abs(X[..., IOM]) <= lim.omega_max + _TOL).all(axis=1)

        return ok[0] if was_2d else ok

    # -------------------------------------------------------------- braking

    def brake_to_hover(self, xi0, a_prev=None, horizon=None):
        """Input sequence that brings the vehicle to a stationary hover.

        Each step asks for the acceleration that would zero the velocity in one
        dt, then lets clip_inputs cut it down to what the acceleration and slew
        limits actually allow. Building it through the real clipping path rather
        than by formula means the result is feasible by construction -- it still
        gets validated by the caller, but it is not a trajectory that was only
        ever checked against the model that generated it.

        Returns U (N, 3). Not guaranteed collision-free: the caller must
        validate it, and that is exactly what planar_mppi does.
        """
        N = int(self.horizon_default if horizon is None else horizon)
        xi = np.asarray(xi0, dtype=np.float64).reshape(1, NXI).copy()
        prev = (np.zeros(2) if a_prev is None
                else np.asarray(a_prev, dtype=np.float64).reshape(-1)[:2].copy())

        U = np.zeros((N, NNU), dtype=np.float64)
        for k in range(N):
            want = np.empty(NNU)
            want[U_ACC] = -xi[0, S_VEL] / self.dt
            want[IAL] = -xi[0, IOM] / self.dt
            u = self.clip_inputs(want[None, :], a_prev=prev)[0]
            U[k] = u
            prev = u[U_ACC]
            xi = self.step(xi, u[None, :])
        return U

    horizon_default = 20

    # ------------------------------------------------------------ properties

    @property
    def is_convex_in_input(self):
        """The dynamics constraints are convex in the input sequence.

        X is affine in U for fixed xi0 (the model is a linear system), so
        |v|<=v_max and |omega|<=omega_max are convex constraints on U, as are
        the input disc and slew bounds. Consequence, and the reason planar_mppi
        can backtrack on beta: any convex combination of two constraint-feasible
        input sequences is itself constraint-feasible. COLLISION avoidance is
        the one constraint that is NOT convex, so it is the only thing the
        damped update can break -- and the only thing the final validator has to
        re-check from scratch.
        """
        return True
