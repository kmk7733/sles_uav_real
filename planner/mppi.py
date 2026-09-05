#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Fixed-altitude planar MPPI, with a validated update.

    2D map -> planar MPPI -> validated planar trajectory -> fixed-altitude 3D
    reference -> existing low-level controller

No ROS, no PX4, no simulator types. The planner takes a PlanarState, a planar
goal and a PlanarOccupancy, and returns an MPPIResult carrying a
PlanarReferenceSequence. Lifting to 3D is one call on that sequence.

THE UPDATE IS NOT ASSUMED SAFE
Standard MPPI weights every sampled sequence and takes the weighted mean. That
step is where a sampling planner quietly loses its safety argument: the
collision-free set is NOT convex, so the mean of a hundred collision-free
sequences can pass straight through the obstacle they all went around --
half the samples dodge left, half dodge right, and the average splits the
difference through the middle. The classic case is a single pillar between the
vehicle and the goal.

Everything else in the dynamics is fine under averaging. X is affine in U, so
the velocity, heading-rate, acceleration and slew constraints are all convex in
U and are preserved exactly by a convex combination (see
PlanarDynamics.is_convex_in_input). Collision is the sole nonconvex constraint,
and therefore the sole thing the update can break.

So the update is a CANDIDATE, and the chain below decides what actually flies:

    1. roll out the weighted update, validate the full swept path
    2. if invalid, backtrack:  U_old + beta dU,  beta = 1, 1/2, 1/4, ...
       and take the largest valid beta. beta -> 0 recovers the previous
       nominal, which was validated last cycle, so the search is over a
       segment with a known-good end.
    3. if no beta works, take the lowest-cost sampled trajectory that passes
       swept validation. Samples were only checked at nodes, so this re-checks.
    4. if no sample survives, continue the previously validated plan
    5. if there is no previous plan, brake to hover and validate that
    6. if even braking is unsafe, return FAILED and let the supervisor act

Steps 1-2 are cheap; each is one rollout of one sequence. Step 3 costs one
swept check per sample tried, capped by ~max_sample_sweeps.
"""

import numpy as np

from planner.types import (NNU, IPSI, S_POS, S_VEL, U_ACC,
                          PlanarState, PlanarReferenceSequence, MPPIResult,
                          PlannerStatus, wrap_angle)
from planner.dynamics import PlanarDynamics, PlanarLimits


class PlanarCostWeights(object):
    """Cost weights. Deliberately few.

    The architecture puts hard limits in the constraint set and safety in the
    validator, so the cost only has to express PREFERENCE: get to the goal,
    arrive stopped, stay off the walls, do not thrash the controller. Piling on
    overlapping penalty terms is what makes an MPPI planner untunable -- every
    weight then trades against every other and none of them mean anything on
    their own.
    """

    __slots__ = ("w_goal", "w_term_pos", "w_term_vel", "w_obs", "d_influence",
                 "R_dnu", "w_yaw", "yaw_mode")

    def __init__(self, w_goal=1.0, w_term_pos=10.0, w_term_vel=2.0,
                 w_obs=20.0, d_influence=None, R_dnu=(1.0, 1.0, 0.2),
                 w_yaw=0.0, yaw_mode="velocity"):
        """
        w_goal       running planar distance-to-goal weight
        w_term_pos   terminal distance-to-goal weight
        w_term_vel   terminal |v|^2 weight -- this is what makes the plan arrive
                     stopped instead of sailing through the goal at v_max
        w_obs        clearance-cost weight (soft; the validator is the hard gate)
        d_influence  clearance below which the obstacle cost starts biting [m].
                     Defaults to r_safe + 0.35 so the cost shapes the approach
                     well before the validator would reject it outright.
        R_dnu        per-channel weight on |nu_k - nu_{k-1}|^2, (ax, ay, alpha)
        w_yaw        heading-alignment weight. 0 disables it, which is the
                     default: with a forward-facing stereo pair you usually DO
                     want yaw to follow the path, but turning it on couples
                     heading to the translation solution, so it is opt-in.
        yaw_mode     'velocity' | 'goal' | 'hold' -- what psi should align to
        """
        self.w_goal = float(w_goal)
        self.w_term_pos = float(w_term_pos)
        self.w_term_vel = float(w_term_vel)
        self.w_obs = float(w_obs)
        self.d_influence = d_influence
        self.R_dnu = np.asarray(R_dnu, dtype=np.float64).reshape(NNU)
        self.w_yaw = float(w_yaw)
        self.yaw_mode = str(yaw_mode)


class PlanarMPPI(object):
    """Vectorised planar MPPI with a validated, damped update."""

    def __init__(self, dynamics, validator, weights=None,
                 horizon=20, num_samples=512,
                 sigma=(1.2, 1.2, 1.5), temperature=1.0, seed=0,
                 cost_normalise=True, beta_min=1.0 / 64.0,
                 goal_tol=0.25, max_sample_sweeps=24):
        """
        dynamics          PlanarDynamics
        validator         PlanarSafetyValidator -- the hard gate
        horizon           N control intervals (N+1 nodes)
        num_samples       K sampled sequences per solve
        sigma             per-channel noise std (ax, ay, alpha)
        temperature       MPPI lambda. With cost_normalise it is dimensionless,
                          in units of the batch cost standard deviation, so 1.0
                          is a sane default and stays sane when the weights or
                          the map scale change.
        cost_normalise    divide (S - S_min) by the batch std before the
                          softmax. Without this the temperature is absolute, and
                          against an O(100) cost range any small lambda makes
                          exp(-dS/lambda) collapse onto the single best sample:
                          the "weighted average" becomes an argmin over a fresh
                          random batch every tick, which re-decides left-vs-right
                          obstacle avoidance at 10 Hz. That failure is silent --
                          it looks like a working planner that dithers.
        beta_min          smallest damping factor tried before giving up on the
                          weighted update
        max_sample_sweeps how many sampled trajectories, in cost order, get a
                          full swept validation in the step-3 fallback
        """
        self.dyn = dynamics
        self.validator = validator
        self.w = weights if weights is not None else PlanarCostWeights()

        self.N = int(horizon)
        self.K = int(num_samples)
        self.sigma = np.asarray(sigma, dtype=np.float64).reshape(NNU)
        self.temperature = float(temperature)
        self.cost_normalise = bool(cost_normalise)
        self.beta_min = float(beta_min)
        self.goal_tol = float(goal_tol)
        self.max_sample_sweeps = int(max_sample_sweeps)

        if self.w.d_influence is None:
            self.d_influence = validator.r_safe + 0.35
        else:
            self.d_influence = float(self.w.d_influence)

        self.rng = np.random.RandomState(seed)
        self.U_nom = None
        self._last_ref = None
        self.reset()

    # ----------------------------------------------------------- warm start

    def reset(self):
        """Nominal = zero input. In this parameterisation that IS hover.

        Unlike the thrust-input model, zero here means "no acceleration", not
        "no lift" -- gravity is the low-level controller's problem. So a fresh
        nominal coasts rather than falls, and no hover-thrust bootstrap is
        needed.
        """
        self.U_nom = np.zeros((self.N, NNU), dtype=np.float64)
        self._last_ref = None

    def warm_start(self, shift=1):
        """Shift the accepted sequence forward and pad with zero input."""
        if self.U_nom is None or self.U_nom.shape != (self.N, NNU):
            self.reset()
            return
        shift = int(shift)
        if shift <= 0:
            return
        if shift >= self.N:
            self.U_nom = np.zeros((self.N, NNU), dtype=np.float64)
            return
        self.U_nom = np.concatenate(
            [self.U_nom[shift:], np.zeros((shift, NNU))], axis=0)

    # ------------------------------------------------------------------ cost

    def _yaw_reference(self, X, goal):
        """Desired heading per node, per yaw_mode. X (K,n,6) -> (K,n)."""
        mode = self.w.yaw_mode
        psi = X[..., IPSI]
        if mode == "goal":
            d = goal[None, None, :] - X[..., S_POS]
            return np.arctan2(d[..., 1], d[..., 0])
        if mode == "velocity":
            v = X[..., S_VEL]
            sp = np.linalg.norm(v, axis=-1)
            # Below a threshold the velocity direction is noise, so align to the
            # current heading instead -- a zero-error term, not a random target.
            return np.where(sp > 0.15, np.arctan2(v[..., 1], v[..., 0]), psi)
        if mode == "hold":
            return np.broadcast_to(psi[:, :1], psi.shape)
        return psi

    def _cost(self, X, U, goal, a_prev):
        """J per sample. X (K,n,6), U (K,N,3), goal (2,) -> (K,)."""
        w = self.w
        P = X[..., S_POS]
        V = X[..., S_VEL]

        d = np.linalg.norm(P - goal[None, None, :], axis=-1)     # (K, n)

        J = w.w_goal * d[:, :-1].sum(axis=1)
        J += w.w_term_pos * d[:, -1]
        J += w.w_term_vel * np.square(V[:, -1, :]).sum(axis=1)

        if w.w_obs > 0.0:
            cl = self.validator.clearance(P)
            J += w.w_obs * np.square(
                np.maximum(0.0, self.d_influence - cl)).sum(axis=1)

        if np.any(w.R_dnu > 0.0):
            nu_prev = np.zeros(NNU)
            if a_prev is not None:
                nu_prev[U_ACC] = np.asarray(a_prev, dtype=np.float64).reshape(-1)[:2]
            chain = np.concatenate(
                [np.broadcast_to(nu_prev, (U.shape[0], 1, NNU)), U], axis=1)
            dU = np.diff(chain, axis=1)
            J += (w.R_dnu[None, None, :] * np.square(dU)).sum(axis=(1, 2))

        if w.w_yaw > 0.0:
            err = wrap_angle(X[..., IPSI] - self._yaw_reference(X, goal))
            J += w.w_yaw * np.square(err).sum(axis=1)

        return J

    # -------------------------------------------------------------- helpers

    def _accept(self, status, U, X, cost, n_valid, beta, reason, X_viz):
        ref = PlanarReferenceSequence.from_rollout(X, U, self.dyn.dt)
        self.U_nom = np.array(U, dtype=np.float64, copy=True)
        self._last_ref = ref
        return MPPIResult(status, ref, U, X, float(cost), n_valid, self.K,
                          beta, reason, X_viz)

    def _candidate_ok(self, U, X, a_prev):
        """Full acceptance test for ONE sequence: constraints + swept path."""
        if not bool(self.dyn.inputs_ok(U, a_prev=a_prev)):
            return False
        if not bool(self.dyn.states_ok(X)):
            return False
        return self.validator.validate_states(X)

    def _fallback_previous(self, n_valid, reason):
        """Continue the last validated plan, re-checked against today's map."""
        if self._last_ref is None:
            return None
        ref = self._last_ref.shifted(1)
        if not self.validator.path_safe(ref.p):
            return None
        self._last_ref = ref
        # Bias the next search toward slowing down rather than resuming a
        # nominal that just failed to produce anything valid.
        self.U_nom = np.zeros((self.N, NNU), dtype=np.float64)
        return MPPIResult(PlannerStatus.PREVIOUS, ref, None, None,
                          float("inf"), n_valid, self.K, 0.0, reason, None)

    def _fallback_braking(self, xi0, a_prev, n_valid, reason):
        U_b = self.dyn.brake_to_hover(xi0, a_prev=a_prev, horizon=self.N)
        X_b = self.dyn.rollout(xi0, U_b)[0]
        if not self._candidate_ok(U_b, X_b, a_prev):
            return None
        return self._accept(PlannerStatus.BRAKING, U_b, X_b, float("inf"),
                            n_valid, 0.0, reason, None)

    # ----------------------------------------------------------------- solve

    def plan(self, state, goal, a_prev=None, warm_shift=1, n_viz=0):
        """One solve. Returns MPPIResult; check .ok before using .reference.

        state   PlanarState or a (6,) array
        goal    planar (gx, gy)
        a_prev  the acceleration currently being flown, (2,). Anchors the slew
                constraint and the input-change cost to reality rather than to
                the start of the horizon.
        """
        xi0 = (state.to_array() if isinstance(state, PlanarState)
               else np.asarray(state, dtype=np.float64).reshape(-1))
        goal = np.asarray(goal, dtype=np.float64).reshape(-1)[:2]

        self.warm_start(warm_shift)
        U_nom = self.U_nom

        # --- sample ------------------------------------------------------
        noise = self.rng.randn(self.K, self.N, NNU) * self.sigma[None, None, :]
        U = U_nom[None, :, :] + noise
        # Sample 0 IS the nominal. It costs one slot and guarantees the pool is
        # never worse than last cycle's answer, which matters for the step-3
        # fallback: there is always at least one candidate that was valid once.
        U[0] = U_nom
        U = self.dyn.clip_inputs(U, a_prev=a_prev)

        X = self.dyn.rollout(xi0, U)

        # --- reject ------------------------------------------------------
        valid = self.dyn.states_ok(X)
        valid &= self.dyn.inputs_ok(U, a_prev=a_prev)
        valid &= self.validator.nodes_safe(X[..., S_POS])
        n_valid = int(valid.sum())

        X_viz = None
        if n_viz > 0 and n_valid > 0:
            Xv = X[valid]
            step = max(1, Xv.shape[0] // n_viz)
            X_viz = Xv[::step][:n_viz]

        if n_valid == 0:
            reason = "all %d samples rejected" % self.K
            r = self._fallback_previous(0, reason)
            if r is not None:
                return r
            r = self._fallback_braking(xi0, a_prev, 0, reason)
            if r is not None:
                return r
            return MPPIResult(PlannerStatus.FAILED, None, None, None,
                              float("inf"), 0, self.K, 0.0,
                              reason + "; braking also unsafe", X_viz)

        # --- weight ------------------------------------------------------
        S = self._cost(X, U, goal, a_prev)
        S = np.where(valid, S, np.inf)
        Sv = S[valid]
        Smin = float(Sv.min())

        z = S - Smin
        if self.cost_normalise:
            std = float(Sv.std())
            z = z / max(std, 1e-9)
        wts = np.where(valid, np.exp(-z / max(self.temperature, 1e-9)), 0.0)
        wsum = float(wts.sum())
        if not np.isfinite(wsum) or wsum <= 0.0:
            wts = np.zeros(self.K)
            wts[int(np.argmin(S))] = 1.0
            wsum = 1.0
        wts = wts / wsum

        dU = np.tensordot(wts, U - U_nom[None, :, :], axes=(0, 0))   # (N, 3)

        # --- 1 & 2: validated, damped update -----------------------------
        beta = 1.0
        while beta >= self.beta_min:
            U_c = self.dyn.clip_inputs(U_nom + beta * dU, a_prev=a_prev)
            X_c = self.dyn.rollout(xi0, U_c)[0]
            if self._candidate_ok(U_c, X_c, a_prev):
                status = (PlannerStatus.WEIGHTED if beta >= 1.0
                          else PlannerStatus.DAMPED)
                cost = float(self._cost(X_c[None], U_c[None], goal, a_prev)[0])
                reason = "" if beta >= 1.0 else "damped to beta=%.4g" % beta
                return self._accept(status, U_c, X_c, cost, n_valid, beta,
                                    reason, X_viz)
            beta *= 0.5

        # --- 3: best valid sample, swept-validated -----------------------
        order = np.argsort(S, kind="stable")
        tried = 0
        for idx in order:
            if not valid[idx] or tried >= self.max_sample_sweeps:
                break
            tried += 1
            if self.validator.validate_states(X[idx]):
                return self._accept(
                    PlannerStatus.BEST_SAMPLE, U[idx], X[idx], float(S[idx]),
                    n_valid, 0.0,
                    "weighted update unsafe at every beta; used sample %d "
                    "of %d swept-checked" % (tried, self.max_sample_sweeps),
                    X_viz)

        # --- 4 & 5: previous plan, then braking --------------------------
        reason = ("no sample survived swept validation (%d of %d tried)"
                  % (tried, n_valid))
        r = self._fallback_previous(n_valid, reason)
        if r is not None:
            return r
        r = self._fallback_braking(xi0, a_prev, n_valid, reason)
        if r is not None:
            return r

        # --- 6: nothing is valid -----------------------------------------
        return MPPIResult(PlannerStatus.FAILED, None, None, None, float("inf"),
                          n_valid, self.K, 0.0,
                          reason + "; no previous plan and braking unsafe",
                          X_viz)

    # ------------------------------------------------------------- utilities

    def at_goal(self, state, goal):
        p = (state.position if isinstance(state, PlanarState)
             else np.asarray(state, dtype=np.float64)[:2])
        return bool(np.linalg.norm(p - np.asarray(goal)[:2]) <= self.goal_tol)

    def describe(self):
        return ("PlanarMPPI  N=%d dt=%.3f (%.1f s)  K=%d  sigma=%s  "
                "lambda=%.3g%s\n  %s\n  %s"
                % (self.N, self.dyn.dt, self.N * self.dyn.dt, self.K,
                   np.round(self.sigma, 3).tolist(), self.temperature,
                   " (normalised)" if self.cost_normalise else " (absolute)",
                   self.dyn.lim.describe(), self.validator.describe()))


def build_planner(occ, r_safe, horizon=20, num_samples=512, dt=0.1,
                  limits=None, weights=None, **kw):
    """Convenience wiring: occupancy -> dynamics + validator -> planner."""
    from planner.safety import PlanarSafetyValidator
    dyn = PlanarDynamics(limits if limits is not None else PlanarLimits(), dt)
    val = PlanarSafetyValidator(occ, r_safe)
    return PlanarMPPI(dyn, val, weights=weights, horizon=horizon,
                      num_samples=num_samples, **kw)
