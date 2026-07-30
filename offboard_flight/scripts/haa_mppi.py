#!/usr/bin/python
# -*- coding: utf-8 -*-
"""MPPI planner for HAA, over the full 12-state quadrotor model.

Sampling-based solve of the HAA stochastic OCP (DeSimplex eq 17), following the
paper's MPPI instantiation:

  * sample K control sequences by injecting Gaussian noise into a nominal
    sequence (paper: N_s = 1000 over a 40-step horizon)
  * discard samples that violate the hard constraints or hit an inflated
    obstacle -- "Infeasible trajectories violating constraints are discarded.
    If all samples are infeasible, the MPC problem is declared infeasible."
  * weight the survivors by w ~ exp(-S/lambda)   (eq 60)
  * cost shape from eq 59: running goal distance + terminal goal distance +
    obstacle indicator

Three things differ from the paper's quadrotor section by project decision:

  1. The rollout uses the full 12-state dynamics, not the reduced planar
     surrogate (eq 71-73), so z and attitude are real states.

  2. The control is CTBR -- collective thrust and body rates -- following
     PA-MPPI rather than DeSimplex's planar velocity input. Thrust being a
     decision variable means the nominal sequence must start at hover or every
     early rollout falls out of the sky; warm_start()/reset() handle that.
     Note that inertia has no effect under this input (see ThrustRateQuadrotor).

  3. The altitude-hold surrogate (eq 68, z0 = 1 m) is enforced through a cost
     term on z rather than by construction, since z is now a real state.

The planner returns the NOMINAL trajectory. With PX4 tracking position
setpoints, PX4 is the ancillary feedback controller kappa of eq (6), and the
robust margin Z it induces is what haa_obstacles inflates by.
"""

import numpy as np

from haa_dynamics import NX, NU, P, V, RPY


class MPPIResult(object):
    """Outcome of one solve. feasible=False means HAA declares infeasibility."""

    def __init__(self, feasible, X=None, U=None, n_valid=0, n_samples=0,
                 best_cost=float("inf"), reason="", X_viz=None):
        self.feasible = feasible
        self.X = X                  # (N+1, 12) nominal state trajectory
        self.U = U                  # (N, 4)    nominal control sequence
        self.n_valid = n_valid
        self.n_samples = n_samples
        self.best_cost = best_cost
        self.reason = reason
        self.X_viz = X_viz          # (n, N+1, 12) surviving rollouts, for rviz


class HAAMPPI(object):

    def __init__(self, dynamics, grid,
                 horizon=15, num_samples=256, dt=0.1,
                 lam=0.02,
                 sigma_thrust=3.0, sigma_rate_xy=0.35, sigma_rate_z=0.2,
                 w_goal=10.0, w_terminal=100.0, w_obstacle=3.0,
                 w_z=50.0, w_vel=0.5, w_yaw=0.0, w_smooth=1.0,
                 z_hold=1.0, goal_tol=0.25, seed=0):
        """
        dynamics       AttitudeStabilizedQuadrotor (or anything with the same
                       rollout/feasible/clip_control/hover_control interface)
        grid           InflatedGrid (already inflated by r_eff)
        horizon        N steps
        num_samples    K rollouts per solve
        dt             step [s]
        lam            MPPI temperature (DeSimplex 0.1, PA-MPPI 0.02)
        sigma_thrust   std of collective-thrust noise [N]
        sigma_rate_xy  std of roll/pitch rate-command noise [rad/s]
        sigma_rate_z   std of yaw rate-command noise [rad/s]
        w_*            cost weights; w_goal/w_terminal/w_obstacle mirror the
                       10 / 100 / 3 of eq 59
        w_z            altitude-hold weight (this model's stand-in for eq 68)
        z_hold         commanded altitude [m] (paper z0 = 1 m)

        Control space is CTBR -- collective thrust and body rates, u = [c, w] --
        following PA-MPPI (Zhai et al., RA-L 2026), which plans quadrotor
        navigation on an occupancy grid with the same input. See
        ThrustRateQuadrotor, and note that J drops out of the model under this
        parameterisation.

        Defaults follow PA-MPPI where we can afford to: H = 15, dt = 0.1
        (1.5 s horizon), lambda = 0.02. Their N = 17,500 samples we cannot --
        that runs on an A1000 GPU at 50 Hz. Profiled on this Jetson with the
        perception stack up (loadavg ~8 on 6 cores), per-step numpy overhead
        is a fixed charge, so K is comparatively cheap: K = 64 -> 1024 costs
        48 -> 120 ms at N = 20. K = 512 at H = 15 fits the 100 ms budget at
        10 Hz. Raise K before raising H if you have time to spare; moving the
        rollout to CUDA is what would buy a PA-MPPI-scale sample count.

        Collision handling differs from PA-MPPI on purpose: they weight a
        binary indicator by a large constant, we hard-reject. DeSimplex
        requires the discard ("infeasible trajectories are discarded; if all
        samples are infeasible the MPC problem is declared infeasible"),
        because that rejection IS the safety certificate of theorem 1.
        """
        self.dyn = dynamics
        self.grid = grid

        self.N = int(horizon)
        self.K = int(num_samples)
        self.dt = float(dt)
        self.lam = float(lam)

        self.sigma = np.array([sigma_thrust, sigma_rate_xy,
                               sigma_rate_xy, sigma_rate_z], dtype=np.float64)

        self.w_goal = float(w_goal)
        self.w_terminal = float(w_terminal)
        self.w_obstacle = float(w_obstacle)
        self.w_z = float(w_z)
        self.w_vel = float(w_vel)
        self.w_yaw = float(w_yaw)
        self.w_smooth = float(w_smooth)

        self.z_hold = float(z_hold)
        self.goal_tol = float(goal_tol)

        self.rng = np.random.RandomState(seed)
        self.U_nom = None
        self.reset()

    # ----------------------------------------------------------- warm start

    def reset(self):
        """Nominal sequence = hover, whatever that is in the control space.

        In the acceleration parameterisation hover is the zero vector, since
        gravity compensation happens inside the inner loop. Asking the model
        for it rather than assuming zeros keeps this correct if the control
        space changes again."""
        self.U_nom = np.tile(self.dyn.hover_control(), (self.N, 1))

    def warm_start(self, shift=1):
        """Shift the previous solution forward and pad with hover."""
        if self.U_nom is None:
            self.reset()
            return
        if shift <= 0:
            return
        pad = np.tile(self.dyn.hover_control(), (shift, 1))
        self.U_nom = np.concatenate([self.U_nom[shift:], pad], axis=0)

    # ------------------------------------------------------------------ cost

    def _cost(self, X, U, goal):
        """Cost per sample. X (K,N+1,12), U (K,N,4), goal (2,) -> (K,)."""
        pos = X[:, :, P]
        pxy = pos[:, :, :2]
        z = pos[:, :, 2]
        vel = X[:, :, V]
        yaw = X[:, :, RPY][:, :, 2]

        d = np.linalg.norm(pxy - goal[None, None, :], axis=-1)

        # eq 59: running progress + terminal
        c = self.w_goal * d[:, :-1].sum(axis=1)
        c += self.w_terminal * d[:, -1]

        # altitude hold (this model's stand-in for the eq 68 surrogate)
        c += self.w_z * np.square(z - self.z_hold).sum(axis=1)

        # keep it calm: penalise speed and control rate
        if self.w_vel > 0.0:
            c += self.w_vel * np.square(np.linalg.norm(vel, axis=-1)).sum(axis=1)
        if self.w_smooth > 0.0 and self.N > 1:
            dU = np.diff(U, axis=1) / self.sigma[None, None, :]
            c += self.w_smooth * np.square(dU).sum(axis=(1, 2))
        if self.w_yaw > 0.0:
            c += self.w_yaw * np.square(np.arctan2(np.sin(yaw), np.cos(yaw))).sum(axis=1)

        return c

    # ----------------------------------------------------------------- solve

    def plan(self, x0, goal, warm_shift=1, n_viz=0):
        """One MPPI solve from state x0 toward planar goal (gx, gy).

        Returns MPPIResult. feasible=False means every sample was rejected,
        which is the HAA infeasibility signal the safety monitor keys on.

        n_viz > 0 keeps that many surviving rollouts on the result for
        visualisation. They are the cheapest evenly-spaced slice of the valid
        set, not the best ones, so what you see is what MPPI actually explored.
        """
        goal = np.asarray(goal, dtype=np.float64)[:2]
        self.warm_start(warm_shift)

        # sample around the nominal
        noise = self.rng.randn(self.K, self.N, NU) * self.sigma[None, None, :]
        U = self.U_nom[None, :, :] + noise
        U = self.dyn.clip_control(U)

        X = self.dyn.rollout(x0, U, self.dt)

        # --- hard constraints (eq 2) and obstacle avoidance (eq 15) ---
        ok = self.dyn.feasible(X)

        pxy = X[:, :, P][:, :, :2]
        hit = self.grid.collides(pxy[:, :, 0], pxy[:, :, 1])   # (K, N+1)
        ok &= ~hit.any(axis=1)

        n_valid = int(ok.sum())
        if n_valid == 0:
            return MPPIResult(False, n_samples=self.K,
                              reason="all %d samples infeasible" % self.K)

        Uv = U[ok]
        Xv = X[ok]
        S = self._cost(Xv, Uv, goal)

        X_viz = None
        if n_viz > 0:
            step = max(1, Xv.shape[0] // n_viz)
            X_viz = Xv[::step][:n_viz]

        # eq 60, shifted for numerical stability
        Smin = S.min()
        w = np.exp(-(S - Smin) / self.lam)
        wsum = w.sum()
        if not np.isfinite(wsum) or wsum <= 0.0:
            # temperature collapsed; fall back to the single best sample
            best = int(np.argmin(S))
            self.U_nom = Uv[best].copy()
            return MPPIResult(True, Xv[best], Uv[best], n_valid, self.K,
                              float(S[best]), "degenerate weights -> best sample",
                              X_viz)

        w /= wsum
        U_new = np.tensordot(w, Uv, axes=(0, 0))
        U_new = self.dyn.clip_control(U_new)

        # The weighted mean can leave the feasible set even when every sample
        # in the mean was feasible (the set is non-convex once obstacles are in
        # it). Re-check it, and keep the best single sample if it fails.
        X_new = self.dyn.rollout(x0, U_new[None, :, :], self.dt)
        mean_ok = bool(self.dyn.feasible(X_new)[0]) and not bool(
            self.grid.collides(X_new[0, :, 0], X_new[0, :, 1]).any())

        if not mean_ok:
            best = int(np.argmin(S))
            self.U_nom = Uv[best].copy()
            return MPPIResult(True, Xv[best], Uv[best], n_valid, self.K,
                              float(S[best]), "mean infeasible -> best sample",
                              X_viz)

        self.U_nom = U_new
        return MPPIResult(True, X_new[0], U_new, n_valid, self.K,
                          float(S.min()), "", X_viz)

    # ------------------------------------------------------------- utilities

    def at_goal(self, x0, goal):
        d = np.linalg.norm(np.asarray(x0)[0:2] - np.asarray(goal)[:2])
        return bool(d <= self.goal_tol)
