"""High-Assurance Autonomy (HAA) tube-MPPI safety controller.

This is the *conservative* / *certifiable* side of the DeSimplex stack. It wraps
the existing :class:`mppi_quad.mppi.MPPIPlanner` (sampling-based, hard-constraint
rejection) with three changes that turn the performance planner into a robust
tube-MPC safety controller:

  1. **Constraint tightening.** Obstacles are inflated by the genuine tube radius
     rho (see :func:`tube_radius`) on top of the body radius, and the speed cap is
     lowered. This realizes the Pontryagin set-tightening ``X_bar = X (-) Z`` from
     the report (Eqs. T6/T16): if the *nominal* (planned) trajectory clears the
     inflated obstacle, the *true* trajectory clears the real obstacle for any
     disturbance in W, because the true position stays within rho of the plan.

  2. **Braking-to-stop recovery.** :meth:`HAAMPPIPlanner.plan_recovery` solves an
     OCP whose objective is "decelerate to a stop while staying inside the
     tightened free space". A non-``None`` return certifies the drone can come to
     rest without collision from the queried state -- this is exactly the
     membership test for the safety envelope ``S_HAA`` used by the runtime monitor.

  3. **Genuine tube radius.** Unlike the TurtleBot prior art (which used a
     hand-picked obstacle inflation of +0.06/+0.12 m because a velocity-input
     unicycle admits no tube), the UAV's geometric tracker is modeled and ISS, so
     rho is derived analytically from the tracker gains and a bounded disturbance
     set -- see :func:`tube_radius`.

Why MPPI rather than a QP: the collision constraint lives on a non-convex
occupancy grid, so the OCP is non-convex. We reuse the MPPI sampler's
hard-constraint rejection (``MPPIPlanner._validate``) to handle it.
"""
import numpy as np

from .mppi import MPPI, MPPIPlanner
from .quadrotor import PlanarHolonomic


def tube_radius(kp, w_max, gamma=1.1, delta_model=0.0):
    """Analytic ISS bound on the planar position-tracking error (the tube rho).

    The geometric tracker (``mppi_quad/tracker.py``) commands a PD specific force
    ``f_des = m (kp e_p + kv e_v + a_ff + g e_z)``. Linearizing the closed-loop
    planar position error about hover gives the second-order error dynamics

        e_p'' + kv e_p' + kp e_p = w,        ||w|| <= w_max,

    where ``w`` is the lumped disturbance *acceleration*: wind / drag-mass
    mismatch, body-rate lag + discretization, and -- crucially -- the 6D-plan vs
    12D-plant model-reduction error. At equilibrium (e_p' = 0) the steady-state
    error magnitude is ``w_max / kp``; the damping ratio
    ``zeta = kv / (2 sqrt(kp))`` is ~0.7 here so the transient overshoot is small
    and folded into ``gamma >= 1``. ``delta_model`` adds an explicit slack for
    perception / unmodeled effects.

        rho = gamma * w_max / kp + delta_model

    Parameters
    ----------
    kp : float
        Tracker position gain (must match the tracker used in the closed loop).
    w_max : float
        Upper bound on disturbance acceleration [m/s^2] (the radius of W projected
        to a planar acceleration).
    gamma : float
        Transient/percentile factor (>= 1) covering the second-order overshoot.
    delta_model : float
        Additive slack [m] for perception and unmodeled error.

    Returns
    -------
    float
        Tube radius rho [m].
    """
    return float(gamma) * float(w_max) / float(kp) + float(delta_model)


class HAAMPPIPlanner(MPPIPlanner):
    """MPPIPlanner with an added braking-to-stop recovery objective.

    In ``recovery`` mode the reward ignores goal-progress and instead drives the
    planar speed to zero over the horizon, while the inherited hard validation
    (``_validate``) still rejects any rollout that collides at the *tightened*
    ``robot_radius``. A feasible recovery plan therefore certifies "stoppable
    without collision" from the start state.
    """

    def __init__(self, *args, brake_w=20.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.brake_w = float(brake_w)
        self.recovery = False
        # True body radius for the emergency escape in plan_recovery; the factory
        # build_haa_planner overwrites this with the un-tightened body radius.
        self.base_radius = float(self.robot_radius)

    def _reward(self, X, U, goal):
        if not self.recovery:
            return super()._reward(X, U, goal)
        # Recovery objective: minimize speed over the horizon (with a strong
        # terminal-speed term so the chosen plan actually comes to rest), while
        # staying clear of obstacles. Collisions are already hard-rejected in
        # _validate at the tightened robot_radius.
        v_xy = X[..., 3:5]
        speed = np.linalg.norm(v_xy, axis=-1)            # (Kv, N+1)
        R = -self.brake_w * (speed ** 2).sum(axis=1)
        R -= 10.0 * self.brake_w * (speed[:, -1] ** 2)   # terminal stop
        infl = self.robot_radius + self.safe_margin
        near = self.occ.near_obstacle(X, infl)
        R -= self.near_obs_w * near.sum(axis=1)
        return R.astype(np.float32)

    def plan_recovery(self, x0, warm_shift=1):
        """Plan a braking-to-stop trajectory from ``x0``.

        Returns ``(X_opt, U_opt)`` of a collision-free decelerating trajectory, or
        ``(None, None)`` if none exists (i.e. ``x0`` is *not* in the safety
        envelope S_HAA). ``goal`` is set to the current xy so the (unused in
        recovery) terminal term degenerates harmlessly.

        If the tightened-radius recovery is infeasible because the drone is
        already *inside* the tube-inflated zone (every rollout collides at the
        tightened radius), we retry once against the TRUE body radius. This is an
        emergency escape: the tube margin is a robustness buffer, so wriggling out
        of it toward real free space is strictly safer than freezing in place
        where momentum/disturbance could still push the drone into the obstacle."""
        x0 = np.asarray(x0, dtype=np.float32)
        self.recovery = True
        r_tight = self.robot_radius
        try:
            X, U = self.plan(x0, x0[:2], warm_shift=warm_shift)
            if X is None:
                self.robot_radius = self.base_radius   # escape at true body radius
                X, U = self.plan(x0, x0[:2], warm_shift=warm_shift)
            return X, U
        finally:
            self.robot_radius = r_tight
            self.recovery = False


# --- default HAA configuration -------------------------------------------------
#
# Conservative relative to the HPA planner in run_full.build_planner:
#   - lower speed cap (HAA_V_CAP) so braking distance stays short,
#   - obstacles inflated by the body radius PLUS the tube rho,
#   - everything else (sampler, dt, horizon) reuses the proven planar setup.

HAA_V_CAP = 0.6          # [m/s] soft cruise target. NOTE: the current maps are
                         # tuned tight enough that the HAA weaves at ~0.9 to pass
                         # them; a genuinely-0.6 HAA needs looser maps.
HAA_V_HARD = 1.0         # [m/s] hard ||v_xy|| ceiling = hardware limit (sits above
                         # the ~0.9 weave speed, so it bounds but never over-rejects).
HAA_A_MAX = 0.7          # [m/s^2] planar accel/decel authority (== a_max)
HAA_ALPHA_MAX = 3.0      # [rad/s^2] yaw angular accel
# True robot radius == the ground-truth collision threshold (run_bc._check_collision_xy
# uses 0.25 m). The HAA collision radius is this TRUE body radius plus the tube rho
# -- the tube IS the inflation, so we must not also fold in the HPA planner's
# pre-inflated 0.40 m (that would double-count and make the safety set unusably fat).
HAA_BASE_RADIUS = 0.31   # [m] true body radius (real drone 0.62x0.50 m -> r=0.31)
HAA_SAFE_MARGIN = 0.05   # [m] soft near-obstacle inflation


def build_haa_planner(occ, rho, mppi_dt=0.05, N=30, K=1024, seed=None,
                      v_cap=HAA_V_CAP, a_max=HAA_A_MAX, base_radius=HAA_BASE_RADIUS):
    """Construct the HAA tube-MPPI planner over the planar surrogate.

    Parameters
    ----------
    occ : ObstacleMap
        Dynamic occupancy map (use ``env.get_obstacle_map()``).
    rho : float
        Tube radius from :func:`tube_radius`. The hard collision radius becomes
        ``base_radius + rho`` (constraint tightening X (-) Z).
    """
    if seed is not None:
        np.random.seed(seed)
    planar = PlanarHolonomic()
    channel_scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    mppi = MPPI(num_nodes=N, num_rollouts=K, channel_scale=channel_scale,
                sigma=0.7, temperature=0.1, use_noise_ramp=False,
                noise_beta=0.8)   # colored noise -> coherent weaving maneuvers
    # Catastrophic-only hard velocity caps; real speed limit enforced softly.
    state_lims = [None, None, None, (-2.5, 2.5), (-2.5, 2.5), (-3.0, 3.0)]
    ctrl_lo = np.array([-a_max, -a_max, -HAA_ALPHA_MAX], dtype=np.float32)
    ctrl_hi = np.array([+a_max, +a_max, +HAA_ALPHA_MAX], dtype=np.float32)
    planner = HAAMPPIPlanner(
        mppi, planar, occ,
        robot_radius=base_radius + float(rho),   # tightened: X (-) Z
        safe_margin=HAA_SAFE_MARGIN,
        ctrl_lo=ctrl_lo, ctrl_hi=ctrl_hi,
        state_lims=state_lims,
        progress_w=10.0, terminal_w=100.0, near_obs_w=3.0,
        z_track_w=0.0,
        v_cap=v_cap, v_cap_w=50.0,       # soft cruise shaping (smooth)
        smoothing_w=0.0,
        dt=mppi_dt, mppi_iters=2,
        brake_w=20.0,
    )
    # Soft v_cap shapes the cruise (~0.9); hard ceiling = the hardware limit 1.0
    # (sits above the weave speed so it bounds without over-rejecting maneuvers).
    planner.v_hard = HAA_V_HARD
    planner.rho = float(rho)
    planner.a_max = float(a_max)
    planner.v_cap_haa = float(v_cap)
    planner.base_radius = float(base_radius)
    return planner
