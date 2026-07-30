"""MPPI core (sample / softmax-update) + MPPIPlanner.plan, ported from
/home/acrl/research/Turtlebot3_sles/planner_haa_only (lines 25-90, 598-726)
to plain numpy.

Key design notes carried over verbatim from the reference:
  - Always keep the previous nominal as rollout #0 (concatenate it before
    the noisy samples).
  - Per-channel adaptive noise scale (sigma * channel_scale[c]), with a
    minimum of 0.1*sigma. Optional time-ramp from 1/N up to 1.
  - Softmax weighting around min(costs); temperature controls sharpness.
  - Hard validation (state limits + collision); only valid samples are
    weighted into the update. If zero are valid, plan() returns (None, None)
    and the caller should halt.
  - Rewards (higher = better):
        -progress_w * sum_t ||p_t - goal_xy||
        -terminal_w * ||p_T - goal||
        -near_w     * sum_t [near_obstacle(p_t)]
        -smooth_w   * sum_t ||u_{t+1} - u_t||^2
"""
import numpy as np


class MPPI:
    def __init__(self, num_nodes, num_rollouts, channel_scale,
                 sigma=1.0, temperature=0.1,
                 use_noise_ramp=False, noise_ramp=1.0,
                 min_noise_frac=0.1, noise_beta=0.0, cost_norm=False):
        self.N = int(num_nodes)
        self.K = int(num_rollouts)
        self.channel_scale = np.asarray(channel_scale, dtype=np.float32)
        self.nu = self.channel_scale.shape[0]
        self.sigma = float(sigma)
        self.temperature = float(temperature)
        # cost_norm: divide (cost - min) by the batch std before the softmax,
        # making `temperature` dimensionless (units of cost std). Without it the
        # temperature is absolute: against O(100+) cost ranges the softmax
        # degenerates to argmin over a fresh random batch each tick -- a lottery
        # that re-decides left-vs-right avoidance every replan.
        self.cost_norm = bool(cost_norm)
        self.use_ramp = bool(use_noise_ramp)
        self.noise_ramp = float(noise_ramp)
        self.min_noise_frac = float(min_noise_frac)
        # Temporal correlation of the control noise (AR(1) coefficient in [0,1)).
        # 0 = white noise (iid per step); higher = low-frequency "colored" noise
        # that yields coherent multi-step maneuvers — far better at committing to
        # a lateral detour / threading a gap than white noise (log-MPPI / colored-
        # noise MPPI). See sample().
        self.noise_beta = float(noise_beta)

    def _sigma_block(self):
        # (N, nu) per-step per-channel std
        sig = np.maximum(self.channel_scale * self.sigma,
                         self.min_noise_frac * self.sigma)
        sig = np.broadcast_to(sig, (self.N, self.nu)).astype(np.float32)
        if self.use_ramp:
            ramp = self.noise_ramp * np.linspace(
                1.0 / self.N, 1.0, self.N, endpoint=True, dtype=np.float32)
            sig = ramp[:, None] * sig
        return sig

    def sample(self, U_bar):
        """U_bar: (N, nu) -> (K, N, nu) with index 0 == U_bar (verbatim nominal)."""
        sig = self._sigma_block()                              # (N, nu)
        eps = np.random.randn(self.K - 1, self.N, self.nu).astype(np.float32)
        if self.noise_beta > 0.0:
            # AR(1) low-pass along the time axis -> temporally-correlated
            # ("colored") noise with unit marginal variance. Coherent multi-step
            # pushes let rollouts commit to a lateral detour from standstill and
            # thread tight gaps, where white noise stalls.
            b = self.noise_beta
            scale = np.sqrt(1.0 - b * b)
            for t in range(1, self.N):
                eps[:, t] = b * eps[:, t - 1] + scale * eps[:, t]
        noisy = U_bar[None] + sig[None] * eps
        return np.concatenate([U_bar[None].astype(np.float32), noisy], axis=0)

    def update(self, U_samples, rewards):
        """U_samples: (Kv, N, nu); rewards: (Kv,). Returns (N, nu)."""
        costs = -rewards
        beta = costs.min()
        scale = self.temperature
        if self.cost_norm:
            scale *= max(float(costs.std()), 1e-6)
        w = np.exp(-(costs - beta) / scale)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            # All weights underflowed; fall back to argmin-cost picked sample
            j = int(np.argmin(costs))
            return U_samples[j].copy()
        w = w / s
        return (w[:, None, None] * U_samples).sum(axis=0).astype(np.float32)


class MPPIPlanner:
    """Vectorized MPPI planner. The dynamics object is duck-typed: it must
    expose .step_vec(X, U, dt) -> X_next over a leading batch dimension and
    a .hover_control attribute used for cold-start initialization."""

    def __init__(self, mppi, dynamics, obstacle_map,
                 ctrl_lo, ctrl_hi,
                 robot_radius=0.20, safe_margin=0.05,
                 state_lims=None,
                 progress_w=10.0, terminal_w=100.0, near_obs_w=3.0,
                 z_track_w=20.0,
                 v_cap=None, v_cap_w=0.0,
                 smoothing_w=0.0, dt=0.05, mppi_iters=1,
                 consistency_w=0.0, near_obs_soft=False, near_obs_falloff=0.30,
                 revalidate_mean=False, retry_keep_warm=False,
                 path_consistency_w=0.0):
        self.mppi = mppi
        self.dyn = dynamics
        self.occ = obstacle_map
        self.ctrl_lo = np.asarray(ctrl_lo, dtype=np.float32)
        self.ctrl_hi = np.asarray(ctrl_hi, dtype=np.float32)
        self.robot_radius = float(robot_radius)
        self.safe_margin = float(safe_margin)
        # state_lims: list of (lo, hi) per state index, or None to skip
        self.state_lims = state_lims
        self.progress_w = float(progress_w)
        self.terminal_w = float(terminal_w)
        self.near_obs_w = float(near_obs_w)
        self.z_track_w = float(z_track_w)
        self.v_cap = float(v_cap) if v_cap is not None else None
        self.v_cap_w = float(v_cap_w)
        # Hard cap on planar speed ||v_xy|| (None = only the soft v_cap applies).
        # Rollouts exceeding it are rejected in _validate, like the TurtleBot's
        # hard velocity constraint -- so the planned speed never blows past it.
        self.v_hard = None
        self.smoothing_w = float(smoothing_w)
        # Hysteresis toward the previous plan: penalize each rollout's mean
        # squared control deviation from the warm-start nominal. The nominal
        # itself (rollout #0) pays zero, so with two near-tied avoidance
        # homotopies the previously chosen side wins -- no left/right flipping.
        self.consistency_w = float(consistency_w)
        # Smooth EDT-based proximity cost instead of the binary near/far
        # indicator: quadratic ramp from 0 at (infl + falloff) to 1 at infl.
        # Gives a left/right tie-breaking gradient and is robust to 1-cell
        # grid flicker.
        self.near_obs_soft = bool(near_obs_soft)
        self.near_obs_falloff = float(near_obs_falloff)
        # Re-validate the softmax-averaged trajectory: the mean of a bimodal
        # (left+right) rollout population can cut straight through the
        # obstacle. If it collides, fall back to the best valid rollout.
        self.revalidate_mean = bool(revalidate_mean)
        # First recovery retry keeps the warm-start nominal (only widens
        # sigma); cold hover reset is the last resort. Prevents a single
        # zero-valid tick from randomly re-seeding the avoidance side.
        self.retry_keep_warm = bool(retry_keep_warm)
        self.last_mean_fallback = False
        self.n_mean_fallbacks = 0
        # Hysteresis in PATH space: penalize mean squared xy deviation from the
        # previous plan's trajectory. Positions integrate the controls, so the
        # left/right signature is far stronger here than in control space --
        # this is the term that actually pins the avoidance side.
        self.path_consistency_w = float(path_consistency_w)
        self.dt = float(dt)
        self.M = int(mppi_iters)
        self.prev_U = None  # warm-start cache
        self.prev_X = None  # previous optimal trajectory (path hysteresis)
        # Optional geodesic cost-to-go field (cost_to_go.CostToGoField). When
        # set, the progress/terminal goal terms use shortest-path-around-obstacle
        # distance instead of Euclidean, which removes head-on local minima.
        self.cost_to_go = None

    # -------- public API --------------------------------------------------

    def update_obstacle_map(self, obstacle_map):
        self.occ = obstacle_map  # MPPI history (prev_U) is preserved

    def set_cost_to_go(self, field):
        """Install a cost_to_go.CostToGoField (or None to revert to Euclidean)."""
        self.cost_to_go = field

    def reset_warm_start(self):
        self.prev_U = None
        self.prev_X = None

    def plan(self, x0, goal, warm_shift=1):
        """Returns (X_opt, U_opt) of shapes ((N+1, nx), (N, nu)), or
        (None, None) if no valid trajectory was found after a recovery
        retry with widened noise.

        `warm_shift` is the number of plan-steps that have elapsed since the
        last successful plan() call (i.e. round(plan_period / mppi_dt))."""
        x0 = np.asarray(x0, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)

        # Hysteresis anchors from the previous plan (only when actually warm).
        warm = self.prev_U is not None
        X_ref = self._shift_path(warm_shift) if (
            warm and self.path_consistency_w > 0.0) else None
        U_bar = self._warm_start(shift=warm_shift)
        U_ref = U_bar.copy() if (warm and self.consistency_w > 0.0) else None
        sigma_orig = self.mppi.sigma
        # Recovery ladder when no rollout validates:
        #   0: warm nominal, sigma
        #   1: warm nominal, 2x sigma      (retry_keep_warm; else -> cold)
        #   2: cold hover,   2x sigma      (last resort; drops the hysteresis)
        n_retries = 3 if self.retry_keep_warm else 2
        U_valid = R = None
        for retry in range(n_retries):
            ok = True
            for it in range(self.M):
                U = self.mppi.sample(U_bar)
                U = np.clip(U, self.ctrl_lo[None, None, :],
                            self.ctrl_hi[None, None, :])
                X = self._simulate(x0, U)
                valid = self._validate(X)
                if not valid.any():
                    ok = False
                    break
                U_valid = U[valid]
                R = self._reward(X[valid], U_valid, goal,
                                 U_ref=U_ref, X_ref=X_ref)
                U_bar = self.mppi.update(U_valid, R)
                U_bar = np.clip(U_bar, self.ctrl_lo[None, :],
                                self.ctrl_hi[None, :])
            if ok:
                break
            self.mppi.sigma = sigma_orig * 2.0
            if self.retry_keep_warm and retry == 0 and self.prev_U is not None:
                # Keep the warm nominal; only widen the noise.
                U_bar = self._warm_start(shift=warm_shift)
            else:
                # Cold-start nominal (hover); hysteresis no longer applies.
                U_ref = None
                X_ref = None
                U_bar = np.broadcast_to(self.dyn.hover_control,
                                        (self.mppi.N, self.mppi.nu))
                U_bar = np.clip(U_bar.astype(np.float32),
                                self.ctrl_lo[None, :],
                                self.ctrl_hi[None, :]).copy()
        # Restore sigma
        self.mppi.sigma = sigma_orig

        if not ok:
            return None, None

        X_opt = self._simulate(x0, U_bar[None])[0]
        self.last_mean_fallback = False
        if self.revalidate_mean and not self._validate(X_opt[None])[0]:
            # The softmax mean of a bimodal (left+right) population is
            # infeasible. Fall back to the softmax mean of the ELITE set
            # (top rollouts, which cluster in the winning mode) -- much
            # smoother tick-to-tick than a single rollout; if even that
            # collides, use the best single rollout (valid by construction).
            U_bar, X_opt = self._elite_fallback(x0, U_valid, R)
            self.last_mean_fallback = True
            self.n_mean_fallbacks += 1
        self.prev_U = U_bar.copy()
        self.prev_X = X_opt.copy()
        return X_opt, U_bar

    def _shift_path(self, shift):
        """Previous X_opt shifted left by `shift` steps (like _warm_start),
        last node repeated -- the path-hysteresis reference."""
        if self.prev_X is None:
            return None
        shift = max(1, min(int(shift), self.prev_X.shape[0] - 1))
        tail = np.repeat(self.prev_X[-1:], shift, axis=0)
        return np.vstack([self.prev_X[shift:], tail])

    def _elite_fallback(self, x0, U_valid, R):
        """Softmax mean over the top ~10% rollouts; best single as last resort."""
        E = max(4, U_valid.shape[0] // 10)
        top = np.argsort(R)[-E:]
        U_e = self.mppi.update(U_valid[top], R[top])
        U_e = np.clip(U_e, self.ctrl_lo[None, :], self.ctrl_hi[None, :])
        X_e = self._simulate(x0, U_e[None])[0]
        if self._validate(X_e[None])[0]:
            return U_e, X_e
        j = int(np.argmax(R))
        U_b = U_valid[j].copy()
        return U_b, self._simulate(x0, U_b[None])[0]

    def plan_debug(self, x0, goal, warm_shift=1):
        """Like plan(), but also returns the full sampled batch + valid mask
        + rewards for visualization."""
        x0 = np.asarray(x0, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)

        warm = self.prev_U is not None
        X_ref = self._shift_path(warm_shift) if (
            warm and self.path_consistency_w > 0.0) else None
        U_bar = self._warm_start(shift=warm_shift)
        U_ref = U_bar.copy() if (warm and self.consistency_w > 0.0) else None
        # We only support M=1 in debug for simplicity
        U = self.mppi.sample(U_bar)
        U = np.clip(U, self.ctrl_lo[None, None, :], self.ctrl_hi[None, None, :])
        X = self._simulate(x0, U)
        valid = self._validate(X)
        info = dict(X=X, U=U, valid=valid)
        if not valid.any():
            return None, None, info
        R = self._reward(X[valid], U[valid], goal, U_ref=U_ref, X_ref=X_ref)
        U_bar = self.mppi.update(U[valid], R)
        U_bar = np.clip(U_bar, self.ctrl_lo[None, :], self.ctrl_hi[None, :])
        X_opt = self._simulate(x0, U_bar[None])[0]
        self.last_mean_fallback = False
        if self.revalidate_mean and not self._validate(X_opt[None])[0]:
            U_bar, X_opt = self._elite_fallback(x0, U[valid], R)
            self.last_mean_fallback = True
            self.n_mean_fallbacks += 1
        self.prev_U = U_bar.copy()
        self.prev_X = X_opt.copy()
        info['rewards'] = R
        info['X_opt'] = X_opt
        info['mean_fallback'] = self.last_mean_fallback
        return X_opt, U_bar, info

    # -------- internals ---------------------------------------------------

    def _warm_start(self, shift=1):
        """Shift the previous nominal sequence left by `shift` plan-steps and
        pad on the right with hover. `shift` should equal the number of
        plan-steps that elapsed since the last successful plan() call —
        with plan_period == mppi_dt that's 1; with a 10/50 Hz split where
        plan_period == 2*mppi_dt that's 2.

        Pad-with-hover (rather than pad-with-last) keeps the right-end of the
        nominal sequence stable: duplicating an extremal U_opt[N-1] would
        bias the next plan toward whatever rare action ended the horizon."""
        N, nu = self.mppi.N, self.mppi.nu
        hover = self.dyn.hover_control.astype(np.float32)
        shift = max(1, int(shift))
        if self.prev_U is not None and self.prev_U.shape == (N, nu):
            shift = min(shift, N - 1)
            tail = np.broadcast_to(hover, (shift, nu)).astype(np.float32)
            U_bar = np.vstack([self.prev_U[shift:], tail])
            return np.clip(U_bar, self.ctrl_lo[None, :], self.ctrl_hi[None, :])
        # Cold start: hover
        U_bar = np.broadcast_to(hover, (N, nu)).astype(np.float32).copy()
        return np.clip(U_bar, self.ctrl_lo[None, :], self.ctrl_hi[None, :])

    def _simulate(self, x0, U_batch):
        """U_batch: (K, N, nu); x0: (nx,) (broadcast across K). Returns (K, N+1, nx)."""
        K, N, _ = U_batch.shape
        nx = x0.shape[-1]
        X = np.empty((K, N + 1, nx), dtype=np.float32)
        X[:, 0] = np.broadcast_to(x0, (K, nx)).astype(np.float32)
        for t in range(N):
            X[:, t + 1] = self.dyn.step_vec(X[:, t], U_batch[:, t], self.dt)
        return X

    def _validate(self, X):
        """X: (K, N+1, nx) -> (K,) bool. Hard constraints: state limits +
        collision. State-limit checks skip the initial step (t=0) because
        the planner is called from the *real* env state, which may briefly
        violate the planner's nominal limits due to tracker overshoot or
        unmodeled disturbance — we want the rollout to be allowed to relax
        back inside the limits, not flagged as invalid for inheriting an
        out-of-band starting condition."""
        K = X.shape[0]
        valid = np.ones(K, dtype=bool)
        if self.state_lims is not None:
            X_future = X[:, 1:, :]   # exclude t=0
            for i, lim in enumerate(self.state_lims):
                if lim is None:
                    continue
                lo, hi = lim
                if lo is not None:
                    valid &= ~(X_future[..., i] < lo).any(axis=1)
                if hi is not None:
                    valid &= ~(X_future[..., i] > hi).any(axis=1)
        if self.v_hard is not None and X.shape[-1] >= 5:
            # Hard planar-speed cap (skip t=0: the real start state may briefly
            # overshoot; let the rollout relax back inside).
            speed = np.linalg.norm(X[:, 1:, 3:5], axis=-1)   # (K, N)
            valid &= ~(speed > self.v_hard).any(axis=1)
        # Skip t=0 (like the state-limit / speed checks above): the planner is
        # called from the real state, whose current cell may briefly read as
        # in-collision (perception noise, or the robot's own footprint sitting
        # <robot_radius from a residual obstacle cell). Rejecting t=0 would kill
        # *every* rollout for that inherited start condition; instead we let
        # rollouts that escape stay valid — any that remain in / move into the
        # obstacle still collide at t>=1 and are rejected.
        coll = self.occ.collides(X[:, 1:], self.robot_radius)
        valid &= ~coll.any(axis=1)
        return valid

    def _reward(self, X, U, goal, U_ref=None, X_ref=None):
        """X: (Kv, N+1, nx), U: (Kv, N, nu), goal: (>=2,) -> (Kv,).
        U_ref: (N, nu) warm-start nominal for the control-consistency term.
        X_ref: (N+1, nx) shifted previous trajectory for the path-consistency
        (side hysteresis) term. Either may be None to disable.

        All position-related terms use xy only (the planning task is planar;
        altitude, if present, is held by the tracker, not the planner). For
        the full-quadrotor planner, set z_track_w > 0 to also penalize
        |z_t - goal[2]|; for the planar planner leave it at 0 (state[..., 2]
        is yaw, not z, so the term is meaningless)."""
        if self.cost_to_go is not None:
            # Geodesic cost-to-go (around obstacles) instead of Euclidean.
            g = self.cost_to_go.query(X[..., :2])          # (Kv, N+1)
            R = -self.progress_w * g.sum(axis=1)
            R -= self.terminal_w * g[:, -1]
        else:
            d_xy = np.linalg.norm(X[..., :2] - goal[None, None, :2], axis=-1)
            R = -self.progress_w * d_xy.sum(axis=1)
            d_terminal_xy = np.linalg.norm(X[:, -1, :2] - goal[None, :2], axis=-1)
            R -= self.terminal_w * d_terminal_xy

        if self.z_track_w > 0.0 and goal.shape[0] >= 3 and X.shape[-1] >= 12:
            # Only meaningful when the state's index-2 column really is z.
            dz = X[..., 2] - goal[None, None, 2]
            R -= self.z_track_w * np.abs(dz).sum(axis=1)

        # Soft speed cap. State-limit validation handles only the catastrophic
        # outer bound; the actual desired max speed is enforced here so the
        # planner has gradient pressure away from over-speed without rejecting
        # rollouts that briefly overshoot (e.g. due to tracker dynamics).
        if self.v_cap is not None and self.v_cap_w > 0.0 and X.shape[-1] >= 5:
            # Planar state has v at indices 3,4 (vx, vy); full quad has at 3,4,5.
            v_xy = X[..., 3:5]
            speed = np.linalg.norm(v_xy, axis=-1)
            excess = np.maximum(speed - self.v_cap, 0.0)
            R -= self.v_cap_w * (excess ** 2).sum(axis=1)

        infl = self.robot_radius + self.safe_margin
        if self.near_obs_soft:
            # Quadratic ramp: 1 at the inflation radius, 0 at infl + falloff.
            # Same scale as the binary indicator inside infl, but with a
            # gradient that breaks left/right ties and tolerates 1-cell
            # grid flicker.
            d = self.occ.clearance(X)
            falloff = max(self.near_obs_falloff, 1e-6)
            prox = np.clip((infl + falloff - d) / falloff, 0.0, 1.0)
            R -= self.near_obs_w * (prox * prox).sum(axis=1)
        else:
            near = self.occ.near_obstacle(X, infl)
            R -= self.near_obs_w * near.sum(axis=1)

        if self.smoothing_w > 0.0:
            du = np.diff(U, axis=1)
            R -= self.smoothing_w * (du * du).sum(axis=(1, 2))

        if self.consistency_w > 0.0 and U_ref is not None:
            dU = U - U_ref[None]
            R -= self.consistency_w * (dU * dU).mean(axis=(1, 2))

        if self.path_consistency_w > 0.0 and X_ref is not None:
            dP = X[..., :2] - X_ref[None, :, :2]
            R -= self.path_consistency_w * (dP * dP).sum(axis=-1).mean(axis=1)

        return R.astype(np.float32)
