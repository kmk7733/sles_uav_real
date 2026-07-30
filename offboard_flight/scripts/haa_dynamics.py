#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Full 12-state quadrotor dynamics for HAA (MPPI) rollouts.

DeSimplex paper eq (64)-(67):

    p_dot     = v
    m v_dot   = -m g e3 + f R e3
    R_dot     = R Omega_hat
    J Om_dot  = M - Omega x J Omega

State  x = [px py pz  vx vy vz  roll pitch yaw  wx wy wz]   (nx = 12)
Input  u = [f Mx My Mz]                                      (nu =  4)

Frames: position/velocity in world ENU. Attitude as ZYX (yaw-pitch-roll)
Euler angles, matching how attitude_library.quatToEuler reports them, so a
mavros pose can be turned into this state without a convention change.
Body rates (wx wy wz) are in the body frame.

Everything is vectorised over the leading axis so MPPI can roll out K
samples at once: X is (K, 12) and U is (K, 4).

NOTE ON THE MODEL CHOICE
The paper's HAA planner uses the reduced planar surrogate (eq 71-73), which
has no mass and no inertia; m and J appear only in the full dynamics used by
the low-level controller. This module implements the full dynamics instead
(an explicit project decision), so the planner must itself discover hover
thrust. Callers should therefore initialise the nominal control to hover
(see hover_control) rather than to zero, or the first rollouts all fall.
"""

import numpy as np

G = 9.80665

NX = 12
NU = 4

# Measured airframe mass for this vehicle [kg]. rogx.sdf sums to 3.1928 kg
# across its 10 links; 3.245 is the measured value and takes precedence.
ROGX_MASS = 3.245

# Inertia of rogx.sdf's base_link [kg m^2], off-diagonal terms included.
#   <ixx>0.0218</ixx> <ixy>0.0000</ixy> <ixz>0.0022</ixz>
#   <iyy>0.0242</iyy> <iyz>0.0002</iyz> <izz>0.0349</izz>
# Caveat: this is base_link only. The 8 rotor/prop links (0.00548 kg each, at
# roughly 0.2 m) add ~1.8e-3 to Jzz by the parallel-axis theorem, i.e. about
# 5%. Small relative to the model error we already accept, but it means the
# true Jzz is slightly larger than the value below.
ROGX_INERTIA = np.array([[0.0218, 0.0000, 0.0022],
                         [0.0000, 0.0242, 0.0002],
                         [0.0022, 0.0002, 0.0349]])

# state slices
P = slice(0, 3)
V = slice(3, 6)
RPY = slice(6, 9)
OM = slice(9, 12)


class QuadrotorDynamics(object):

    def __init__(self, mass=ROGX_MASS, inertia=ROGX_INERTIA,
                 tilt_max=0.5236, omega_max=3.0, vel_max=1.5,
                 thrust_min=0.0, thrust_max=None):
        """
        mass        vehicle mass [kg]. Defaults to the measured ROGX_MASS.
        inertia     inertia [kg m^2], either (3,) diagonal or a full (3,3)
                    matrix. Defaults to ROGX_INERTIA, taken from rogx.sdf's
                    base_link (see the note there on rotor contributions).
        tilt_max    max |roll|,|pitch| treated as feasible [rad] (30 deg)
        omega_max   max |body rate| treated as feasible [rad/s]
        vel_max     max |velocity| treated as feasible [m/s]
        thrust_max  max collective thrust [N]; defaults to 2x hover
        """
        self.m = float(mass)
        J = np.asarray(inertia, dtype=np.float64)
        if J.ndim == 1:
            J = np.diag(J)
        if J.shape != (3, 3):
            raise ValueError("inertia must be (3,) or (3,3), got %r" % (J.shape,))
        if not np.allclose(J, J.T, atol=1e-12):
            raise ValueError("inertia matrix must be symmetric")
        self.J = J
        self.Jinv = np.linalg.inv(self.J)

        self.tilt_max = float(tilt_max)
        self.omega_max = float(omega_max)
        self.vel_max = float(vel_max)

        self.hover_thrust = self.m * G
        self.thrust_min = float(thrust_min)
        self.thrust_max = (2.0 * self.hover_thrust if thrust_max is None
                           else float(thrust_max))

    # ------------------------------------------------------------------ util

    def hover_control(self):
        """Control that holds altitude with level attitude: [m g, 0, 0, 0]."""
        u = np.zeros(NU)
        u[0] = self.hover_thrust
        return u

    def clip_control(self, U):
        """Clamp thrust to the actuator range. Moments are left to the caller's
        sampling covariance -- they are bounded implicitly by omega_max via
        feasibility rejection."""
        U = np.array(U, dtype=np.float64, copy=True)
        U[..., 0] = np.clip(U[..., 0], self.thrust_min, self.thrust_max)
        return U

    @staticmethod
    def thrust_dir(rpy):
        """Body z axis expressed in world = R e3, for ZYX Euler angles.

        rpy is (..., 3). Returns (..., 3).
        """
        r = rpy[..., 0]
        p = rpy[..., 1]
        y = rpy[..., 2]
        sr, cr = np.sin(r), np.cos(r)
        sp, cp = np.sin(p), np.cos(p)
        sy, cy = np.sin(y), np.cos(y)
        return np.stack([cr * sp * cy + sr * sy,
                         cr * sp * sy - sr * cy,
                         cr * cp], axis=-1)

    @staticmethod
    def rpy_dot(rpy, omega):
        """Euler-angle rates from body rates, ZYX convention.

        Near pitch = +-90 deg this mapping is singular; pitch is clamped in
        step_vec well before that, and rollouts that get there are rejected as
        infeasible anyway, so no special handling is needed here beyond a
        guard on cos(pitch).
        """
        r = rpy[..., 0]
        p = rpy[..., 1]
        wx = omega[..., 0]
        wy = omega[..., 1]
        wz = omega[..., 2]

        sr, cr = np.sin(r), np.cos(r)
        cp = np.cos(p)
        tp = np.tan(p)

        # keep the singular term finite; feasibility rejection discards these
        cp = np.where(np.abs(cp) < 1e-4, np.sign(cp) * 1e-4 + 1e-12, cp)

        return np.stack([wx + sr * tp * wy + cr * tp * wz,
                         cr * wy - sr * wz,
                         (sr / cp) * wy + (cr / cp) * wz], axis=-1)

    # ------------------------------------------------------------- dynamics

    def step_vec(self, X, U, dt):
        """One explicit-Euler step. X (K,12), U (K,4) -> (K,12)."""
        X = np.atleast_2d(X)
        U = np.atleast_2d(U)

        p = X[:, P]
        v = X[:, V]
        rpy = X[:, RPY]
        om = X[:, OM]

        f = U[:, 0:1]
        Mom = U[:, 1:4]

        # p_dot = v
        p_new = p + dt * v

        # m v_dot = -m g e3 + f R e3
        acc = (f / self.m) * self.thrust_dir(rpy)
        acc[:, 2] -= G
        v_new = v + dt * acc

        # attitude kinematics
        rpy_new = rpy + dt * self.rpy_dot(rpy, om)

        # J om_dot = M - om x J om
        Jom = om.dot(self.J.T)
        om_new = om + dt * (Mom - np.cross(om, Jom)).dot(self.Jinv.T)

        # keep pitch away from the gimbal-lock singularity, wrap yaw
        rpy_new[:, 1] = np.clip(rpy_new[:, 1], -1.5, 1.5)
        rpy_new[:, 2] = np.arctan2(np.sin(rpy_new[:, 2]), np.cos(rpy_new[:, 2]))

        return np.concatenate([p_new, v_new, rpy_new, om_new], axis=1)

    def rollout(self, x0, U_seq, dt):
        """Roll K control sequences forward.

        x0     (12,)              initial state, shared by every sample
        U_seq  (K, N, 4)          control sequence per sample
        returns X (K, N+1, 12)
        """
        U_seq = np.atleast_3d(U_seq)
        K, N = U_seq.shape[0], U_seq.shape[1]
        X = np.empty((K, N + 1, NX))
        X[:, 0, :] = x0
        for i in range(N):
            X[:, i + 1, :] = self.step_vec(X[:, i, :], U_seq[:, i, :], dt)
        return X

    # ---------------------------------------------------------- feasibility

    def feasible(self, X):
        """Per-sample feasibility over a whole rollout. X (K,N+1,12) -> (K,).

        A sample is infeasible if it ever exceeds the tilt, body-rate or
        velocity limits, or goes non-finite. This is the eq (2) hard state
        constraint; the paper discards such samples before weighting.
        """
        rpy = X[..., RPY]
        om = X[..., OM]
        v = X[..., V]

        ok = np.isfinite(X).all(axis=(1, 2))
        ok &= (np.abs(rpy[..., 0]) <= self.tilt_max).all(axis=1)
        ok &= (np.abs(rpy[..., 1]) <= self.tilt_max).all(axis=1)
        ok &= (np.linalg.norm(om, axis=-1) <= self.omega_max).all(axis=1)
        ok &= (np.linalg.norm(v, axis=-1) <= self.vel_max).all(axis=1)
        return ok


class ThrustRateQuadrotor(object):
    """CTBR model: collective thrust + body-rate commands, as in PA-MPPI.

        u = [c, wx_cmd, wy_cmd, wz_cmd]

    PA-MPPI (Zhai et al., RA-L 2026) uses exactly this input for quadrotor
    navigation on an occupancy grid, with state (p, q, v, omega_B). We keep the
    same 12-vector layout as the rest of this module (Euler angles instead of a
    quaternion) so the state indices do not change:

        p'    = p + dt v
        v'    = v + dt (c/m R(rpy) e3 - g e3)
        rpy'  = rpy + dt rpy_dot(rpy, omega)
        omega'= omega_cmd + (omega - omega_cmd) exp(-dt / tau)

    The body-rate command is tracked with a first-order lag of time constant
    tau, standing in for PX4's rate loop. The update is the exact solution of
    that linear system rather than an Euler step, so it is stable for any dt --
    which matters because a realistic tau (~0.03 s) is far shorter than the
    0.1 s planner step, and explicit Euler would blow up there.

    WHERE J WENT
    Nothing here uses the inertia matrix. Commanding body rates means never
    forming a moment, and if you do write the rate loop out as
    M = J k (omega_cmd - omega) + omega x J omega and push it through
    J omega_dot = M - omega x J omega, the J cancels exactly. So under CTBR the
    inertia has no effect on the predicted trajectory. Mass still matters, via
    thrust-to-acceleration. If you want J to influence the plan you need moments
    as the input, i.e. QuadrotorDynamics driven directly -- but see
    AttitudeStabilizedQuadrotor for why raw moment sampling does not work.

    This model needs no inner attitude loop: attitude is driven by a bounded
    commanded rate rather than integrated twice from a moment, so it does not
    random-walk out of the tilt limit the way raw [f, M] sampling does.
    """

    nu = 4

    def __init__(self, base, tau_rate=0.03):
        self.base = base
        self.tau_rate = float(tau_rate)

    @property
    def m(self):
        return self.base.m

    @property
    def J(self):
        return self.base.J

    @property
    def hover_thrust(self):
        return self.base.hover_thrust

    @property
    def thrust_max(self):
        return self.base.thrust_max

    def hover_control(self):
        """Hover: thrust = m g, zero body rates."""
        u = np.zeros(self.nu)
        u[0] = self.base.hover_thrust
        return u

    def clip_control(self, U):
        U = np.array(U, dtype=np.float64, copy=True)
        U[..., 0] = np.clip(U[..., 0], self.base.thrust_min, self.base.thrust_max)
        U[..., 1:4] = np.clip(U[..., 1:4],
                              -self.base.omega_max, self.base.omega_max)
        return U

    def step_vec(self, X, U, dt):
        X = np.atleast_2d(X)
        U = np.atleast_2d(U)

        p = X[:, P]
        v = X[:, V]
        rpy = X[:, RPY]
        om = X[:, OM]

        c = U[:, 0:1]
        om_cmd = U[:, 1:4]

        p_new = p + dt * v

        acc = (c / self.base.m) * self.base.thrust_dir(rpy)
        acc[:, 2] -= G
        v_new = v + dt * acc

        rpy_new = rpy + dt * self.base.rpy_dot(rpy, om)

        # exact first-order lag toward the commanded rate
        a = np.exp(-dt / self.tau_rate)
        om_new = om_cmd + (om - om_cmd) * a

        rpy_new[:, 1] = np.clip(rpy_new[:, 1], -1.5, 1.5)
        rpy_new[:, 2] = np.arctan2(np.sin(rpy_new[:, 2]), np.cos(rpy_new[:, 2]))

        return np.concatenate([p_new, v_new, rpy_new, om_new], axis=1)

    def rollout(self, x0, U_seq, dt):
        U_seq = np.atleast_3d(U_seq)
        K, N = U_seq.shape[0], U_seq.shape[1]
        X = np.empty((K, N + 1, NX))
        X[:, 0, :] = x0
        for i in range(N):
            X[:, i + 1, :] = self.step_vec(X[:, i, :], U_seq[:, i, :], dt)
        return X

    def feasible(self, X):
        return self.base.feasible(X)


class AttitudeStabilizedQuadrotor(object):
    """The same 12-state physics, but driven through an inner attitude loop.

    WHY THIS EXISTS
    Sampling raw [f, Mx, My, Mz] over a multi-second horizon does not work for
    planning: attitude is integrated open-loop, so random moments make it
    random-walk, tilt grows without bound, and the horizontal acceleration
    g*tan(tilt) makes velocity diverge. Measured on this airframe with
    sigma_M = 0.05 N m over a 4 s horizon: 0.9% of samples stay inside the tilt
    limit, 0.0% inside the velocity limit, peak roll 22 rad, peak speed 53 m/s.
    Shrinking sigma to 0.001 keeps 72% feasible but then the vehicle barely
    tilts, so it cannot translate and MPPI has nothing to search over.

    The real vehicle is not open-loop: PX4 stabilises attitude. This wrapper
    models that. The sampled input becomes a desired world-frame acceleration
    plus a yaw rate, and the moments are produced by a P attitude loop and a P
    rate loop, exactly the cascade PX4 runs.

    m and J are still used, and the state is still the full 12-vector -- only
    the input parameterisation changes:

        v = [ax_des, ay_des, az_des, yaw_rate_des]

    The tilt limit turns into an acceleration limit, which is the paper's
    eq (78):  a_max <= g tan(theta_max).
    """

    nu = 4

    def __init__(self, base, k_att=5.0, k_rate=10.0, substeps=1):
        """
        base      QuadrotorDynamics providing the physics and the limits
        k_att     attitude P gain [1/s]
        k_rate    body-rate P gain [1/s]
        substeps  inner integration substeps per planner step. Explicit Euler
                  needs dt_sub * k_rate < 2 to be stable; at dt = 0.1 and
                  k_rate = 10 that is 1.0, so substeps=1 is stable and is the
                  default. Raising substeps multiplies solve time almost
                  linearly (measured on this Jetson at K=1024, N=20:
                  173 / 261 / 458 ms for 1 / 2 / 3), which is why it is 1.
        """
        self.base = base
        self.k_att = float(k_att)
        self.k_rate = float(k_rate)
        self.substeps = max(1, int(substeps))

        # eq (78): the tilt limit expressed as a horizontal acceleration bound
        self.a_max_xy = G * np.tan(base.tilt_max)

    # convenience passthroughs so callers can treat this like the base model
    @property
    def m(self):
        return self.base.m

    @property
    def J(self):
        return self.base.J

    @property
    def hover_thrust(self):
        return self.base.hover_thrust

    @property
    def thrust_max(self):
        return self.base.thrust_max

    def hover_control(self):
        """Hover is the zero of this input space -- no acceleration, no yaw rate."""
        return np.zeros(self.nu)

    def clip_control(self, V):
        """Bound horizontal acceleration by eq (78) and vertical by thrust."""
        V = np.array(V, dtype=np.float64, copy=True)
        axy = V[..., 0:2]
        n = np.linalg.norm(axy, axis=-1, keepdims=True)
        scale = np.minimum(1.0, self.a_max_xy / np.maximum(n, 1e-9))
        V[..., 0:2] = axy * scale
        az_max = self.base.thrust_max / self.base.m - G
        V[..., 2] = np.clip(V[..., 2], -G * 0.5, az_max)
        V[..., 3] = np.clip(V[..., 3], -self.base.omega_max, self.base.omega_max)
        return V

    def _moments(self, X, V):
        """Inner cascade: desired accel -> (thrust, moments). X (K,12), V (K,4)."""
        rpy = X[:, RPY]
        om = X[:, OM]
        yaw = rpy[:, 2]

        # required thrust vector in world: f_vec = m (a_des + g e3)
        a_des = V[:, 0:3].copy()
        f_vec = self.base.m * (a_des + np.array([0.0, 0.0, G]))
        f = np.linalg.norm(f_vec, axis=1)
        f = np.clip(f, 1e-6, self.base.thrust_max)

        b3 = f_vec / np.maximum(np.linalg.norm(f_vec, axis=1, keepdims=True), 1e-9)
        sy, cy = np.sin(yaw), np.cos(yaw)

        # invert R e3 = b3 for the ZYX Euler angles at the current yaw
        roll_des = np.arcsin(np.clip(b3[:, 0] * sy - b3[:, 1] * cy, -1.0, 1.0))
        pitch_des = np.arctan2(b3[:, 0] * cy + b3[:, 1] * sy, b3[:, 2])

        roll_des = np.clip(roll_des, -self.base.tilt_max, self.base.tilt_max)
        pitch_des = np.clip(pitch_des, -self.base.tilt_max, self.base.tilt_max)

        # attitude P -> desired body rates (yaw rate is commanded directly)
        om_des = np.stack([self.k_att * (roll_des - rpy[:, 0]),
                           self.k_att * (pitch_des - rpy[:, 1]),
                           V[:, 3]], axis=1)
        om_des = np.clip(om_des, -self.base.omega_max, self.base.omega_max)

        # rate P, with the gyroscopic term fed forward
        Jom = om.dot(self.base.J.T)
        M = (self.k_rate * (om_des - om)).dot(self.base.J.T) + np.cross(om, Jom)

        return np.concatenate([f[:, None], M], axis=1)

    def step_vec(self, X, V, dt):
        X = np.atleast_2d(X)
        V = np.atleast_2d(V)
        h = dt / self.substeps
        for _ in range(self.substeps):
            U = self._moments(X, V)
            X = self.base.step_vec(X, U, h)
        return X

    def rollout(self, x0, V_seq, dt):
        V_seq = np.atleast_3d(V_seq)
        K, N = V_seq.shape[0], V_seq.shape[1]
        X = np.empty((K, N + 1, NX))
        X[:, 0, :] = x0
        for i in range(N):
            X[:, i + 1, :] = self.step_vec(X[:, i, :], V_seq[:, i, :], dt)
        return X

    def feasible(self, X):
        return self.base.feasible(X)


def state_from_pose(x, y, z, vx, vy, vz, roll, pitch, yaw,
                    wx=0.0, wy=0.0, wz=0.0):
    """Assemble a 12-vector in the layout this module expects."""
    return np.array([x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz],
                    dtype=np.float64)


def hover_state(x, y, z, yaw=0.0):
    """Level, motionless state at a given position and yaw."""
    return state_from_pose(x, y, z, 0, 0, 0, 0, 0, yaw)
