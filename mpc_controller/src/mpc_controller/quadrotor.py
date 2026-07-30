"""Quadrotor dynamics for MPPI rollouts.

State (nx=12): [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
Control (nu=4): [T, wx_des, wy_des, wz_des]

z is up. Body-rate-controlled (cascaded) — the inner attitude loop is
modelled as a first-order lag w -> w_des with time constant tau_rate;
collective thrust is commanded directly. RPY uses ZYX (yaw-pitch-roll)
intrinsic Euler angles; we never approach pitch ±pi/2 in this regime.
"""
import numpy as np

GRAVITY = 9.81


class Quadrotor:
    nx = 12
    nu = 4

    def __init__(self, mass=1.0, tau_rate=0.04, drag=0.1):
        self.m = float(mass)
        self.tau = float(tau_rate)
        self.drag = float(drag)
        self.g = np.array([0.0, 0.0, -GRAVITY], dtype=np.float32)
        self.T_hover = self.m * GRAVITY

    @staticmethod
    def thrust_dir(rpy):
        """Body-z axis expressed in world frame; rpy: (..., 3) -> (..., 3)."""
        r = rpy[..., 0]; p = rpy[..., 1]; y = rpy[..., 2]
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        bx = cy * sp * cr + sy * sr
        by = sy * sp * cr - cy * sr
        bz = cp * cr
        return np.stack([bx, by, bz], axis=-1)

    @staticmethod
    def rpy_dot(rpy, omega):
        """ZYX-intrinsic body-rate -> RPY-rate map; (..., 3), (..., 3) -> (..., 3)."""
        r, p = rpy[..., 0], rpy[..., 1]
        wx, wy, wz = omega[..., 0], omega[..., 1], omega[..., 2]
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cp_safe = np.where(np.abs(cp) < 1e-3, np.sign(cp + 1e-12) * 1e-3, cp)
        tp = sp / cp_safe
        rdot = wx + sr * tp * wy + cr * tp * wz
        pdot = cr * wy - sr * wz
        ydot = (sr * wy + cr * wz) / cp_safe
        return np.stack([rdot, pdot, ydot], axis=-1)

    def step_vec(self, X, U, dt):
        """X: (..., 12), U: (..., 4) -> (..., 12). Semi-implicit Euler."""
        p = X[..., 0:3]
        v = X[..., 3:6]
        rpy = X[..., 6:9]
        w = X[..., 9:12]

        T = U[..., 0:1]
        w_des = U[..., 1:4]

        e_b = self.thrust_dir(rpy)
        a = self.g + (T / self.m) * e_b - self.drag * v

        # Body-rate first-order lag toward w_des
        alpha = dt / max(self.tau, 1e-6)
        alpha = min(alpha, 1.0)
        w_new = w + (w_des - w) * alpha

        rpy_new = rpy + self.rpy_dot(rpy, w) * dt
        v_new = v + a * dt
        p_new = p + v_new * dt

        return np.concatenate([p_new, v_new, rpy_new, w_new], axis=-1)

    def step(self, x, u, dt):
        return self.step_vec(np.atleast_2d(x), np.atleast_2d(u), dt)[0]

    @property
    def hover_control(self):
        return np.array([self.T_hover, 0.0, 0.0, 0.0], dtype=np.float32)


def hover_state(p, yaw=0.0):
    """Build a 12-vec at given position with zero velocity, level attitude, yaw."""
    x = np.zeros(12, dtype=np.float32)
    x[0:3] = p
    x[8] = yaw
    return x


class PlanarHolonomic:
    """Planar (altitude-hold) double-integrator surrogate used by MPPI.

    State (nx=6):  [x, y, yaw, vx, vy, w_yaw]
    Control (nu=3): [ax, ay, alpha_yaw]    (accelerations / angular accel)

    The plant is the full Quadrotor; this surrogate is what MPPI rolls out
    in plan-space. Altitude is decoupled (held by the tracker / outer loop).
    Yaw is in the state because the user wants it planned, but no explicit
    yaw cost is added in the reward; MPPI samples ω_yaw noise around 0 and
    yaw stays near its initial value unless something pushes it.
    """
    nx = 6
    nu = 3

    def __init__(self):
        # Hover == "no acceleration command"
        self.hover_control = np.zeros(3, dtype=np.float32)

    def step_vec(self, X, U, dt):
        """X: (..., 6), U: (..., 3) -> (..., 6). Semi-implicit Euler."""
        p   = X[..., 0:2]
        yaw = X[..., 2:3]
        v   = X[..., 3:5]
        w   = X[..., 5:6]
        a     = U[..., 0:2]
        alpha = U[..., 2:3]
        v_new = v + a * dt
        w_new = w + alpha * dt
        p_new = p + v_new * dt
        yaw_new = yaw + w_new * dt
        return np.concatenate([p_new, yaw_new, v_new, w_new], axis=-1)

    def step(self, x, u, dt):
        return self.step_vec(np.atleast_2d(x), np.atleast_2d(u), dt)[0]


def planar_state(p_xy, yaw=0.0):
    """Build a 6-vec planar state at given (x, y) with zero velocity and yaw."""
    x = np.zeros(6, dtype=np.float32)
    x[0:2] = p_xy[:2]
    x[2] = yaw
    return x


def quad_to_planar(x_quad):
    """Project a 12-D quadrotor state down to a 6-D planar state."""
    out = np.zeros(6, dtype=np.float32)
    out[0] = x_quad[0]      # x
    out[1] = x_quad[1]      # y
    out[2] = x_quad[8]      # yaw
    out[3] = x_quad[3]      # vx
    out[4] = x_quad[4]      # vy
    out[5] = x_quad[11]     # w_yaw
    return out


def planar_to_quad_setpoint(x_planar, z_hold):
    """Inflate a planar planner waypoint into a 12-D setpoint that the
    GeometricTracker can consume. Sets z = z_hold, vz = 0, level attitude."""
    out = np.zeros(12, dtype=np.float32)
    out[0] = x_planar[0]    # x
    out[1] = x_planar[1]    # y
    out[2] = z_hold         # z (held)
    out[3] = x_planar[3]    # vx
    out[4] = x_planar[4]    # vy
    out[5] = 0.0            # vz
    out[8] = x_planar[2]    # yaw
    return out
