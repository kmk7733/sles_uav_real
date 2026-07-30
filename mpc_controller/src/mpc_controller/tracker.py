"""Geometric (Lee, Leok, McClamroch 2010-style) trajectory tracker that
emits the same low-level (T, omega_des) commands the env consumes.

Inputs to __call__:
  x      : current state (12,) [p, v, rpy, w]
  x_des  : desired state (12,) — only [p_des(3), v_des(3), yaw_des] are used
  a_ff   : optional feedforward acceleration in world frame (3,)

Outputs:
  u : (T, wx_des, wy_des, wz_des)

Design choices:
  - Specific-force computation gives both the desired thrust magnitude T
    AND the desired body-z direction b3_des.
  - Desired yaw is provided; b1_des is built from a yaw-aligned heading
    vector projected into the plane perpendicular to b3_des. If yaw is
    underdetermined (b3_des nearly aligned with the heading), the controller
    falls back to keeping the current body-x.
  - Attitude error follows the Lee et al. SO(3) error e_R = vee(0.5*(R_des^T
    R - R^T R_des)); the rate command is omega_des = -k_R e_R.
"""
import numpy as np


def rpy_to_rotmat(rpy):
    """ZYX intrinsic Euler -> 3x3 rotation matrix."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [   -sp,                  cp * sr,                 cp * cr],
    ], dtype=np.float32)


GRAVITY = 9.81


class GeometricTracker:
    def __init__(self, mass=1.0, kp=6.0, kv=4.0, k_att=12.0,
                 omega_max=3.0, T_min_frac=0.2, T_max_frac=2.0,
                 yaw_rate_max=None):
        self.m = float(mass)
        self.kp = float(kp)
        self.kv = float(kv)
        self.k_att = float(k_att)
        self.omega_max = float(omega_max)
        # Separate hard cap on the yaw (body-z) rate, e.g. to match a hardware
        # heading-rate limit, WITHOUT throttling roll/pitch (which drive
        # translation). None -> only the ||omega|| cap applies.
        self.yaw_rate_max = None if yaw_rate_max is None else float(yaw_rate_max)
        self.T_min = T_min_frac * mass * GRAVITY
        self.T_max = T_max_frac * mass * GRAVITY

    def __call__(self, x, x_des, a_ff=None):
        p = x[0:3]
        v = x[3:6]
        rpy = x[6:9]
        p_d = x_des[0:3]
        v_d = x_des[3:6]
        yaw_d = float(x_des[8])
        if a_ff is None:
            a_ff = np.zeros(3, dtype=np.float32)

        # Desired specific force in world frame:
        # m * (kp e_p + kv e_v + a_ff + g e_z) where g points up here because
        # specific force = a + g e_z (must overcome gravity).
        e_p = p_d - p
        e_v = v_d - v
        f_des = self.m * (self.kp * e_p + self.kv * e_v + a_ff
                          + np.array([0.0, 0.0, GRAVITY], dtype=np.float32))
        nf = max(np.linalg.norm(f_des), 1e-6)
        b3_des = f_des / nf

        # Build R_des from (b3_des, yaw_des). c_des is the desired heading
        # (b1) projected away from b3_des to enforce orthogonality.
        c_des = np.array([np.cos(yaw_d), np.sin(yaw_d), 0.0], dtype=np.float32)
        b2_des = np.cross(b3_des, c_des)
        nb2 = np.linalg.norm(b2_des)
        R = rpy_to_rotmat(rpy)
        if nb2 < 1e-4:
            # Singular: use current body-x to break the tie.
            c_des = R[:, 0]
            b2_des = np.cross(b3_des, c_des)
            nb2 = max(np.linalg.norm(b2_des), 1e-6)
        b2_des = b2_des / nb2
        b1_des = np.cross(b2_des, b3_des)
        R_des = np.column_stack([b1_des, b2_des, b3_des]).astype(np.float32)

        # Lee 2010 thrust formula: project f_des onto current body-z so that
        # only the gravity-compensation + altitude-error portion is applied
        # vertically; the lateral component is realized by attitude rotation.
        T = float(f_des @ R[:, 2])
        T = float(np.clip(T, self.T_min, self.T_max))
        S = 0.5 * (R_des.T @ R - R.T @ R_des)
        e_R = np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=np.float32)

        omega_des = -self.k_att * e_R
        n_omega = float(np.linalg.norm(omega_des))
        if n_omega > self.omega_max:
            omega_des = omega_des * (self.omega_max / n_omega)
        # Hard-cap the yaw rate (body-z) separately; roll/pitch keep their
        # ||omega||-clipped values.
        if self.yaw_rate_max is not None:
            omega_des = omega_des.copy()
            omega_des[2] = float(np.clip(omega_des[2],
                                         -self.yaw_rate_max, self.yaw_rate_max))

        return np.array([T, omega_des[0], omega_des[1], omega_des[2]],
                        dtype=np.float32)
