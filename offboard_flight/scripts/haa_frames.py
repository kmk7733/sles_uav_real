#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Frame adapters for the shared, fixed EKF local -> world alignment.

The EKF planner consumes SharedFrameAlignment, owned by ekf_world_align.py.
WorldToFcu below remains for compatibility with legacy callers only. Its
pair-based EMA estimator must not be used to re-estimate EKF-derived poses.
"""

import numpy as np

from ekf_alignment import SharedFrameAlignment


def planar_ekf_velocity(alignment, linear_local, angular_z):
    """MAVROS local ENU velocity -> map axes; preserve existing angular.z."""
    velocity_world = alignment.rotate_to_world(
        np.asarray(linear_local, dtype=np.float64))
    return float(velocity_world[0]), float(velocity_world[1]), float(angular_z)


def yaw_from_quat(x, y, z, w):
    """ZYX yaw from a quaternion."""
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_pi(a):
    return np.arctan2(np.sin(a), np.cos(a))


class WorldToFcu(object):
    """Estimates and applies vicon/world -> FCU local ENU."""

    def __init__(self, alpha=0.05, max_pair_age=0.3, min_updates=10):
        """
        alpha         EMA weight per update. Small on purpose: the true offset
                      only moves as fast as EKF2 drifts, so heavy smoothing
                      costs nothing and keeps sensor noise out of the setpoint.
        max_pair_age  reject a pair whose two stamps differ by more than this
                      [s]; a stale half would bias the estimate while moving.
        min_updates   how many accepted pairs before the transform is usable.
        """
        self.alpha = float(alpha)
        self.max_pair_age = float(max_pair_age)
        self.min_updates = int(min_updates)

        self.dyaw = None
        self.t = None
        self.n_updates = 0
        self.last_update = None
        self.last_residual = float("nan")

    @property
    def ready(self):
        return self.dyaw is not None and self.n_updates >= self.min_updates

    def update(self, p_world, yaw_world, t_world,
               p_fcu, yaw_fcu, t_fcu):
        """Fold in one matched pair of poses of the same body.

        p_* are (3,) positions, yaw_* radians, t_* message stamps in seconds.
        Returns True if the pair was accepted.
        """
        if abs(t_world - t_fcu) > self.max_pair_age:
            return False

        dyaw = wrap_pi(yaw_fcu - yaw_world)
        R = self._rz(dyaw)
        t = np.asarray(p_fcu, dtype=np.float64) - R.dot(np.asarray(p_world, dtype=np.float64))

        if self.dyaw is None:
            self.dyaw, self.t = dyaw, t
        else:
            # residual before folding in: how far the new pair sits from the
            # current estimate. Published as a health signal -- a growing
            # residual means the two sources disagree (Vicon dropout, EKF2
            # jump) and the alignment should not be trusted.
            self.last_residual = float(np.linalg.norm(t - self.t))
            a = self.alpha
            self.dyaw = wrap_pi(self.dyaw + a * wrap_pi(dyaw - self.dyaw))
            self.t = (1.0 - a) * self.t + a * t

        self.n_updates += 1
        self.last_update = max(t_world, t_fcu)
        return True

    def to_fcu(self, p_world, yaw_world=None):
        """Map a vicon/world point (and optionally a yaw) into the FCU frame."""
        if not self.ready:
            raise RuntimeError("world->FCU transform not established yet")
        p = self._rz(self.dyaw).dot(np.asarray(p_world, dtype=np.float64)) + self.t
        if yaw_world is None:
            return p
        return p, wrap_pi(yaw_world + self.dyaw)

    def rotate_to_fcu(self, v_world):
        """Rotate a FREE vector (velocity, acceleration) into the FCU frame.

        Velocities and accelerations are differences of positions, so the
        translation cancels and only the yaw rotation applies. Passing them
        through to_fcu() instead would add the offset t and turn a 1 m/s
        velocity reference into a nonsense several-metres-per-second one.
        """
        if not self.ready:
            raise RuntimeError("world->FCU transform not established yet")
        return self._rz(self.dyaw).dot(np.asarray(v_world, dtype=np.float64))

    def to_world(self, p_fcu, yaw_fcu=None):
        """Inverse map, for sanity checks and logging."""
        if not self.ready:
            raise RuntimeError("world->FCU transform not established yet")
        p = self._rz(-self.dyaw).dot(np.asarray(p_fcu, dtype=np.float64) - self.t)
        if yaw_fcu is None:
            return p
        return p, wrap_pi(yaw_fcu - self.dyaw)

    def age(self, now):
        return float("inf") if self.last_update is None else now - self.last_update

    def describe(self):
        if not self.ready:
            return "world->FCU: NOT READY (%d/%d pairs)" % (self.n_updates,
                                                            self.min_updates)
        return ("world->FCU: t=[%.3f %.3f %.3f] dyaw=%.1fdeg n=%d resid=%.3fm"
                % (self.t[0], self.t[1], self.t[2],
                   np.degrees(self.dyaw), self.n_updates, self.last_residual))

    @staticmethod
    def _rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0.0],
                         [s, c, 0.0],
                         [0.0, 0.0, 1.0]])
