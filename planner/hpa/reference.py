"""Physical HPA actions to the simulator's planar reference, without ROS/torch.

The body frame is fixed at the observation anchor for all ten action nodes.
``anchor_yaw`` is expressed in the planning frame, independently of the raw PX4
yaw encoded in the V4 network's state. The caller supplies a planning state at
that same anchor; this module does not compensate for sensor or solve latency.
"""
import numpy as np

from planner.types import (IAL, NNU, U_ACC, MPPIResult,
                           PlanarReferenceSequence, PlannerStatus)


def finite_array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError("%s must be finite with shape %s" % (name, shape))
    return result.copy()


class BodyActionChunk(object):
    """Owned physical-unit V4 output plus its REQUIRED planning-frame yaw.

    Actions are [ax, ay, planar yaw acceleration], ten 0.1 s intervals. This
    object is not a position, velocity, thrust, or actuator command.
    """
    __slots__ = ("actions", "anchor_yaw")

    def __init__(self, actions, anchor_yaw):
        self.actions = finite_array(actions, (10, 3), "body actions")
        self.actions.setflags(write=False)
        if np.ndim(anchor_yaw) != 0 or not np.isfinite(anchor_yaw):
            raise ValueError("anchor_yaw must be an explicit finite scalar")
        self.anchor_yaw = float(anchor_yaw)


def world_actions(chunk, limits):
    """The rotation and envelope clipping in hpa.policy.LearnedHPA.act."""
    if not isinstance(chunk, BodyActionChunk):
        raise TypeError("provider must return BodyActionChunk with anchor_yaw")
    a = finite_array(chunk.actions, (10, 3), "body actions")
    psi = float(chunk.anchor_yaw)
    if not np.isfinite(psi):
        raise ValueError("nonfinite anchor_yaw")
    c, s = np.cos(psi), np.sin(psi)
    ax, ay = c * a[:, 0] - s * a[:, 1], s * a[:, 0] + c * a[:, 1]
    alpha = np.clip(a[:, 2], -limits.alpha_max, limits.alpha_max)
    cap = float(limits.a_max_eff)
    n = np.hypot(ax, ay)
    if cap > 0:
        scale = np.where(n > cap, cap / np.maximum(n, 1e-12), 1.0)
        ax, ay = ax * scale, ay * scale
    return np.stack([ax, ay, alpha], axis=1)


def actions_to_reference(chunk, state, dynamics, a_prev=None):
    """Exact LearnedHPA clipping/jerk/capped rollout order after inference.

    Finite/shape rejection is an input boundary, separate from the unchanged
    algorithm. CappedDynamics intentionally updates U in place during rollout.
    """
    xi = finite_array(state, (6,), "planning state")
    prev = None if a_prev is None else finite_array(a_prev, (2,), "a_prev")
    u = world_actions(chunk, dynamics.lim)
    U = np.zeros((10, NNU))
    U[:, U_ACC], U[:, IAL] = u[:, :2], u[:, 2]
    for k in range(10):
        U[k:k + 1] = dynamics.clip_inputs(
            U[k:k + 1], a_prev=(prev if k == 0 else U[k - 1, U_ACC]))
    X = dynamics.rollout(xi, U)[0]
    finite_array(U, (10, 3), "rolled inputs")
    finite_array(X, (11, 6), "rolled states")
    ref = PlanarReferenceSequence.from_rollout(X, U, dynamics.dt)
    for name in ("p", "v", "a", "psi", "psi_dot"):
        if not np.isfinite(getattr(ref, name)).all():
            raise ValueError("nonfinite reference %s" % name)
    return MPPIResult(PlannerStatus.WEIGHTED, ref, U, X,
                      0.0, 1, 1, 0.0, "learned HPA")
