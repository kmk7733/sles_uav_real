"""ROS/simulator-free V4 producer for the unchanged DeSimplex interface.

The provider receives copies of (state6, goal2) at the SAME observation anchor
and returns BodyActionChunk or None. No implicit conversion between a depth
anchor and a newer planner state occurs here. The real adapter must freeze
depth, PX4 state, goal, frame transform and epoch together.

Freshness must be checked on EVERY planning tick, including cache hits. Supply
validity_gate(state6, goal2), or perform that gate in the calling adapter before
calling plan. A cached reference is never evidence of a fresh sensor input.
"""
import numpy as np

from planner.haa.capped import CappedDynamics
from planner.types import MPPIResult, PlanarState, PlannerStatus
from planner.hpa.commit import commit_plan
from planner.hpa.reference import actions_to_reference, finite_array


class HPAProducer(object):
    def __init__(self, action_provider, limits, dt=0.1, chunk=10, commit=1,
                 goal_tol=0.25, validity_gate=None):
        if not callable(action_provider):
            raise TypeError("action_provider must be callable")
        if chunk != 10 or float(dt) != 0.1:
            raise ValueError("V4 contract requires ten action intervals at 0.1 s")
        if isinstance(commit, bool) or int(commit) != commit or int(commit) < 1:
            raise ValueError("commit must be a positive integer")
        if validity_gate is not None and not callable(validity_gate):
            raise TypeError("validity_gate must be callable")
        for name in ("v_max", "a_max", "omega_max", "alpha_max", "tilt_max", "j_max"):
            value = float(getattr(limits, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError("invalid limit %s" % name)
        if not np.isfinite(limits.a_max_eff) or limits.a_max_eff < 0:
            raise ValueError("invalid effective acceleration limit")
        if not np.isfinite(goal_tol) or float(goal_tol) < 0:
            raise ValueError("invalid goal tolerance")
        self.action_provider = action_provider
        self.validity_gate = validity_gate
        self.lim = limits
        self.dyn = CappedDynamics(limits, dt=float(dt))
        self.chunk = self.horizon = 10
        self.commit = int(commit)
        self.goal_tol = float(goal_tol)
        self.goal = np.zeros(2)
        self.n_calls = 0
        self.n_no_frame = 0
        method = type(self)._plan_once
        self._committed_plan = (method if self.commit <= 1 else
                                commit_plan(method, self.commit))

    def reset(self):
        """Legacy LearnedHPA reset is a no-op, including the commit cache."""

    def clear_commit(self):
        """Explicit epoch/stale-input invalidation, never an implicit switch."""
        self.__dict__.pop('_commit_state', None)

    def new_episode(self, goal=None):
        """Restore fresh-episode HPA state after a known lifecycle boundary."""
        new_goal = np.zeros(2) if goal is None else finite_array(goal, (2,), "goal")
        self.clear_commit()
        self.goal = new_goal
        self.n_calls = 0
        self.n_no_frame = 0

    @property
    def commit_index(self):
        st = getattr(self, '_commit_state', None)
        return None if st is None else int(st['i'])

    @staticmethod
    def _failed(reason):
        return MPPIResult(PlannerStatus.FAILED, None, None, None,
                          float('inf'), 0, 1, 0.0, reason)

    def _plan_once(self, state, goal=None, a_prev=None, **kw):
        self.goal = goal.copy()
        chunk = self.action_provider(state.copy(), goal.copy())
        if chunk is None:
            self.n_no_frame += 1
            return self._failed('no depth frame available yet')
        result = actions_to_reference(chunk, state, self.dyn, a_prev=a_prev)
        self.n_calls += 1
        return result

    def plan(self, state, goal=None, a_prev=None, **kw):
        """Produce MPPIResult; invalid observations never return a cached plan.

        Exception/finite handling below is a hardware input boundary, separate
        from the unchanged reference and commitment algorithm on valid inputs.
        """
        try:
            raw = state.to_array() if isinstance(state, PlanarState) else state
            xi = finite_array(raw, (6,), "planning state")
            g = finite_array(self.goal if goal is None else goal, (2,), "goal")
            prev = None if a_prev is None else finite_array(a_prev, (2,), "a_prev")
            if self.validity_gate is not None and not self.validity_gate(xi.copy(), g.copy()):
                raise ValueError("observation validity gate rejected this tick")
            return self._committed_plan(self, xi, goal=g, a_prev=prev, **kw)
        except Exception as exc:
            self.clear_commit()
            return self._failed("HPA input rejected: %s: %s" % (type(exc).__name__, exc))
