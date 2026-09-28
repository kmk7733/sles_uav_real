"""Public role name for the simulator's unchanged DeSimplex implementation.

This is the switching/recovery/transition authority inside the planner, not
the independent Vicon CollisionStopGuard and not an FCU stop executor.
An alias preserves class identity, imports and the exact existing algorithm.
"""

from planner.supervisor import DeSimplexSupervisor

DeSimplexSwitcher = DeSimplexSupervisor

__all__ = ["DeSimplexSwitcher"]
