"""Name the existing synchronous planner trajectory validator by its role.

It runs inside HAA/DeSimplex, using the planner occupancy snapshot. It is not
an additional parallel GT monitor for bare HPA and never commands the FCU.
"""

from planner.safety import PlanarSafetyValidator

TrajectorySafetyValidator = PlanarSafetyValidator

__all__ = ["TrajectorySafetyValidator"]
