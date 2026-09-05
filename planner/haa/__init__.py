#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""High-assurance autonomy: the sampling MPPI producer and its parts.

    cost.py       FrontierMPPI -- owns the WHOLE cost, plus the geodesic swap
                  and the PA-MPPI frontier term. Subclasses `mppi.py`'s
                  PlanarMPPI and overrides `_cost`, so THAT file stays as it
                  flies (VENDOR.md) while the cost lives here.
    geodesic.py   CostToGo -- Dijkstra distance-to-goal over traversable space,
                  the navigation function that replaces ‖p − goal‖.
    capped.py     CappedDynamics -- the v_max/omega_max PROJECTION variant of
                  the rollout, selected by `mppi.cap_velocity`.

WHAT IS NOT HERE, DELIBERATELY
`safety.py` (the swept-path gate) and `types.py` sit one level up. The gate
validates whatever flies, not whatever MPPI produced: `planner/supervisor.py`
runs the learned HPA's proposals through the same object. If the validator
lived under `haa/`, the learned producer would have to import the classical one
just to be checked, and the two would stop being siblings.

Nothing here reads `config.yaml`. `planar_sim/build.py` does that translation.
"""
