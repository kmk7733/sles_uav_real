#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The planner. Two producers, one interface.

    planner/
        types.py       the contract: PlanarState in, MPPIResult out
        grid.py        the clearance oracle both producers are gated against
        safety.py      the swept-path validator, the only hard gate
        supervisor.py  DeSimplex: which producer flies this tick
        haa/           high-ASSURANCE autonomy -- planar MPPI, validated update
        hpa/           high-PERFORMANCE autonomy -- the behaviour-cloned policy

WHY THIS IS TOP LEVEL AND NOT INSIDE planar_sim
This code came FROM the vehicle -- `vendor/planner` is a copy of
`sles_uav_real/offboard_flight/scripts` at commit e45a4e5 (VENDOR.md) -- and it
is meant to go back. `planar_sim` is the test rig: harness, plants, perception,
arena, metrics, video. A directory that becomes a ROS node's import must not be
a subpackage of the simulator, or deploying it drags the simulator's `__init__`
and its siblings along.

    planar_sim  ->  planner        allowed, and the only direction
    planner     ->  planar_sim     FORBIDDEN. grep for it.

That rule is the whole point of the move, and it has one visible consequence:
anything that reads `config.yaml` stays on the simulator side. `build_planner`
and `build_supervisor` live in `planar_sim/build.py` and translate a `Config`
into the plain constructors this package exposes. `planner` never learns what
a config file is.

THE SHAPE IS THE ARGUMENT
`haa` and `hpa` are siblings, not layers. Both satisfy

    plan(state, goal, a_prev=None) -> MPPIResult

so `planar_sim/harness.py` cannot tell them apart, `supervisor.py` switches
between them tick by tick, and `planar_sim/metrics.py` measures both with one
definition. That is only true while neither imports the other, which is why
`types.py`, `grid.py`, `safety.py` and `supervisor.py` sit HERE rather than
under `haa/` -- they were the classical planner's files historically, but they
were never the classical planner's property.

THE DE-VENDORING IS DONE
`vendor/planner/` is gone. What flew from there now lives here under package
names, and this package is its own code rather than half a wrapper:

    types.py       was vendor/planner/planar_types.py     (code-identical)
    safety.py      was vendor/planner/planar_safety.py    (import path only)
    grid.py        was vendor/planner/planar_map.py       (plus edt_backend())
    dynamics.py    was vendor/planner/planar_dynamics.py
    mppi.py        was vendor/planner/planar_mppi.py
    scenarios.py   was vendor/planner/planar_scenarios.py (tests/test_planner.py only)

The extractions of `types`, `safety` and `grid` had been sitting here unused
since the split, and the honest note this replaces called them "dead weight
kept for the day someone finishes". They were not stale: `types.py` differed
from the vendored file by ZERO lines of code, `safety.py` by its own import
line, and `grid.py` only by additions. So finishing was a rename, not a merge.

`vendor/rotorpy/` stays vendored, and for the reason this tree no longer
qualifies for: it is genuinely upstream, tracked at v2.1.2, and a diff against
github.com/spencerfolk/rotorpy has to keep meaning something.

WHAT WENT WITH IT: the flat-module sys.path scaffolding at the bottom of this
file, and the same trick in `planar_sim/paths.py`. `from planar_types import
...` needed a directory on sys.path before anything could import it; `from
planner.types import ...` does not.

THIS DIRECTORY IS THE DEPLOYABLE UNIT, AND THAT IS CHECKED
Copy `planner/` on its own to the vehicle and it imports and solves. Verified
by copying the tree to an empty directory with no `planar_sim` anywhere and
running a real `FrontierMPPI.plan` -- WEIGHTED, 154/192 valid. Dependencies
are numpy and scipy (`scipy.ndimage.distance_transform_edt`, in `haa/cost.py`
and `grid.py`); nothing else.

What comes along, and it is the whole package rather than `haa/`:

    types.py grid.py safety.py dynamics.py mppi.py       the HAA needs these
    haa/cost.py haa/capped.py haa/geodesic.py
    supervisor.py transition.py hpa/                     DeSimplex adds these
    scenarios.py                                         tests only, droppable

WHAT DOES NOT COME, BY DESIGN: `build_planner` and `build_supervisor`. They
read `config.yaml` and so they live in `planar_sim/build.py` -- see the rule
above. A vehicle-side consumer writes its own translation from whatever it
configures with into the plain constructors here, and `build.py` is the
worked example of what has to be derived on the way (`limits_flown` = X (-) Z,
`mppi_sigma`, `r_eff`).

THE ONE EXCEPTION is `supervisor._main`, the self-test: it needs maps and a
config, so it imports `planar_sim` from inside the function and says so if it
cannot. The module-scope import graph stays clean, which is what the rule is
actually about.
"""

# (The flat-module sys.path scaffolding that used to be here is gone with
# vendor/planner. Nothing needs a path trick to import this package, and the
# four modules that still carried `import planner  # side effect: sys.path`
# have dropped it.)
