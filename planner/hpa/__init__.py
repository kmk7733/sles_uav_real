#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""High-performance autonomy: the behaviour-cloned producer. NOT MOVED YET.

The learned producer is still `hpa/policy.py` at the repo root, and this
directory is the place reserved for it rather than a copy of it. Two reasons it
did not move with the rest of the planner:

1. `hpa/policy.py` imports `hpa/model.py` for the network definition and the
   checkpoint loader, and `model.py` pulls torch. Moving `policy.py` alone would
   create `planner -> hpa` -- the reverse edge in a different disguise -- so
   `model.py` has to come with it, and the split between "the network the
   vehicle runs" and "the training code that produced it" has to be drawn
   first. Today `hpa/` holds both: `policy.py` and `model.py` are deployment,
   `process.py`, `train.py` and `make_report.py` are not.

2. The demonstration set is being re-collected (the learned policy parks short
   of the goal under supervision -- see docs/HAA_MPPI.md §8F), so `hpa/` is
   about to change underneath any move.

The interface is what matters and it already holds: `LearnedHPA` satisfies
`plan(state, goal, a_prev=None) -> MPPIResult`, which is why
`planner/supervisor.py` can switch to it without importing it -- the supervisor
takes a producer, it does not construct one.
"""
