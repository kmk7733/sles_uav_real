#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeSimplex runtime monitor: who flies this tick, the HPA or the HAA.

=========================================================================
STALE ARITHMETIC WARNING -- READ BEFORE TRUSTING A NUMBER IN THIS FILE.

The operating envelope changed when the PX4 limits landed (config.yaml
px4:, planar_sim/plants/px4_control.py). Every worked example below --
rho, the N_R analysis, the bridge-divergence table, the braking figures --
was computed at the OLD envelope:

                     was              now
    HPA a_max        8.23 m/s^2       5.0 m/s^2
    HPA j_max        26.3 m/s^3       8.0 m/s^3
    HAA a_max        5.66 m/s^2       3.5 m/s^2
    HAA j_max        18.1 m/s^3       5.5 m/s^3
    HPA omega_max    2.0 rad/s        0.5236 rad/s

The three consequences that are already MEASURED, and that move in
opposite directions:

  * rho FALLS, 0.520 -> 0.375 m at the 0.5 m/s ceiling, so the geometric
    hand-back condition gets EASIER: clearance >= 0.965 m instead of 1.11.
  * FULL-AUTHORITY BRAKING TAKES TWICE AS LONG, a_max/j_max 0.31 -> 0.625
    s, so N_R = 3 (0.3 s) is further from sufficient than it already was.
  * `brake_to_hover` NOW OVERSHOOTS. At 0.5 m/s the jerk bound cannot
    unwind the deceleration in time: the profile reverses to -0.22 m/s and
    the NET displacement `_brake_distance` returns (0.023 m) is 4.1x
    smaller than the furthest the vehicle actually gets (0.094 m). That is
    ~0.07 m of optimism in `d_return`, against an r_eff-dominated total of
    ~0.63 m. tests/test_switching.py records it.
  * N_BRIDGE = 6 is DERIVED FROM j_max (60 D / T^3, see its own note) and
    the divergence a bridge can absorb fell 0.095 -> 0.029 m. Restoring
    the old capability needs ~9 ticks.

NONE OF THIS IS FIXED HERE. The envelope step deliberately changed limits
and the plant only; supervisor and recovery parameters are a separate
piece of work, and re-deriving them in the same commit would make the
re-derivation unattributable. Treat the numbers below as the reasoning
that produced the current settings, not as a description of today.
=========================================================================

The learned HPA is fast and unverified; the MPPI HAA is slow, conservative and
validated. A Simplex architecture puts a decision module between them, and the
decision is not "does the HPA's plan look safe" -- it is "if the HPA is wrong
about this tick, can the HAA still catch the vehicle". That question is asked
about three sets, all evaluated against the CURRENT occupancy belief:

    S_HAA(p)   the safety envelope. States from which the HAA planner is
               FEASIBLE. Not modelled, MEASURED: the test is to call the HAA
               and ask whether it came back with anything other than FAILED.
               A model of the HAA's feasible set would be one more thing that
               can disagree with the HAA.

    R_Nr(p)    the recoverable region. States that can be driven INTO S_HAA
               within Nr steps using the FULL input authority (the `hpa:`
               block, not the HAA's limits), with every intermediate swept
               path clear of r_eff. This is the invariant the monitor
               maintains: while the vehicle is in R_Nr there is a known,
               validated manoeuvre back to a state the HAA can fly.

    M_Nm(p)    the safety-margin set. States in S_HAA that stay collision-free
               under Nm steps of ARBITRARY admissible action. Used for ONE
               thing: deciding when it is safe to hand control BACK to the
               HPA. Handing back on S_HAA membership alone would hand back at
               the boundary, where the next unverified action leaves it again.

DECISION, one tick:

    HPA mode:  the state reached by applying u_hpa is in R_Nr  -> keep the HPA
               otherwise -> fly the recovery manoeuvre into S_HAA, mode = HAA
    HAA mode:  x in M_Nm  -> mode = HPA
               otherwise  -> solve the HAA and stay

CHANGING AUTHORITY IS NOT THE SAME AS CHANGING THE REFERENCE, and `step`
decides only the first. Two independently planned trajectories agree about
nothing, so handing the geometric controller one and then the other puts a step
in position, velocity and acceleration through a loop whose whole job is to
reject steps. `plan` therefore spends `transition_steps` ticks on a quintic
bridge between them (planner/transition.py), validated with the same swept
obstacle check and the same input/state limits as any other trajectory. The
decision logic is untouched: the bridge changes HOW the reference gets to the
new producer, never WHETHER it does.

    HPA -> HAA   bridge invalid  -> the certified brake, immediately. The
                 rejected proposal is never flown during the transition.
    HAA -> HPA   bridge invalid  -> stay in HAA and ask again next tick.

The monitor starts in HAA mode. Handing an unverified policy the vehicle
before anything has been checked is the one ordering that cannot be justified,
and M_Nm is exactly the test that says when to stop doing it.

WHAT IS EXACT AND WHAT IS NOT
S_HAA is exact by construction -- it is the planner's own answer. M_Nm uses a
closed-form OVER-approximation of the forward reachable set, so it is
conservative. R_Nr uses a SUFFICIENT test (see `recovery`): it can refuse a
state that was in fact recoverable, and it can never accept one that was not.
Every approximation here therefore errs toward taking the vehicle off the HPA,
which is the direction a safety monitor is allowed to be wrong in.

IS M_Nm USEFULLY NON-EMPTY AT a_max = 8.23 m/s^2? YES, BUT IT IS THE THING
THE HPA's ACCELERATION CHOICE COSTS MOST.
rho = 0.5*0.3 + 0.5*8.23*0.09 = 0.52 m at the 0.5 m/s ceiling, so handing back
needs clearance >= 1.11 m inside a band only 5.0 - 2*0.59 = 3.82 m wide.
Measured over the collection families (self-test, per free cell of the
ground-truth map, geometric condition only):

    clutter/7  52.8%   clutter/12  61.0%   gap/3  41.5%
    corridor/5 19.2%   clutter/21  20.5%          -> mean 39.0% at v_max
                                                     mean 53.2% at rest

READ THE TWO COLUMNS CORRECTLY. Both are the GEOMETRIC condition alone
(clearance >= rho + r_eff) and neither is M_Nm. M_Nm requires S_HAA as well,
and `states_ok` bounds node 0 -- the measured state -- at the
HAA's own v_max. So a vehicle actually travelling at the HPA's 0.5 m/s is
outside S_HAA whatever its clearance, and

    M_Nm AT THE HPA's CRUISING SPEED IS EMPTY ON EVERY MAP.

Measured: of 25 cells that pass the geometric test at v_max, 0 are in M_3 by
the full test; membership steps from True at 0.35 m/s to False at 0.36. The
39.0% column therefore describes a set no state can be in, and only the 53.2%
"at rest" column bounds anything real -- itself still an upper bound, since it
too omits the S_HAA call.

This is not a defect in the decision logic, which never asks M_Nm of a fast
vehicle: M_Nm gates HAA -> HPA only, and in HAA mode the vehicle is already at
or below 0.35 m/s. It does mean hand-back can only happen after the HAA has
slowed the vehicle into its own envelope, which is the conservative order to do
it in. What it forecloses is a warm hand-back at speed.

The geometric figure still says where in the room hand-back is possible at all,
and it collapses exactly where it should:
19% inside a corridor, where the walls are 1.0-1.4 m apart and no forward
reachable disc of radius 0.52 m fits. A corridor is HAA territory under this
rule, permanently.

Two thirds of that 0.52 m is an artefact of the bound, not of the vehicle: see
`rho`, where capping the displacement at v_max*T -- which is equally admissible
and equally sound -- asks 0.74 m instead and lifts the mean to 77.0%. The
`a_max` choice is not what makes M_Nm tight; the closed form ignoring v_max is.

COST
Every S_HAA query is a full MPPI solve (K=192 samples, N=30 nodes, 6-13 ms
here depending on machine load). One R_Nr test costs between one and Nr+1 of
them, so the monitor is several times the price of the HAA it is protecting. That is the price of
testing feasibility rather than modelling it; measured numbers are in the
self-test at the bottom of this file.

    $PY -m planner.supervisor        # the self-test table
"""

import numpy as np

from planner.dynamics import PlanarDynamics, PlanarLimits
from planner.types import (IAL, IOM, NNU, NXI, S_POS, S_VEL, U_ACC, MPPIResult,
                          PlanarReferenceSequence, PlanarState, PlannerStatus)

from planner.transition import bridge_ok, bridge_reference, drop_first

# Slack on the `x' in X` gate in `in_s_haa`. The same 1e-6 that
# `states_ok` allows itself, and for the same reason: X's bounds are compared
# against norms that were just built out of floating-point arithmetic, so an
# exact-equality state must not fall on the wrong side of the test by an ulp.
# It is numerical slack and nothing else -- any real margin belongs in
# `safety.z_vel` / `z_omega`, which is what defines X_bar.
_X_TOL = 1e-6

# Defaults at the 10 Hz plan tick, i.e. 0.3 s of lookahead each.
#
# N_R = 3 IS ENOUGH FOR THE SPEED AND NOT FOR THE ACCELERATION. Stopping is
# cheap: zeroing 0.5 m/s in one step asks 5.0 m/s^2, well inside the HPA's
# 8.23 m/s^2 disc, and a 0.5 m/s state with a_prev = 0 is in R_1. What costs
# ticks is the ACCELERATION THE HAA HAS TO INHERIT. Its slew chain is anchored
# on a_prev, so an a_prev outside its own 5.66 m/s^2 disc pins its first
# command near that value (|a - a_prev| <= j_max dt = 1.81 m/s^2), which drives
# the next node past v_max 0.35 -- and `states_ok` then rejects every sample
# AND the braking fallback and returns FAILED. The state is outside S_HAA
# however calm its position and velocity look.
#
# MEASURED, self-test rim case: 0.5 m/s with a_prev on the 8.23 m/s^2 rim. The
# full-authority brake first OVERSHOOTS to 1.39 m/s -- the jerk bound needs
# 8.23/26.3 = 0.31 s, 3.1 ticks, merely to bring the acceleration to zero --
# reaches v = 0 at tick 6 with a_prev still at -6.69 m/s^2, and only settles
# into a state the HAA will accept at TICK 10. R_3 refuses it; R_10 accepts it.
#
# That is the monitor working rather than failing: it says the HPA may not
# spend the last of the acceleration envelope, because it cannot hand it back
# inside 0.3 s. Two ways to buy the behaviour back -- raise N_R to ~10 (1.0 s
# of committed braking, during which the HPA is doing nothing useful), or cap
# the HPA's commanded acceleration at the HAA's a_max_eff so every handover is
# inheritable by construction. The second is far cheaper and is the one to try.
#
# SET TO 2 TO MATCH THE REFERENCE INSTANTIATION
# (Turtlebot3_sles-main/planner_switch:1715, `N_r = 2`). The two are not quite
# the same test: the TurtleBot commits to exactly N_r braking steps and checks
# S_HAA once at the end, while `recovery` below SEARCHES k = 0..n_r and takes
# the first k that lands. Ours is therefore the more permissive condition at
# equal n_r, and shortening it to 2 pulls it back toward the reference.
#
# THE TENSION ABOVE IS SHARPENED BY THIS, not resolved: a shorter recovery
# horizon rejects the rim case sooner, which is the intended direction, but it
# also refuses an a_prev the HAA cannot inherit even faster. Capping the HPA's
# commanded acceleration at the HAA's a_max_eff remains the cheap fix, and is
# still not done.
N_R_DEFAULT = 2

# N_M = 3 mirrors it: the hand-back test looks exactly as far ahead as the
# recovery it is promising will be able to use.
N_M_DEFAULT = 3

# Extra radius [m] used ONLY when asking "is this still recoverable", never in
# flight and never by the HAA when it is actually planning.
#
# WHAT IT CHANGES. `recovery` asks whether a full-authority brake reaches a
# state the HAA can fly without hitting anything. At delta = 0 that question is
# "does the brake CLEAR the wall", and a brake that clears it by a centimetre
# passes -- so the HPA keeps the vehicle until the last moment that is still
# survivable. At delta > 0 the question becomes "does the brake clear it with
# delta to spare", which is strictly harder and therefore answered "no" EARLIER
# IN SPACE, while there is still room to act on the answer.
#
# WHY NOT JUST RAISE r_eff. Because r_eff decides what is flyable at all:
# pushing it to 0.52 makes narrow gaps geometrically impassable (see
# docs/HAA_MPPI.md 8D on the 0.46/0.54 traversability collision). This margin
# costs no flyable space -- the HAA still plans and flies at r_eff -- it only
# makes the monitor hand over sooner.
#
# 0.06 IS THE REFERENCE INSTANTIATION'S VALUE:
# Turtlebot3_sles-main/planner_switch:1727 plans the HAA feasibility check with
# `robot_radius = self.robot_radius + 0.06` under the comment "Check with even
# tightened constraints", while every other call uses `self.robot_radius`.
#
# MEASURED HERE, gap/61015, first tick `recovery` refuses (wall at y = 0.10):
#     delta 0.00 -> y = +0.27   0.17 m PAST the gap
#     delta 0.06 -> y = -0.17   0.27 m BEFORE it
#     delta 0.12 -> y = -0.36   0.46 m before it
#
# ------------------------------------------------------------------ SET TO 0
# THE MARGIN IS ALREADY IN r_eff, AND COUNTING IT TWICE COSTS ROOMS.
#
# The TurtleBot's `robot_radius` is the airframe and nothing else, so its +0.06
# is the only uncertainty allowance in the recoverability test. Ours is not:
#   r_eff 0.46 = r_quad 0.31 + r_perc 0.10 + tracking
# and r_perc is the p99 of a MEASURED depth error (experiments/depth_error.py
# part A2). Adding delta on top asks the recoverability test to carry the same
# uncertainty a second time.
#
# What delta actually tightens is BOTH halves of `_recovery_search`, not just
# the braking sweep -- `build_supervisor` gives the probe and the monitor the
# SAME validator object (planar_sim/build.py:114-116), so raising r_safe also
# raises the radius the probe's own samples are validated at. The question
# delta really asks is therefore "is the braked state in S_HAA when S_HAA is
# evaluated at r_eff + delta", which is a much stronger demand than "does the
# brake clear the wall".
#
# MEASURED COST, 6 mixed-obstacle rooms x 3 deltas (runs/ds_switch/ds_delta.py,
# with mppi.cap_velocity on):
#     delta 0.00 -> 6/6 reached, switches 1/1/1/1/3/3
#     delta 0.06 -> 5/6         , mixed/51004 times out, 12 switches
#     delta 0.10 -> 5/6         , identical failure
# 51004's only route to its goal runs through a slot with 1.4 cm of lateral
# slack. delta 0.06 does not make the monitor hand over sooner there -- it puts
# the braked state outside S_HAA outright, and the vehicle oscillates at the
# mouth of the gap for the whole episode.
#
# WHAT PAYS FOR delta = 0: the braking model was flown on the 6-DOF plant under
# the config's uncertainty (5% mass/inertia, 0.3 m/s wind, aero). Real stopping
# distance overran the plan by 4% -- 2 mm at 0.5 m/s -- and peak tilt was 14
# degrees against a 63 degree airframe limit. A margin sized to that measured
# error is ~0.01 m, not 0.06.
RECOVERY_MARGIN_DEFAULT = 0.0

# Length of the polynomial bridge flown on a mode switch, in plan ticks
# (planner/transition.py). 6 ticks = 0.6 s at the 10 Hz plan rate.
#
# THE ONLY KNOB THIS FEATURE ADDS, and it is one rather than two because a
# bridge is a bridge: nothing about leaving the HPA makes the join harder than
# entering it, so a separate HPA->HAA length would be a parameter with no
# question behind it.
#
# THE FLOOR IS THE JERK BOUND, and it is much higher than it looks. A quintic
# closing a lateral divergence D from rest to rest peaks at 60 D / T^3, and
# `bridge_ok` rejects anything over `j_max` because `inputs_ok` does. At
# j_max = 26.3 m/s^3 that is
#
#     ticks   T       divergence the bridge can absorb
#       3    0.3 s    0.012 m      <- useless
#       4    0.4 s    0.028 m
#       6    0.6 s    0.095 m      <- default
#       8    0.8 s    0.224 m
#
# Two trajectories planned from the same state diverge by centimetres over
# half a second, so 3 ticks would reject almost every bridge and fall through
# to the brake -- a smoothing feature that makes the ride WORSE. 6 is the
# shortest that reliably clears the bound.
#
# ZERO IS LEGAL AND MEANS "BRAKE AND SWITCH". It is not the bridge turned
# off; it is the transition the bridge was added to replace, kept runnable so
# the two can be measured against each other rather than argued about. See
# `plan`, and runs/ds_switch_ab/ for the comparison.
#
# n_m IS NOT THE CEILING, which is the tempting mistake. M_Nm buys Nm steps of
# ARBITRARY admissible action; a bridge is not arbitrary, it is one curve
# checked against the map before it is flown and re-checked every tick after
# (`_advance_bridge`). What a long bridge actually costs is steering: for its
# duration neither planner is closing on the goal.
TRANSITION_STEPS_DEFAULT = 6

# Solve the FLYING HAA on every tick, including the ones the HPA is flying,
# and discard the answer. Off by design in the shipped monitor; the question it
# answers is what a takeover costs.
#
# WHAT A TAKEOVER COSTS, MEASURED. 109 takeovers on the 41 held-out map seeds
# (runs/ds_switch_ab/bridge6), binned by time since the HAA was handed the
# vehicle -- the rate at which the distance to the goal actually falls:
#
#       0-1 s   +0.021 m/s        4-5 s   +0.052
#       1-2 s   -0.037   <- AWAY  6-7 s   +0.075
#       2-3 s   +0.019            8-9 s   +0.140
#
# against +0.151 m/s for the SAME planner flying the same rooms alone. The HAA
# is not slow; it is slow FOR THE FIRST SIX SECONDS AFTER IT IS HANDED
# SOMETHING. At 2.7 takeovers per episode that transient, not the 0.4 s
# transition, is where a switched flight loses to an unswitched one -- and it
# is why DeSimplex's median 36.1 s sits ABOVE the HAA's own 32.6 s while the
# HPA flies 74% of the ticks.
#
# WHY IT HAPPENS. `step` calls `self.haa.reset()` on the way back in, which
# zeroes `U_nom`. MPPI's nominal IS its warm start: a fresh one is the hover
# input, so the first solves after a takeover search around "do nothing" while
# the vehicle is in the one state the HPA could not keep. The reset is there
# for `_last_ref`, not for `U_nom` -- a stale PREVIOUS fallback is a rescue
# that will not happen -- and solving every tick fixes the stale reference
# without throwing the nominal away, so the reset is not needed when this is on.
#
# WHAT IT COSTS: one extra MPPI solve on ~74% of ticks. That is real -- it is
# the same order as the probe -- and it buys nothing at all in HAA mode, where
# the planner is already being solved.
WARM_HAA_DEFAULT = False

# How many nodes of the HPA'S OWN COMMITTED PLAN are gated before its first
# input is flown. 1 is the shipped monitor: it checks the tick in front of it
# and nothing else.
#
# WHY MORE THAN ONE IS NOT A LONGER LEASH BUT A SHORTER ONE. The learned HPA
# commits an 8-node chunk (hpa/policy.py; the checkpoint's horizon) and is not
# consulted again until it is spent, so a violation at node 5 is ALREADY
# DECIDED when node 0 is flown -- the monitor simply does not look, meets it
# five ticks later, and takes the vehicle at the last moment that is still
# recoverable. Gating the whole committed path moves the SAME takeover 0.5 s
# earlier, with the obstacle further away. It adds no takeovers, because the
# plan that will violate is the plan that was going to be flown.
#
# IT IS STRICTLY A STRICTER GATE, which is what makes it free of proof
# obligations: the one-tick test still runs, unchanged, and this is an
# additional conjunct. A state the shipped monitor refuses is still refused.
#
# WHAT IS CHECKED IS THE REFERENCE, NOT A ROLLOUT OF THE INPUTS. The controller
# tracks `res_hpa.reference`, `shifted()` pads its tail by REPEATING the last
# node with the derivatives zeroed, and a held chunk's padding is therefore a
# stationary point -- harmless to a swept check. Rolling the INPUT sequence
# forward instead would coast through the zero-padded tail and reject the last
# tick of every chunk for a flight that was never going to happen.
N_LOOK_DEFAULT = 1

# Try "decelerate into the HAA's envelope" as a recovery candidate BEFORE the
# full brake to hover.
#
# WHAT IS WRONG WITH ONLY HAVING THE BRAKE. `recovery` is explicit that it
# tries ONE candidate manoeuvre and that the error is one-sided, so adding a
# candidate cannot make it accept anything it should not -- the new manoeuvre
# is validated by the same swept check, the same input chain and the same
# S_HAA probe. What it changes is which states are found recoverable, and how
# fast the vehicle is when the HAA is handed it.
#
# MEASURED, 109 takeovers, speed against time from the takeover tick:
#
#     -1.00 s  0.171     +0.25 s  0.141
#     -0.50 s  0.208     +0.50 s  0.102   <- 58% of the speed, gone in 0.5 s
#      0.00 s  0.240     +3.00 s  0.169
#
# The vehicle is ACCELERATING into the takeover and is at 0.240 m/s when it
# happens -- comfortably inside the HAA's own 0.35 m/s ceiling, so nothing
# about the fallback's envelope required that deceleration. It takes 3 s to get
# back to 0.17 and 8 s to close on the goal at the rate the same HAA sustains
# flying the same rooms alone. That transient, times 2.7 takeovers, is 17 s an
# episode against the 11 s the HPA's speed is worth.
#
# The target is READ OFF THE FALLBACK (`self.haa.dyn.lim`), never written down
# here: the manoeuvre exists to make the state flyable by that planner, so the
# speed it aims at is that planner's, whatever config.yaml says it is today.
HANDOVER_DECEL_DEFAULT = False

# Seed the HAA's MPPI nominal with the braking manoeuvre instead of leaving it
# at zero, whenever it is about to solve with an empty one.
#
# ZERO IS NOT HOVER FOR A MOVING VEHICLE. The input is an ACCELERATION, so the
# fresh nominal `reset` installs is "no acceleration" -- fly straight, at the
# speed you have, for the whole horizon. `reset`'s own note says it "coasts
# rather than falls", which is the same fact read as reassurance. At the 0.24
# m/s the vehicle is doing when the monitor takes it, 60 nodes of coasting is
# 1.44 m of straight line, and the median belief clearance at that moment is
# 0.50 m: the nominal goes through a pillar, and so does most of the noise
# cloud around it.
#
# WHAT THAT COSTS, MEASURED over 57 takeovers -- what the HAA's solver actually
# returned, by tick since it was handed the vehicle:
#
#     +0   BRAKING 66%   WEIGHTED 32%
#     +1   PREVIOUS 60%  FAILED 32%
#     +4   PREVIOUS 56%  FAILED 21%   WEIGHTED 23%
#     +7   WEIGHTED 65%
#
# It is not planning for the better part of a second, and it is not planning
# because NO SAMPLE IS VALID (`FAILED` is n_valid == 0, a hard-check count, not
# a cost). Worse, the cascade is self-sustaining: `_fallback_previous` sets
# `U_nom` back to zeros, so the next tick starts from the same empty nominal
# and re-serves the brake it emitted at +0. That is the 58% speed collapse.
#
# WHY THE BRAKE IS THE RIGHT SEED. It is the manoeuvre `recovery` has just
# VALIDATED -- swept-collision-free at r_eff, input-feasible, ending in S_HAA
# -- so sample 0 is a valid sample by construction and `n_valid >= 1` stops
# being a hope. And a nominal is not a commitment: `plan` hard-checks every
# sample and re-checks the damped answer (`_candidate_ok`), so seeding cannot
# make anything admissible that was not. It only decides where MPPI looks.
#
# SEEDED ON EMPTINESS, NOT ON THE SWITCH, because the fallback path re-empties
# it. The condition is exactly "the sampler is about to be handed no warm start
# at all".
SEED_NOMINAL_DEFAULT = False

MODE_HPA = "HPA"
MODE_HAA = "HAA"

# Where the tick's input came from. Distinct from PlannerStatus, which says
# which branch of the MPPI safety chain produced a trajectory; this says which
# AUTHORITY produced it, and the two are orthogonal.
SRC_HPA = "HPA"              # the unverified policy, verified recoverable
SRC_RECOVERY = "RECOVERY"    # the braking manoeuvre back into S_HAA
SRC_HAA = "HAA"              # the validated planner


class Verdict(object):
    """The three set memberships at one state, with the evidence for each."""

    __slots__ = ("s_haa", "r_nr", "m_nm", "clearance", "rho", "k_recover",
                 "haa_status", "n_probe", "probe_ms")

    def __init__(self, s_haa=False, r_nr=False, m_nm=False, clearance=0.0,
                 rho=0.0, k_recover=-1, haa_status="", n_probe=0,
                 probe_ms=0.0):
        self.s_haa = bool(s_haa)
        self.r_nr = bool(r_nr)
        self.m_nm = bool(m_nm)
        self.clearance = float(clearance)
        self.rho = float(rho)
        self.k_recover = int(k_recover)   # steps of braking needed, -1 = never
        self.haa_status = haa_status
        self.n_probe = int(n_probe)       # HAA solves this verdict cost
        self.probe_ms = float(probe_ms)

    def __repr__(self):
        return ("Verdict(S_HAA=%s R=%s M=%s  clear=%.2f rho=%.2f k=%d %s)"
                % (self.s_haa, self.r_nr, self.m_nm, self.clearance, self.rho,
                   self.k_recover, self.haa_status))


class Decision(object):
    """What the monitor decided for one tick, and why."""

    __slots__ = ("mode", "source", "U", "X", "verdict", "reason", "fault",
                 "result", "switched")

    def __init__(self, mode, source, U=None, X=None, verdict=None, reason="",
                 fault=False, result=None, switched=False):
        self.mode = mode          # the mode the NEXT tick starts in
        self.source = source      # SRC_HPA / SRC_RECOVERY / SRC_HAA
        self.U = U                # (n, 3) input sequence to fly
        self.X = X                # (n+1, 6) the states it produces
        self.verdict = verdict
        self.reason = reason
        # fault = the assurance argument has already been violated: no
        # recoverable action was found. The monitor still emits its best
        # braking effort, and says so rather than reporting a clean handover.
        self.fault = bool(fault)
        self.result = result      # MPPIResult, when the HAA produced the tick
        self.switched = bool(switched)

    def __repr__(self):
        return ("Decision(%s via %s%s, %r)"
                % (self.mode, self.source, "  FAULT" if self.fault else "",
                   self.reason))


class DeSimplexSupervisor(object):
    """The runtime monitor. One `step` per plan tick.

    It is also a producer in the `plan(state, goal, a_prev) -> MPPIResult`
    sense (see hpa/README.md), so a supervised HPA drops into the same harness
    as bare MPPI and is measured by the same metrics.
    """

    def __init__(self, haa, probe, dyn_full, validator, goal,
                 n_r=N_R_DEFAULT, n_m=N_M_DEFAULT, hpa=None, probe_seed=0,
                 transition_steps=TRANSITION_STEPS_DEFAULT,
                 recovery_margin=RECOVERY_MARGIN_DEFAULT,
                 warm_haa=WARM_HAA_DEFAULT, n_look=N_LOOK_DEFAULT,
                 handover_decel=HANDOVER_DECEL_DEFAULT,
                 seed_nominal=SEED_NOMINAL_DEFAULT, lim_x=None):
        """
        haa        the HAA planner that actually flies the vehicle in HAA mode.
                   Stateful: it warm-starts from its own last answer.
        probe      a SECOND HAA planner, identical settings, used only to
                   answer "is this state in S_HAA". It has to be separate. A
                   membership query is a full `plan()` call, and `plan()`
                   mutates `U_nom` and `_last_ref` -- probing through the
                   flying planner would overwrite the warm start with the
                   answer to a hypothetical, and would let the PREVIOUS
                   fallback report success for a state it never planned from.
        dyn_full   PlanarDynamics at the FULL input authority (the `hpa:`
                   block). The recovery manoeuvre is the HPA's to fly, so it
                   is entitled to the HPA's envelope -- braking under the
                   HAA's smaller disc would be a slower stop than the vehicle
                   is actually capable of, and would shrink R_Nr for no reason.
        validator  PlanarSafetyValidator. THE SAME OBJECT the planners hold,
                   so a map update reaches all three at once.
        goal       (gx, gy), needed because S_HAA membership is a plan call.
        n_r, n_m   the two horizons, in plan ticks.
        hpa        optional producer with `plan(state, goal, a_prev)`. Only
                   `plan()` needs it; `step()` takes the HPA's input directly.
        transition_steps
                   length of the bridge flown on a mode switch, in plan ticks.
                   Only `plan()` uses it -- `step()` is an input-level
                   interface with no reference to be continuous in.
        recovery_margin
                   extra radius the RECOVERABILITY test alone is evaluated at.
                   See RECOVERY_MARGIN_DEFAULT: it is what makes the monitor
                   take the vehicle off the HPA before a tight passage rather
                   than after it.
        warm_haa   solve the flying HAA every tick, including the ticks the HPA
                   is flying, and throw the answer away. See WARM_HAA_DEFAULT:
                   it is what the HAA costs on the tick it is handed a vehicle.
        n_look     nodes of the HPA's committed plan gated before its first
                   input is flown. See N_LOOK_DEFAULT.
        handover_decel
                   offer "decelerate into the HAA's envelope" as a recovery
                   candidate ahead of the brake. See HANDOVER_DECEL_DEFAULT.
        seed_nominal
                   never let the HAA sample around an empty nominal. See
                   SEED_NOMINAL_DEFAULT.
        lim_x      the UNTIGHTENED state set X (the `limits:` block), against
                   which `in_s_haa` tests eq. (33)'s `x' in X`. The planners
                   carry X_bar = X (-) Z and cannot answer this: a state
                   between the two is legitimately in S_HAA -- that gap is
                   exactly what eq. (16) buys -- while a state past X is not,
                   whatever the planner says. None falls back to the probe's
                   own X_bar, which is STRICTER than the paper and so still
                   never launders a violation; it just refuses the eq. (16)
                   band as well. `build_supervisor` always passes the real X.
        """
        self.haa = haa
        self.probe = probe
        self.lim_x = lim_x if lim_x is not None else probe.dyn.lim
        self.dyn_full = dyn_full
        self.validator = validator
        self.goal = np.asarray(goal, dtype=np.float64).reshape(-1)[:2]
        self.n_r = int(n_r)
        self.n_m = int(n_m)
        self.hpa = hpa
        self.probe_seed = int(probe_seed)
        self.transition_steps = int(transition_steps)
        self.recovery_margin = float(recovery_margin)
        self.warm_haa = bool(warm_haa)
        self.n_look = max(1, int(n_look))
        self.handover_decel = bool(handover_decel)
        self.seed_nominal = bool(seed_nominal)
        self._brake_cache = {}

        # THE TRANSITION BUFFER. Non-None while a bridge is being flown:
        # (reference, nodes_left, target_mode). `plan` walks it out one node
        # per tick and only then activates `target_mode`, so the destination
        # planner never re-plans from a state halfway along the bridge -- which
        # would produce exactly the discontinuity the bridge exists to remove.
        self._bridge = None
        # The reference this supervisor last handed the controller. The bridge
        # starts from ITS node 0, not from the measured state, so the join
        # carries no tracking error.
        self._last_ref = None

        # START IN HPA MODE.
        #
        # The R_Nr GATE STILL RUNS ON THE FIRST INPUT, which is what makes this
        # safe: the HPA branch refuses any u_hpa whose successor state is not
        # recoverable, on tick 0 exactly as on every other tick. What starting
        # in HPA skips is the M_Nm margin test, and only at t = 0 -- a state
        # the episode was placed in by `sample_start_goal`, which already
        # screened it for clearance at the flying radius.
        #
        # It used to start in HAA so that "the HPA is handed the vehicle only
        # after M_Nm has been checked once". The cost of that was a switch at
        # t = 0 in EVERY episode: the monitor entered tick 0 in HAA, passed
        # M_Nm immediately, and handed over. That is not an authority change
        # the vehicle experienced -- it is an artefact of the initial value --
        # and it made every reported switch count one too high, including the
        # `sw 1` episodes in runs/ds_switch/ds_delta.png, which really switched
        # zero times.
        self.mode = MODE_HPA
        self.n_ticks = {MODE_HPA: 0, MODE_HAA: 0}
        self.n_switch = {MODE_HPA: 0, MODE_HAA: 0}
        self.n_fault = 0
        self.n_probe = 0
        self.n_bridge = 0          # bridges opened
        self.n_bridge_abort = 0    # bridges the map invalidated mid-flight
        # Switches that wanted a bridge and could not have one. NOT a fault:
        # it is the designed fallback, and counting it is the only way to tell
        # "the transition was smoothed" from "the transition was braked".
        self.n_bridge_refused = 0
        self.n_shadow = 0          # solves spent keeping the HAA warm
        self.n_look_refused = 0    # takeovers the LOOK-AHEAD caused, not the tick
        self.n_decel = 0           # recoveries the deceleration candidate won
        self.n_seed = 0            # solves handed a brake instead of nothing
        self.probe_s = 0.0
        self.last_decision = None

    # ------------------------------------------------------------ construction

    # `from_config` USED TO LIVE HERE and is now
    # `planar_sim.build.build_supervisor`. It read `config.yaml` and called
    # `run_planar_sim.build_planner`, so this module -- the decision logic --
    # imported the simulator's CLI entry point, which in turn imports the cost
    # module this file also uses. Survivable only because the import was
    # deferred inside the method. The monitor takes two producers and a
    # validator and knows nothing about where they came from; that is exactly
    # what lets it switch to a learned HPA it could not have constructed.

    def set_occupancy(self, occ):
        """Point every consumer at the new belief. Call it once per tick."""
        self.validator.occ = occ
        self.haa.validator.occ = occ
        self.probe.validator.occ = occ

    # -------------------------------------------------------------- geometry

    @property
    def dt(self):
        return float(self.dyn_full.dt)

    @property
    def goal_tol(self):
        """Part of the producer interface `plan()` claims to satisfy.

        Harness.run reads `planner.goal_tol` to decide arrival, so without this
        the supervisor drops into Harness in every respect except actually
        running. It is the HAA's tolerance and not a value of its own: the two
        modes have to agree on where the goal is, or a handover could change
        the arrival test mid-flight.
        """
        return float(self.haa.goal_tol)

    @property
    def r_eff(self):
        return float(self.validator.r_safe)

    def rho(self, speed, speed_capped=False):
        """Radius of the Nm-step forward reachable set [m], over-approximated.

            rho = v T + 0.5 a_max T^2,      T = n_m dt

        the double integrator's reachable disc under |a| <= a_max, ignoring
        every other constraint. At n_m = 3, dt = 0.1 and the HPA's 8.23 m/s^2
        that is 0.37 m from rest and 0.52 m at the 0.5 m/s ceiling, so handing
        control back needs clearance >= rho + r_eff = 0.96 to 1.11 m.

        THAT BOUND IGNORES THE SPEED LIMIT, AND THE SPEED LIMIT IS THE BINDING
        ONE HERE. |v| <= v_max is as much a part of admissibility as |a| <=
        a_max, and v_max/a_max = 0.5/8.23 = 0.061 s is a fifth of T = 0.3 s:
        the vehicle spends 5/6 of the window already saturated in speed, so it
        cannot travel further than v_max T = 0.15 m however the acceleration is
        chosen. The acceleration term above is therefore accounting for 0.37 m
        of displacement that no admissible input can produce -- a factor 3.5
        over-conservative, and it is what makes M_Nm demanding in a 3.8 m band.

        `speed_capped=True` returns min(the two), which is still a valid
        over-approximation and is the honest one. It is NOT the default: the
        specified formula is the conservative one, and a monitor should not
        quietly tighten its own margin. The self-test reports both.
        """
        T = self.n_m * self.dt
        a = float(self.dyn_full.lim.a_max_eff)
        r = float(speed) * T + 0.5 * a * T * T
        if speed_capped:
            r = min(r, float(self.dyn_full.lim.v_max) * T)
        return r

    # ---------------------------------------------------------------- S_HAA

    def in_s_haa(self, xi, a_prev=None):
        """Is `xi` a state the HAA planner can fly from? -> (bool, MPPIResult).

        THE TEST IS THE PLANNER. Anything short of running it is a model of
        the HAA's feasible set, and a model that disagrees with the HAA is
        worse than no model: it either promises a rescue that will not happen
        or refuses states the HAA would have handled.

        `reset()` first, so the probe cannot answer out of the PREVIOUS
        fallback -- that branch re-uses the last accepted trajectory, which was
        planned from a different state and says nothing about this one. What
        survives the reset is the SAMPLES and the BRAKING fallback, and BRAKING
        counts: "the HAA can bring this to a validated stop" is precisely what
        membership of the safety envelope means.

        The RNG is reset too. Without that, membership is a fresh random draw
        every call and the same state can be in the envelope on one tick and
        out on the next -- a monitor whose verdicts are not reproducible cannot
        be argued about after the fact.

        THE STATE-BOUND GATE IS OURS, NOT THE PLANNER'S. Eq. (33) opens with
        `x' in X`, so a state outside the physical envelope is outside S_HAA
        whatever a solve would say. This used to be left to `states_ok`
        checking node 0 -- "a state above the HAA's v_max is
        outside S_HAA by construction, samples AND braking alike" -- and that
        stopped being true when `FrontierMPPI._project_start` began repairing
        the start state. The planner's repair is eq. (16) and is correct on its
        own terms; what was wrong was letting the SIZE of that repair stand in
        for a membership test, because then one constant in the cost module
        decided the supervisor's switching signal. It is tested here instead,
        where it costs nothing: no solve, and the answer does not depend on how
        generous the planner chooses to be.

        Note the gate is X, not X_bar. A state between the two IS in S_HAA --
        the planner picks a nominal inside X_bar within Z of it, which is
        precisely what eq. (16) exists for. Only past X is membership lost, and
        there `x_k in x0_bar (+) Z` can no longer be satisfied, so the tube
        invariance (8) has no premise and (13)'s `x_k in X` is not established.
        That is also why the RECOVERY manoeuvre is checked differently (see
        `recovery`): recovery is what runs precisely when this fails.
        """
        import time as _time
        xi = np.asarray(xi, dtype=np.float64).reshape(NXI)

        # Cheap gates first: skipping the solve saves ~10 ms.
        # A state whose own position is inside the inflated set cannot be in
        # S_HAA.
        if float(self.validator.clearance(xi[S_POS])) < self.r_eff:
            return False, None
        # Nor can one outside the physical envelope -- eq. (33)'s `x' in X`.
        if float(np.linalg.norm(xi[S_VEL])) > self.lim_x.v_max + _X_TOL:
            return False, None
        if abs(float(xi[IOM])) > self.lim_x.omega_max + _X_TOL:
            return False, None

        self.probe.reset()
        self.probe.rng = np.random.RandomState(self.probe_seed)
        t0 = _time.time()
        res = self.probe.plan(xi, self.goal, a_prev=a_prev, warm_shift=0)
        self.probe_s += _time.time() - t0
        self.n_probe += 1
        return bool(res.status != PlannerStatus.FAILED), res

    # ----------------------------------------------------------------- R_Nr

    def recovery(self, xi, a_prev=None):
        """Braking manoeuvre from `xi` into S_HAA, or None. THE R_Nr TEST.

        Returns (k, U[:k], X[:k+1], result) where k is the number of steps
        needed, k = 0 meaning `xi` is already in S_HAA.

        A SUFFICIENT CONDITION, NOT AN EXACT ONE. Exact backward reachability
        of S_HAA is intractable -- S_HAA is only defined pointwise, by running
        a sampling planner -- so this tries ONE candidate manoeuvre, the
        full-authority brake to hover, and asks whether that particular one
        lands in the envelope. A state that some other input sequence could
        have recovered, but braking cannot, is reported as not in R_Nr. The
        error is therefore one-sided and points the safe way: the test MAY
        REFUSE A RECOVERABLE STATE, IT CAN NEVER ACCEPT AN UNRECOVERABLE ONE.

        WHAT THE MANOEUVRE IS CHECKED AGAINST, and what it is deliberately not.
        Collision (swept, against r_eff) and INPUT feasibility (the
        acceleration disc, the jerk chain anchored on a_prev) -- both under the
        FULL envelope, because the recovery is flown by the HPA-side authority.
        It is NOT checked against the state bounds. Requiring a braking
        manoeuvre to already satisfy the speed limit it exists to restore is
        circular, and `states_ok` tests |v| <= v_max over ALL
        nodes INCLUDING node 0, the measured initial state -- so routing this
        through `PlanarMPPI._candidate_ok` would reject every deceleration that
        starts above the limit, which is every deceleration worth having.
        Measured in this repo: 0.31 m/s against a 0.30 m/s bound is enough to
        take out the samples and the braking fallback together and return
        FAILED. `dynamics.py` is not patched; the check is done here.

        WHAT ACTUALLY DECIDES k IS a_prev, NOT THE SPEED. Each probe is given
        the acceleration flown INTO that node, because that is what anchors the
        HAA's slew chain once it takes over -- and an a_prev outside the HAA's
        own disc makes the state infeasible for it no matter how slowly the
        vehicle is moving. See the N_R_DEFAULT note: on the rim case the
        velocity is back to zero at tick 6 and the state is still not in S_HAA
        until tick 10, purely because of the acceleration being handed over.
        """
        xi = np.asarray(xi, dtype=np.float64).reshape(NXI)
        cands = []
        if self.handover_decel:
            cands.append(self._decel_to_haa(xi, a_prev=a_prev))
        cands.append(self.dyn_full.brake_to_hover(xi, a_prev=a_prev,
                                                  horizon=self.n_r))
        for i, U in enumerate(cands):
            X = self.dyn_full.rollout(xi, U)[0]          # (n_r+1, 6)
            # k = 0 IS THE SAME QUESTION FOR EVERY CANDIDATE -- it is the state
            # itself, before any of them has been flown -- so only the first
            # one pays for that probe.
            got = self._recovery_k(U, X, a_prev, k0=(0 if i == 0 else 1))
            if got is not None:
                if i == 0 and self.handover_decel and got[0] > 0:
                    self.n_decel += 1
                return got
        return None

    def _decel_to_haa(self, xi, a_prev=None, horizon=None):
        """Input sequence that decelerates INTO the HAA's envelope. (N, 3)

        `brake_to_hover` with a floor, and built the same way: ask for the
        acceleration that would meet the target in one dt, let `clip_inputs`
        cut it to what the disc and the slew chain allow, step, repeat. Once
        the state is inside the envelope the request is zero and the chain
        slews the deceleration back out, so the manoeuvre ENDS COASTING AT THE
        FALLBACK'S OWN CEILING rather than continuing to a stop.

        WHY THIS IS A CANDIDATE AND NOT A REPLACEMENT. It is weaker than the
        brake -- it gives up less speed, so it clears less ground -- and there
        are states only the brake recovers. `recovery` tries this first and
        falls through, which is the same one-sided logic it already had: a
        manoeuvre is accepted only by landing in S_HAA, and offering a second
        one can only find states the first one missed.

        THE TARGET IS THE FALLBACK'S OWN LIMIT OBJECT. S_HAA membership is a
        solve by `self.haa`, whose `states_ok` tests |v| <= v_max at EVERY node
        including the measured one (see `recovery`), so the speed that makes a
        state admissible to the HAA is by definition the HAA's v_max. Reading
        it off the planner rather than writing it down here is what keeps the
        two from disagreeing when config.yaml moves.
        """
        n = self.n_r if horizon is None else int(horizon)
        lim = self.haa.dyn.lim
        v_t, w_t = float(lim.v_max), float(lim.omega_max)
        dyn = self.dyn_full
        xi_k = np.asarray(xi, dtype=np.float64).reshape(1, NXI).copy()
        prev = (np.zeros(2) if a_prev is None
                else np.asarray(a_prev, dtype=np.float64).reshape(-1)[:2].copy())
        U = np.zeros((n, NNU), dtype=np.float64)
        for k in range(n):
            v = xi_k[0, S_VEL]
            sp = float(np.hypot(v[0], v[1]))
            om = float(xi_k[0, IOM])
            want = np.zeros(NNU, dtype=np.float64)
            if sp > v_t:
                want[U_ACC] = (v * (v_t / sp) - v) / self.dt
            if abs(om) > w_t:
                want[IAL] = (np.sign(om) * w_t - om) / self.dt
            u = dyn.clip_inputs(want[None, :], a_prev=prev)[0]
            U[k] = u
            prev = u[U_ACC]
            xi_k = dyn.step(xi_k, u[None, :])
        return U

    def _recovery_k(self, U, X, a_prev, k0=0):
        """The search itself, run at the TIGHTENED radius.

        `recovery_margin` is added to the validator for the duration and taken
        off again in `finally`, so a raised exception cannot leave every other
        consumer -- the flying HAA, the cost, `in_m` -- looking at a fatter
        vehicle than the one that exists. The validator is shared by all three
        (see `build_supervisor`), which is exactly why this has to be a
        borrow-and-return rather than a second validator: a second one would
        drift out of sync with the map on the next `set_occupancy`.

        The tightening reaches BOTH halves of the test, which is what the
        reference implementation does: the swept path check below, and the
        `in_s_haa` probe solve, whose own cheap gate reads `self.r_eff` off the
        same validator. TurtleBot's equivalent passes the inflated radius
        straight into `HAA_mppi_planner.plan` (planner_switch:1727).

        `d_influence` does NOT move with it -- `PlanarMPPI` fixes that at
        construction -- so the obstacle COST is untouched and only the hard
        gate tightens. That is the intended split: this is a feasibility
        question, not a preference.
        """
        r0 = self.validator.r_safe
        self.validator.r_safe = r0 + self.recovery_margin
        try:
            return self._recovery_search(U, X, a_prev, k0)
        finally:
            self.validator.r_safe = r0

    def _recovery_search(self, U, X, a_prev, k0=0):
        for k in range(int(k0), self.n_r + 1):
            if k > 0:
                # Prefix checks. Both are monotone in k -- a longer prefix
                # contains this one -- so failing here ends the search rather
                # than skipping to k+1.
                if not bool(self.dyn_full.inputs_ok(U[:k], a_prev=a_prev)):
                    return None
                if not self.validator.validate_states(X[:k + 1]):
                    return None
            # a_prev for the probe is the acceleration flown INTO node k, so
            # the HAA's own slew constraint is anchored to reality.
            ap = a_prev if k == 0 else U[k - 1, U_ACC]
            ok, res = self.in_s_haa(X[k], a_prev=ap)
            if ok:
                return k, U[:k], X[:k + 1], res
        return None

    def _haa_plan(self, xi, a_prev):
        """`self.haa.plan` at the goal, never with an empty nominal.

        The one place the flying HAA is solved from inside `step`, so the seed
        described by SEED_NOMINAL_DEFAULT has exactly one home. With the flag
        off this is `self.haa.plan` and nothing else.

        `np.any` rather than a flag on the switch: the nominal is emptied by
        `reset` on the way in AND by `_fallback_previous` every time the
        sampler comes up dry, and it is the second one that turns a bad tick
        into a bad second. The test asks the question that matters -- is the
        sampler about to be given nothing to sample around -- rather than
        guessing which code path emptied it.
        """
        if self.seed_nominal:
            U = getattr(self.haa, "U_nom", None)
            if U is None or not np.any(U):
                self.haa.U_nom = self.haa.dyn.brake_to_hover(
                    xi, a_prev=a_prev, horizon=self.haa.N)
                self.n_seed += 1
        return self.haa.plan(xi, self.goal, a_prev=a_prev)

    def in_r(self, xi, a_prev=None):
        return self.recovery(xi, a_prev=a_prev) is not None

    def _committed_ok(self, hpa_ref, a_prev=None):
        """Is the whole plan the HPA is committing to still recoverable? -> bool

        The extra conjunct N_LOOK_DEFAULT describes, and it is deliberately a
        CONJUNCT: `step` has already run the one-tick test in full and this
        only adds to it. No reference (`step` called with the input alone, as
        the self-test does) means no committed path to look along, and the test
        passes -- the safety condition is the one-tick one, which ran.

        Two questions, in the cheap-first order the rest of this module uses:
        is the committed path clear at the flying radius, and is the state it
        ENDS in still one a validated manoeuvre can leave. Between those two
        the vehicle is not unmonitored -- this runs again next tick, against
        the belief grid as it is then, exactly as `_advance_bridge` re-checks
        the remaining bridge rather than trusting the one it built.
        """
        if hpa_ref is None or not hpa_ref.n_nodes:
            return True
        n = min(self.n_look, hpa_ref.n_nodes)
        if not self.validator.path_safe(hpa_ref.p[:n]):
            return False
        i = n - 1
        xi_end = np.empty(NXI, dtype=np.float64)
        xi_end[S_POS] = hpa_ref.p[i]
        xi_end[S_VEL] = hpa_ref.v[i]
        xi_end[4] = float(hpa_ref.psi[i])
        xi_end[5] = float(hpa_ref.psi_dot[i])
        # The acceleration flown INTO that node anchors the slew chain of
        # whatever has to recover from it -- the same argument `_recovery_search`
        # makes for handing the probe `U[k-1]`.
        return self.recovery(xi_end, a_prev=hpa_ref.a[i]) is not None

    # ----------------------------------------------------------------- M_Nm

    @property
    def d_return(self):
        """Clearance [m] the HPA may be handed the vehicle at. Paper eq. (62).

            d_return = v_max (N_m dt)  +  v_max^2 / (2 a_brake)  +  d_margin

        Read left to right: how far an unverified policy can travel while the
        monitor is not looking, plus how far the certified brake then needs to
        stop it, plus the margin the map is not already carrying.

          v_max     the FULL envelope's speed (`dyn_full`, the `hpa:` block).
                    Once the hand-back happens the HPA is flying, so the bound
                    on speed during the window is the HPA's ceiling, not the
                    HAA's.

                    THE TURTLEBOT USES ITS `v_limit_haa` HERE
                    (planner_switch:1871) AND THAT DOES NOT TRANSFER. Its bound
                    is sound because its `a_limit` is 0.25 m/s^2: over one
                    N_s step the vehicle can gain 0.025 m/s, so with the
                    precondition `v < v_limit_haa - N_s dt a_limit` the speed
                    genuinely stays under `v_limit_haa` for the whole window.
                    Our a_max_eff is 8.23 m/s^2. Even jerk-limited that is
                    ~1.2 m/s of authority over 0.3 s, so nothing but the HPA's
                    own 0.5 m/s ceiling bounds the speed here. A revision that
                    copied `v_limit_haa` across was measured and reverted.
          stop      NOT `v^2/(2 a_max_eff)`. That closed form assumes the
                    deceleration is available instantly, and it is not: the
                    jerk bound needs `a_max_eff / j_max` = 0.31 s -- 3.1 plan
                    ticks -- merely to reach full braking, and the vehicle
                    keeps travelling through the ramp. MEASURED against the
                    manoeuvre `recovery` actually flies:

                        v0     v^2/(2a)   brake_to_hover   ratio
                        0.31    0.0058       0.0202        3.5x
                        0.35    0.0074       0.0262        3.5x
                        0.50    0.0152       0.0487        3.2x

                    The formula was under-counting the stopping distance by a
                    factor of three inside a safety threshold. So this term
                    ROLLS THE ACTUAL BRAKE and measures it (`_brake_distance`),
                    which is the same principle as reusing `a_max_eff` taken
                    one step further: reuse the MANOEUVRE, not just its bound,
                    and the threshold cannot disagree with what gets flown.
          d_margin  `r_eff`. NOT a duplicate of the vehicle radius: this map's
                    `clearance()` is a pure distance to the nearest occupied
                    cell (planar_sim/perception/occupancy.py) and is not
                    pre-inflated, so the body radius, the perception margin and
                    the tracking tube all still have to be added here. If the
                    grid is ever changed to publish an inflated distance, this
                    term is the one that goes.

        WHY THIS REPLACED `rho(speed) + r_eff`. `rho` is the double
        integrator's reachable disc under arbitrary acceleration and IGNORES
        v_max, which this module's own header already flags as the reason M_Nm
        is tight: at n_m = 3 it charges 0.5*a*T^2 = 0.37 m of displacement that
        no admissible input can produce, because v_max/a_max = 0.06 s is a
        fifth of T. Eq. (62) charges travel at the speed limit instead, and
        then pays for the stop explicitly. It is smaller and it is the bound
        that is actually true.

        IT IS ALSO THE HYSTERESIS. The rejection side of the loop is
        `recovery()` -- can this state still be braked into S_HAA -- which
        bites at roughly the braking distance alone. Re-entry additionally
        pays for a whole unmonitored horizon of travel, so the gap between
        losing the HPA and being allowed it back is `v_max * N_m * dt` wide by
        construction. That is why no dwell timer is needed, and why adding one
        would only hide which of the two conditions actually fired.
        """
        v = float(self.dyn_full.lim.v_max)
        return v * (self.n_m * self.dt) + self._brake_distance(v) + self.r_eff

    def _brake_distance(self, v):
        """How far `brake_to_hover` actually travels stopping from `v` [m].

        Cached on the speed: it depends only on that and on `dyn_full`'s
        limits, neither of which moves during a run, and `d_return` is read
        every HAA tick.

        `a_prev = 0` is the reference case. A vehicle braking from the
        acceleration rim travels further, but that state is R_Nr's business,
        not this threshold's -- `recovery` tests it directly and with the
        actual `a_prev` in hand.
        """
        key = round(float(v), 6)
        hit = self._brake_cache.get(key)
        if hit is not None:
            return hit
        xi = np.zeros(NXI)
        xi[3] = float(v)                       # heading-aligned, +y
        U = self.dyn_full.brake_to_hover(xi, a_prev=np.zeros(2), horizon=64)
        X = self.dyn_full.rollout(xi, U)[0]
        sp = np.linalg.norm(X[:, S_VEL], axis=1)
        stopped = np.nonzero(sp < 1e-3)[0]
        k = int(stopped[0]) if stopped.size else X.shape[0] - 1
        d = float(np.linalg.norm(X[k, S_POS] - X[0, S_POS]))
        self._brake_cache[key] = d
        return d

    def in_m(self, xi, a_prev=None, check_s_haa=True, hpa_ref=None):
        """Is `xi` far enough inside S_HAA to hand the HPA the vehicle?

        Two conditions. The geometric one is `clearance > d_return` (see that
        property): far enough that Nm ticks of unverified action followed by a
        full-authority stop all fit inside the free space. That is what "the
        HPA may act unverified for the next Nm ticks" requires. The second is
        membership of S_HAA itself -- M_Nm is a subset of it by definition, and
        clearance says nothing about whether the HAA can fly the state.

        The cheap test runs first: the geometric one costs a table lookup, the
        S_HAA one costs an MPPI solve, and in HAA mode the geometric test is
        the one that usually fails.
        """
        xi = np.asarray(xi, dtype=np.float64).reshape(NXI)
        if float(self.validator.clearance(xi[S_POS])) <= self.d_return:
            return False
        if not self._worth_handing_back(hpa_ref):
            return False
        if not check_s_haa:
            return True
        return self.in_s_haa(xi, a_prev=a_prev)[0]

    def _worth_handing_back(self, hpa_ref):
        """Will the hand-back outlast the bridge that pays for it? -> bool.

        NOT A SAFETY TEST, and the distinction matters. `d_return` above IS
        M_Nm's condition and it is correct: it already prices `N_m dt` of
        travel plus the stop, so every hand-back it allows really is safe for
        the window it promises. What it does not ask is whether the hand-back
        is WORTH MAKING -- and measured on `mixed/51004`, clearance crosses
        `d_return` twenty times in 31 s, so the monitor handed the vehicle over
        on an upward crossing and took it back 0.7 s later, seven times. A
        transition costs `transition_steps` (0.6 s) of bridge; a 0.7 s tenure
        is the bridge and nothing else.

        So: require the threshold to hold along the HPA's own proposed
        trajectory for `transition_steps + n_m` nodes -- long enough to cover
        the bridge and then the window the hand-back is promising. No new
        parameter: it is the sum of the two that already decide those two
        things.

        WITHOUT A REFERENCE THIS PASSES. `step(xi, u_hpa)` is an input-level
        interface and has no trajectory to look along; callers that use it (the
        self-test) get the point test alone, which is the safety condition
        intact. Only `plan()`, which has just solved the HPA, can supply the
        path -- and it is the caller that cares about thrash.
        """
        if hpa_ref is None:
            return True
        n = min(self.transition_steps + self.n_m, hpa_ref.n_nodes)
        if n <= 0:
            return True
        c = self.validator.clearance(hpa_ref.p[:n])
        return bool(np.all(np.asarray(c) > self.d_return))

    # -------------------------------------------------------------- reporting

    def classify(self, xi, a_prev=None):
        """All three memberships at one state. For logging and the self-test.

        Costs up to n_r + 2 HAA solves, so this is a diagnostic, not the
        decision path -- `step` evaluates only what it needs.
        """
        import time as _time
        xi = np.asarray(xi, dtype=np.float64).reshape(NXI)
        n0, t0 = self.n_probe, self.probe_s
        wall0 = _time.time()

        s_ok, res = self.in_s_haa(xi, a_prev=a_prev)
        rec = self.recovery(xi, a_prev=a_prev)
        speed = float(np.hypot(xi[2], xi[3]))
        clear = float(self.validator.clearance(xi[S_POS]))
        rho = self.rho(speed)          # reported for continuity of the table
        m_ok = s_ok and clear > self.d_return

        return Verdict(s_haa=s_ok, r_nr=rec is not None, m_nm=m_ok,
                       clearance=clear, rho=rho,
                       k_recover=(-1 if rec is None else rec[0]),
                       haa_status=("" if res is None else res.status),
                       n_probe=self.n_probe - n0,
                       probe_ms=1000.0 * (_time.time() - wall0))

    # --------------------------------------------------------------- the tick

    def step(self, xi, u_hpa, a_prev=None, hpa_ref=None):
        """One tick of the decision logic. Returns a Decision.

        `u_hpa` is the input the HPA wants to apply, (3,), or None when the
        HPA has nothing to offer -- which is treated exactly like an input
        that fails the recoverability test.
        """
        xi = np.asarray(xi, dtype=np.float64).reshape(NXI)
        self.n_ticks[self.mode] += 1

        if self.mode == MODE_HAA:
            # HAND BACK ONLY FROM WELL INSIDE THE ENVELOPE, never from its
            # boundary: at the boundary the first unverified action leaves it,
            # and the monitor would oscillate at the plan rate.
            if self.in_m(xi, a_prev=a_prev, hpa_ref=hpa_ref):
                self.mode = MODE_HPA
                self.n_switch[MODE_HPA] += 1
                # Fall through into the HPA branch for THIS tick rather than
                # spending it in HAA mode. The R_Nr gate below still has to
                # pass, so handing back never skips a check -- it only avoids
                # wasting the tick that proved the state was safe.
                d = self.step(xi, u_hpa, a_prev=a_prev, hpa_ref=hpa_ref)
                self.n_ticks[MODE_HAA] -= 1     # counted by the recursive call
                if d.source == SRC_HPA:
                    d.switched = True
                    return d
                # THE HAND-BACK DID NOT SURVIVE ITS OWN R_Nr GATE: M_Nm said
                # the state was safe to hand over and the very same tick's
                # u_hpa was refused, so the HAA flew the tick and the vehicle
                # changed authority ZERO times. Counting it as a handover is
                # how a 12-tick run flown end to end by the HAA reported
                # n_ticks HPA 12 / HAA 0 and 6 switches each way -- measured
                # with a stub producer whose input fails the slew bound, which
                # re-enters this branch every tick because M_Nm stays true.
                # "Fraction of ticks on the HPA" is the first number an
                # assurance argument quotes, so it counts the AUTHORITY THAT
                # FLEW, never the mode the tick was entered in.
                self.n_switch[MODE_HPA] -= 1
                self.n_switch[MODE_HAA] -= 1
                d.switched = False
                return d
            res = self._haa_plan(xi, a_prev)
            return Decision(MODE_HAA, SRC_HAA, U=res.U, X=res.X, result=res,
                            fault=not res.ok,
                            reason="held by the HAA: %s" % (res.status,))

        # ---- HPA mode ----------------------------------------------------
        ok_u = (u_hpa is not None
                and bool(self.dyn_full.inputs_ok(
                    np.asarray(u_hpa, dtype=np.float64).reshape(1, NNU),
                    a_prev=a_prev)))
        rec_next = None
        if ok_u:
            u = np.asarray(u_hpa, dtype=np.float64).reshape(NNU)
            xi_next = self.dyn_full.step(xi, u)
            # The tick the HPA is asking for has to be survivable itself, not
            # just end somewhere recoverable: at 0.5 m/s one tick covers 0.05 m
            # and the swept segment is what a nodal check would step over.
            if self.validator.validate_states(np.stack([xi, xi_next])):
                rec_next = self.recovery(xi_next, a_prev=u[U_ACC])

        look_ok = True
        if rec_next is not None and self.n_look > 1:
            look_ok = self._committed_ok(hpa_ref, a_prev=a_prev)
            if not look_ok:
                self.n_look_refused += 1

        if rec_next is not None and look_ok:
            return Decision(MODE_HPA, SRC_HPA, U=u[None, :],
                            X=np.stack([xi, xi_next]),
                            reason="u_hpa lands in R_%d (%d braking steps to "
                                   "S_HAA)" % (self.n_r, rec_next[0]))

        # ---- the HPA's tick is not recoverable: take the vehicle back -----
        self.mode = MODE_HAA
        self.n_switch[MODE_HAA] += 1
        # The tick was credited to the HPA on the way in and the HAA is the one
        # flying it. Moving it here also makes the hand-back path above come
        # out right without a second correction: that path adds one HAA tick on
        # entry, the recursive call adds one more here, and its single
        # decrement leaves exactly one tick against the HAA.
        self.n_ticks[MODE_HPA] -= 1
        self.n_ticks[MODE_HAA] += 1
        # DROP THE HAA'S WARM START ON THE WAY BACK IN -- unless `warm_haa` is
        # keeping it current, in which case `_last_ref` is one tick old rather
        # than a handover old and there is nothing stale to drop.
        # It has not planned
        # since the handover, so `_last_ref` is a trajectory from wherever the
        # vehicle was that many ticks ago, and `_fallback_previous` would
        # happily return it: it re-checks the stale path against today's map
        # but never against today's STATE, so a plan the vehicle has long since
        # left comes back as PREVIOUS, i.e. as a rescue that will not happen.
        # Seen in the self-test before this line existed -- a state inside an
        # obstacle was answered with PREVIOUS. Costs one warm start, which the
        # HAA rebuilds on the tick it is re-entered on.
        #
        # WITH `warm_haa` THE TWO HALVES OF `reset` COME APART, and only one of
        # them is wanted. `_last_ref` must still go: the shadow solve keeps it
        # one tick old ONLY while it succeeds, and a run of FAILED or PREVIOUS
        # answers walks it forward from a state the vehicle has left -- the
        # same stale rescue this reset was added for, reintroduced through the
        # back door. `U_nom` must stay: it is the warm start, it is what the
        # shadow solve exists to keep current, and zeroing it is what makes a
        # takeover cost six seconds.
        if self.warm_haa:
            self.haa._last_ref = None
        else:
            self.haa.reset()
        why = ("the HPA had no input" if u_hpa is None else
               "u_hpa is not an admissible input" if not ok_u else
               "the state after u_hpa is outside R_%d" % self.n_r if
               rec_next is None else
               "the committed plan leaves R_%d within %d nodes"
               % (self.n_r, self.n_look))

        rec = self.recovery(xi, a_prev=a_prev)
        if rec is not None and rec[0] == 0:
            # Already in S_HAA: there is nothing to recover from, so the HAA
            # plans this tick itself instead of flying a zero-length manoeuvre.
            res = self._haa_plan(xi, a_prev)
            return Decision(MODE_HAA, SRC_HAA, U=res.U, X=res.X, result=res,
                            fault=not res.ok, switched=True,
                            reason="%s; already in S_HAA, HAA takes over"
                                   % why)
        if rec is not None:
            k, U, X, _ = rec
            return Decision(MODE_HAA, SRC_RECOVERY, U=U, X=X, switched=True,
                            reason="%s; %d-step recovery into S_HAA" % (why, k))

        # ---- nothing is recoverable ---------------------------------------
        # The invariant has already been broken -- normally because the map
        # changed underneath it, which is the one way a state can leave R_Nr
        # without the monitor having agreed to it. Say so; do not report a
        # clean handover.
        self.n_fault += 1
        res = self._haa_plan(xi, a_prev)
        if res.ok:
            return Decision(MODE_HAA, SRC_HAA, U=res.U, X=res.X, result=res,
                            fault=True, switched=True,
                            reason="%s AND the current state is outside R_%d; "
                                   "the HAA still solved (%s)"
                                   % (why, self.n_r, res.status))
        U = self.dyn_full.brake_to_hover(xi, a_prev=a_prev, horizon=self.n_r)
        X = self.dyn_full.rollout(xi, U)[0]
        return Decision(MODE_HAA, SRC_RECOVERY, U=U, X=X, fault=True,
                        switched=True,
                        reason="%s AND no validated action exists; braking at "
                               "full authority, UNVALIDATED" % why)

    # ------------------------------------------------------- producer wrapper

    def plan(self, state, goal=None, a_prev=None, **kw):
        """`plan(state, goal, a_prev) -> MPPIResult`, the producer interface.

        A thin wrapper around `_decide`, which is the tick. All this adds is
        the SHADOW SOLVE: with `warm_haa` on, the HAA is planned on every tick
        it did not fly, from the state the vehicle is actually in, and the
        answer is discarded. Its only effect is on `U_nom` and `_last_ref` --
        the warm start it will be re-entered with. See WARM_HAA_DEFAULT.

        It runs AFTER the decision, never before: the tick's authority must not
        depend on whether a discarded solve happened to succeed.
        """
        res = self._decide(state, goal=goal, a_prev=a_prev, **kw)
        d = self.last_decision
        if self.warm_haa and (d is None or d.source != SRC_HAA):
            xi = (state.to_array() if isinstance(state, PlanarState)
                  else np.asarray(state, dtype=np.float64).reshape(NXI))
            self.haa.plan(xi, self.goal, a_prev=a_prev)
            self.n_shadow += 1
        return res

    def _decide(self, state, goal=None, a_prev=None, **kw):
        """The tick itself. See `plan`.

        Lets a supervised HPA drop into `Harness` next to bare MPPI and the
        teleop pilot, so the three are measured by the same `metrics.py`.
        Needs an `hpa` producer; `step` is the interface when the caller
        already has the policy's input in hand.
        """
        if self.hpa is None:
            raise RuntimeError("no HPA producer given; call step(xi, u_hpa) "
                               "with the policy's input instead")
        if goal is not None:
            self.goal = np.asarray(goal, dtype=np.float64).reshape(-1)[:2]
        xi = (state.to_array() if isinstance(state, PlanarState)
              else np.asarray(state, dtype=np.float64).reshape(NXI))

        # A BRIDGE IN PROGRESS OWNS THE REFERENCE, BUT NOT THE MONITORING.
        # Neither PRODUCER is consulted -- re-planning from a state halfway
        # along the bridge is the discontinuity the bridge exists to remove,
        # and in the HPA -> HAA direction it would mean asking the policy whose
        # proposal was just rejected what it would like to do next. R_Nr is
        # still tested every tick inside `_advance_bridge`, which is what keeps
        # the invariant alive across the transition.
        if self._bridge is not None:
            out = self._advance_bridge(xi, a_prev=a_prev)
            if out is not None:
                return out
            # The remaining bridge stopped being safe -- the map moved under
            # it. Fall through to the certified fallback below.

        res_hpa = self.hpa.plan(xi, self.goal, a_prev=a_prev, **kw)
        u = (res_hpa.U[0] if (res_hpa.ok and res_hpa.U is not None
                              and len(res_hpa.U)) else None)
        was = self.mode
        d = self.step(xi, u, a_prev=a_prev, hpa_ref=res_hpa.reference)
        self.last_decision = d

        # `step` has already decided WHO flies. All that is added here is HOW
        # the reference gets there: on a change of authority, bridge into the
        # destination instead of handing the controller a step.
        #
        # transition_steps = 0 IS THE OTHER POLICY, NOT A DISABLED FEATURE.
        # With no bridge the switch is exactly what `step` already returned:
        # leaving the HPA that is the certified braking manoeuvre (SRC_RECOVERY,
        # at most n_r ticks, often k = 0 and the HAA simply plans), and entering
        # it the policy's own reference handed over as a step. Both are
        # admissible -- the R_Nr gate ran either way -- so the only thing given
        # up is C2 continuity of the reference the controller tracks. See
        # TRANSITION_STEPS_DEFAULT for what that costs.
        if d.switched and was != d.mode and self.transition_steps > 0:
            out = self._begin_bridge(xi, d, res_hpa, a_prev=a_prev)
            if out is not None:
                return out
            # No admissible bridge. HPA -> HAA must not stay on the rejected
            # proposal, so it takes the fallback below; HAA -> HPA simply does
            # not switch, and `step` is asked again next tick.
            if d.mode == MODE_HPA:
                # `step` counted a hand-back that is not going to happen.
                self.mode = MODE_HAA
                self.n_switch[MODE_HPA] -= 1
                d = self.step(xi, None, a_prev=a_prev)
                self.last_decision = d

        return self._as_result(d, res_hpa)

    # --------------------------------------------------------- the transition

    def _as_result(self, d, res_hpa):
        """One Decision -> the MPPIResult the harness consumes."""
        if d.source == SRC_HPA:
            self._last_ref = res_hpa.reference
            return res_hpa
        if d.source == SRC_HAA:
            if d.result is not None:
                self._last_ref = d.result.reference
            return d.result
        ref = PlanarReferenceSequence.from_rollout(d.X, d.U, self.dt)
        self._last_ref = ref
        return MPPIResult(PlannerStatus.BRAKING, ref, d.U, d.X, float("inf"),
                          0, 0, 0.0, d.reason)

    def _src_node(self, xi):
        """(p, v, a, psi, psi_dot) to leave from: the COMMAND, else the state.

        The commanded reference is preferred because the tracking error between
        it and `xi` is a disturbance the controller is already rejecting;
        bridging from the estimate would fold that error into the reference and
        ask it to reject the same error twice. `_last_ref` is None only on the
        first tick, where there is no command yet.
        """
        r = self._last_ref
        if r is not None and r.n_nodes:
            return (r.p[0], r.v[0], r.a[0], float(r.psi[0]),
                    float(r.psi_dot[0]))
        return (xi[S_POS], xi[S_VEL], np.zeros(2), float(xi[4]),
                float(xi[5]))

    def _destination(self, d, res_hpa):
        """The reference the switch is heading for, or None."""
        if d.mode == MODE_HPA:
            return res_hpa.reference if res_hpa is not None else None
        if d.result is not None and d.result.reference is not None:
            return d.result.reference
        if d.X is not None and d.U is not None and len(d.U):
            return PlanarReferenceSequence.from_rollout(d.X, d.U, self.dt)
        return None

    def _begin_bridge(self, xi, d, res_hpa, a_prev=None):
        """Try to open a bridge into `d.mode`. -> MPPIResult, or None.

        None means "no admissible bridge exists this tick"; the caller decides
        what that costs, and the two directions do not pay the same price.
        """
        dst = self._destination(d, res_hpa)
        k = self.transition_steps
        ref = (None if dst is None
               else bridge_reference(self._src_node(xi), dst, k))
        if ref is None or not bridge_ok(ref, k, self.validator,
                                        self.dyn_full, a_prev=a_prev):
            self.n_bridge_refused += 1
            return None

        # ONE MODE IS REPORTED FOR BOTH DIRECTIONS WHILE THE BRIDGE FLIES, and
        # it is the conservative one. Leaving the HPA, the vehicle is already
        # off it; entering the HPA, it has not been given away yet. Either way
        # the authority in force is not the policy's, and `step` has already
        # counted the switch -- this only delays when `mode` starts saying so
        # until the handover has actually happened.
        self.mode = MODE_HAA
        # One node is flown now, so the buffer holds the remaining k-1.
        self._bridge = (ref, k - 1, d.mode)
        self._last_ref = ref
        self.n_bridge += 1
        return self._bridge_result(ref, d.mode, k,
                                   "bridging to %s over %d steps"
                                   % (d.mode, k))

    def _advance_bridge(self, xi, a_prev=None):
        """Emit the next node of the bridge. -> MPPIResult, or None if unsafe.

        TWO CHECKS, AND THE SECOND ONE IS THE INVARIANT.

        1. The REMAINING bridge is re-validated every tick rather than trusted
           from when it was built: the occupancy grid updates at its own rate
           (10 Hz, `planar_sim/perception/pipeline.py`), so a bridge can be
           overtaken by a pillar that was unknown when it was planned.

        2. R_Nr is re-tested at the state the vehicle is actually in. Geometry
           alone is not enough. A bridge that clears every obstacle can still
           carry the vehicle to a state from which nothing recovers, and while
           the bridge flies neither producer is being consulted -- so without
           this test the monitor would go blind for `transition_steps` ticks
           and the invariant it exists to maintain ("there is always a
           validated manoeuvre back into S_HAA") would simply lapse. MEASURED
           BEFORE THIS CHECK EXISTED: 12 of 196 ticks on gap/61015, 1.2 s of a
           19.6 s flight, ran with no monitoring at all.

        Either failure returns None, which drops the caller onto the same
        certified braking fallback a rejected HPA proposal takes.
        """
        ref, left, target = self._bridge
        ref = drop_first(ref)
        left = int(left)
        if left > 0 and not bridge_ok(ref, left, self.validator,
                                      self.dyn_full, a_prev=a_prev):
            self._bridge = None
            self.n_bridge_abort += 1
            return None
        if self.recovery(xi, a_prev=a_prev) is None:
            self._bridge = None
            self.n_bridge_abort += 1
            return None

        if left <= 0:
            # Exhausted: the vehicle is now ON the destination trajectory, in
            # every derivative. Activate the target mode and let it plan
            # normally from the next tick.
            self._bridge = None
            self.mode = target
        else:
            self._bridge = (ref, left - 1, target)

        self._last_ref = ref
        return self._bridge_result(ref, target, left,
                                   "bridge to %s, %d steps left"
                                   % (target, left))

    def _bridge_result(self, ref, target, left, reason):
        """The bridge's MPPIResult, and the Decision that reports it.

        `last_decision` MUST be refreshed here. `experiments/haa_vs_hpa.py`
        reads it every tick to log which authority flew, and a bridge that
        returned early without touching it left the log repeating whatever the
        last real decision said -- so a run would report 100% HPA while the
        vehicle was on a bridge it had been taken off the HPA to fly.

        SRC_RECOVERY, not SRC_HPA or SRC_HAA: the tick is flown by neither
        producer. It is a validated manoeuvre the monitor generated, which is
        exactly what that source already means for the braking path.
        """
        self.last_decision = Decision(
            MODE_HAA, SRC_RECOVERY, U=None, X=None, reason=reason)
        return MPPIResult(PlannerStatus.BRAKING, ref, None, None,
                          float("inf"), 0, 0, 0.0, reason)

    # -------------------------------------------------------------- describe

    def describe(self):
        T = self.n_m * self.dt
        lim = self.dyn_full.lim
        return "\n".join([
            "DeSimplex  n_r=%d  n_m=%d  dt=%.2f s  (%.1f s recovery, %.1f s "
            "margin)" % (self.n_r, self.n_m, self.dt, self.n_r * self.dt, T),
            "  full envelope   v<=%.2f m/s  a<=%.2f m/s^2  jerk<=%.1f m/s^3"
            % (lim.v_max, lim.a_max_eff, lim.j_max),
            "  HAA envelope    v<=%.2f m/s  a<=%.2f m/s^2  (S_HAA is whatever "
            "this planner accepts)"
            % (self.haa.dyn.lim.v_max, self.haa.dyn.lim.a_max_eff),
            "  M_%d needs      clearance > d_return = %.2f m  "
            "(= HAA v_max %.2f x %.1f s travel + %.3f m stop + r_eff %.2f)"
            % (self.n_m, self.d_return, self.haa.dyn.lim.v_max,
               self.n_m * self.dt,
               self.haa.dyn.lim.v_max ** 2 / (2.0 * lim.a_max_eff),
               self.r_eff),
            "                  the superseded rho bound asked %.2f m at rest, "
            "%.2f m at v_max (see d_return)"
            % (self.rho(0.0) + self.r_eff, self.rho(lim.v_max) + self.r_eff),
            "  transition      %d steps = %.2f s quintic bridge on a switch"
            % (self.transition_steps, self.transition_steps * self.dt),
            "  jerk turnaround %.2f s = %.1f ticks to zero the acceleration "
            "from the rim" % (lim.a_max_eff / lim.j_max,
                              lim.a_max_eff / lim.j_max / self.dt),
        ])


# ===========================================================================
#  SELF-TEST
# ===========================================================================

def _pick_by_clearance(cfg, occ, target):
    """Interior point whose clearance is closest to `target` [m].

    Found from the field rather than typed in, so "just outside the inflated
    set" stays that when the map or r_eff changes.
    """
    x0, y0, x1, y1 = cfg.interior
    xs = np.arange(x0 + 0.05, x1, 0.05)
    ys = np.arange(y0 + 0.05, y1, 0.05)
    gx, gy = np.meshgrid(xs, ys)
    c = occ.clearance(gx, gy)
    i = int(np.argmin(np.abs(c - target)))
    return float(gx.flat[i]), float(gy.flat[i]), float(c.flat[i])


def _toward_obstacle(occ, x, y, h=0.05):
    """Unit vector down the clearance gradient: straight at the nearest wall."""
    gx = occ.clearance(x + h, y) - occ.clearance(x - h, y)
    gy = occ.clearance(x, y + h) - occ.clearance(x, y - h)
    n = float(np.hypot(gx, gy))
    if n < 1e-9:
        return np.array([0.0, 1.0])
    return -np.array([float(gx), float(gy)]) / n


def _m_survey(cfg, sup, seeds):
    """How much of the flyable area is in M_Nm, on a few procedural maps.

    Geometric part only -- clearance >= rho + r_eff over the free set. The
    S_HAA half of the membership cannot be surveyed over 14000 cells at 20 ms
    a solve, and it is not the part in question: the question is whether the
    reachable disc leaves any room at all in a 5 x 7 m arena.
    """
    from planar_sim import arena
    print("\n  M_%d COVERAGE: fraction of the flyable area (clearance >= "
          "r_eff = %.2f m)" % (sup.n_m, sup.r_eff))
    print("  from which the HPA could be handed the vehicle.\n")
    need0 = sup.rho(0.0) + sup.r_eff
    needv = sup.rho(sup.dyn_full.lim.v_max) + sup.r_eff
    needc = sup.rho(sup.dyn_full.lim.v_max, speed_capped=True) + sup.r_eff
    print("    %-22s %7s   %7s %7s %7s" % ("map", "flyable", "at rest",
                                           "at v_max", "capped"))
    print("    %-22s %7s   %7.2fm %7.2fm %7.2fm"
          % ("", "", need0, needv, needc))
    x0, y0, x1, y1 = cfg.interior
    xs = np.arange(x0 + 0.025, x1, 0.05)
    ys = np.arange(y0 + 0.025, y1, 0.05)
    gx, gy = np.meshgrid(xs, ys)
    tot = float(gx.size)
    rows = []
    for family, seed in seeds:
        segs = arena.collect_world(cfg, family, seed)
        occ = arena.occupancy_from_segments(cfg, segs)
        c = occ.clearance(gx, gy)
        free = c >= sup.r_eff
        nf = float(free.sum())
        f = lambda need: (100.0 * float((c >= need).sum()) / max(nf, 1.0))
        rows.append((f(need0), f(needv), f(needc)))
        print("    %-22s %6.1f%%   %6.1f%% %6.1f%% %6.1f%%"
              % ("%s/%d" % (family, seed), 100.0 * nf / tot,
                 rows[-1][0], rows[-1][1], rows[-1][2]))
    a = np.array(rows)
    print("    %-22s %7s   %6.1f%% %6.1f%% %6.1f%%"
          % ("MEAN", "", a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()))
    # GEOMETRIC coverage, not M_Nm: the S_HAA term is omitted here, and with it
    # M_Nm at v_max is empty on every map (states_ok bounds node 0 at the HAA's
    # own v_max, so a vehicle at 0.5 m/s is outside S_HAA whatever its
    # clearance). The "at v_max" column is an upper bound on nothing reachable;
    # the "at rest" one is what hand-back actually gets, and is itself an upper
    # bound. Printed anyway because it says WHERE IN THE ROOM hand-back is
    # possible at all, which is the question the survey exists to answer.
    print("\n    M_%d geometric coverage, S_HAA term omitted: %.0f%% of the "
          "flyable area at v_max," % (sup.n_m, a[:, 1].mean()))
    print("    %.0f%% at rest. With S_HAA applied, M_%d at v_max is EMPTY -- "
          "hand-back" % (a[:, 0].mean(), sup.n_m))
    print("    happens only after the HAA has slowed the vehicle into its own "
          "envelope.")
    print("    It collapses in the corridor family (%.0f%%), where the walls "
          "are closer" % a[3, 1])
    print("    together than one reachable disc -- a corridor is HAA territory "
          "under this rule.")
    print("    Two thirds of rho is the closed form ignoring v_max, not the "
          "a_max choice: the")
    print("    speed-capped bound asks %.2f m instead of %.2f m and reaches "
          "%.0f%%." % (needc, needv, a[:, 2].mean()))
    return a


def _main():
    import os
    import sys
    import time

    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(HERE)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    # THE SELF-TEST IS THE ONE PLACE THIS FILE MAY LOOK AT THE SIMULATOR, and
    # it does so from inside `_main` rather than at module scope so the import
    # graph the rule is about (`planner` must not import `planar_sim`) stays
    # true for every real consumer. A monitor whose self-test needs a map has
    # to get the map from somewhere.
    #
    # GUARDED, because `planner/` is meant to be copied to the vehicle on its
    # own. Deployed that way this file imports and runs; only THIS function
    # cannot, and a bare ModuleNotFoundError would read as a broken package
    # rather than as the one documented dependency.
    try:
        from planar_sim import arena
        from planar_sim.build import build_supervisor
        from planar_sim.config import load as load_config
    except ImportError as exc:
        sys.stderr.write(
            "planner.supervisor self-test needs the simulator (%s).\n"
            "The MODULE does not: `planner/` imports and runs standalone, and\n"
            "this is the one function that reads config.yaml and builds maps.\n"
            "Run it from a tree that has planar_sim/ beside planner/.\n" % exc)
        return 1

    cfg = load_config(quiet=True)
    lim = cfg.hpa

    # One clutter map with three pillars: enough obstacles that "near an
    # obstacle" is a real place on the map and not a corner of the arena.
    segs = arena.collect_world(cfg, "clutter", 7, n_pillars=3)
    occ = arena.occupancy_from_segments(cfg, segs)
    sg = arena.sample_start_goal(cfg, occ, 7)
    start, goal = sg if sg else (tuple(cfg.flight.start), tuple(cfg.flight.goal))

    t0 = time.time()
    sup = build_supervisor(cfg, occ, goal, seed=0)
    print(sup.describe())
    print("  built in %.1f s   map: clutter/7, %d obstacles   goal (%.2f, %.2f)"
          % (time.time() - t0,
             len(arena.loops(segs)), goal[0], goal[1]))

    psi = arena.start_yaw(cfg)
    v_haa, v_hpa = cfg.limits.v_max, lim.v_max
    # a_max_eff, not hpa.a_max. The two differ by 1.1 mm/s^2 (8.23 is written
    # out in config.yaml, g tan(40 deg) is 8.2289), and that gap is enough:
    # `clip_inputs` projects a_prev onto the disc before anchoring the slew
    # chain while `inputs_ok` does not, so an a_prev 1.1 mm/s^2 outside the
    # disc fails its own slew test by the same amount and the row would test
    # the input gate instead of the recoverability gate.
    a_rim = float(sup.dyn_full.lim.a_max_eff)

    # Open water, and a spot right on the inflated boundary.
    ox, oy, _ = _pick_by_clearance(cfg, occ, 2.0)
    nx, ny, nc = _pick_by_clearance(cfg, occ, cfg.r_eff + 0.03)
    into = _toward_obstacle(occ, nx, ny)

    cases = [
        # label, xi, a_prev
        ("hover, open room",
         [ox, oy, 0.0, 0.0, psi, 0.0], None),
        ("HAA cruise 0.35 m/s",
         [ox, oy, 0.0, v_haa, psi, 0.0], None),
        ("HPA fast 0.50 m/s",
         [ox, oy, 0.0, v_hpa, psi, 0.0], None),
        ("HPA fast, a_prev at rim",
         [ox, oy, 0.0, v_hpa, psi, 0.0], np.array([0.0, a_rim])),
        ("HPA fast + omega 2.0",
         [ox, oy, 0.0, v_hpa, psi, lim.omega_max], None),
        ("near obstacle, hover",
         [nx, ny, 0.0, 0.0, psi, 0.0], None),
        ("near obstacle, 0.5 into it",
         [nx, ny, v_hpa * into[0], v_hpa * into[1], psi, 0.0], None),
        ("inside an obstacle",
         [nx + into[0] * (nc + 0.1), ny + into[1] * (nc + 0.1),
          0.0, 0.0, psi, 0.0], None),
    ]

    print("\n  S_HAA is measured by calling the HAA; R_%d and M_%d follow from "
          "it." % (sup.n_r, sup.n_m))
    print("  `k` is the braking steps needed to re-enter S_HAA (- = never, "
          "0 = already in).\n")
    hdr = ("%-26s %6s %6s %5s %6s  %5s %5s %5s %3s  %-9s %5s %4s"
           % ("state", "x", "y", "|v|", "clear", "S_HAA", "R_%d" % sup.n_r,
              "M_%d" % sup.n_m, "k", "mode/src", "ms", "n"))
    print("  " + hdr)
    print("  " + "-" * len(hdr))

    for label, xi, a_prev in cases:
        xi = np.asarray(xi, dtype=np.float64)
        v = sup.classify(xi, a_prev=a_prev)
        # THE HPA ASKS TO COAST -- zero acceleration, projected onto the
        # admissible set given a_prev. Projecting matters for the a_prev-at-the-
        # rim case: a raw zero input is an 8.23 m/s^2 step against a 2.63
        # m/s^2 slew bound, so the monitor would reject it as an inadmissible
        # INPUT and never reach the recoverability question the row is about.
        u = sup.dyn_full.clip_inputs(np.zeros((1, NNU)), a_prev=a_prev)[0]
        # Both branches of the decision are exercised: the monitor is forced
        # into each mode in turn so one table shows what it does from either.
        sup.mode = MODE_HPA
        d_hpa = sup.step(xi, u, a_prev=a_prev)
        sup.mode = MODE_HAA
        d_haa = sup.step(xi, u, a_prev=a_prev)
        yn = lambda b: "yes" if b else " no"
        print("  %-26s %6.2f %6.2f %5.2f %6.2f  %5s %5s %5s %3s  %-9s %5.0f %4d"
              % (label, xi[0], xi[1], float(np.hypot(xi[2], xi[3])),
                 v.clearance, yn(v.s_haa), yn(v.r_nr), yn(v.m_nm),
                 ("-" if v.k_recover < 0 else str(v.k_recover)),
                 "%s/%s" % (d_hpa.mode, d_hpa.source[:4]),
                 v.probe_ms, v.n_probe))
        for tag, d in (("in HPA mode", d_hpa), ("in HAA mode", d_haa)):
            if d.fault or d.switched:
                print("  %-26s   %s: %s/%s%s -- %s"
                      % ("", tag, d.mode, d.source,
                         "  FAULT" if d.fault else "", d.reason))

    # ---- how long a recovery actually needs, per state -------------------
    print("\n  RECOVERY DEPTH: smallest n_r for which each state is in R_n_r.")
    print("  The acceleration the HAA has to inherit sets this, not the speed:")
    print("  see the N_R_DEFAULT note.\n")
    keep = sup.n_r
    for label, xi, a_prev in cases[:5]:
        xi = np.asarray(xi, dtype=np.float64)
        found = None
        for n in range(0, 15):
            sup.n_r = n
            if sup.in_r(xi, a_prev=a_prev):
                found = n
                break
        print("    %-26s n_r >= %-4s (%s)"
              % (label, "15+" if found is None else str(found),
                 "never, within 1.5 s" if found is None
                 else "%.1f s of committed braking" % (found * sup.dt)))
    sup.n_r = keep

    # ---- is M_Nm usefully non-empty? -------------------------------------
    _m_survey(cfg, sup, [("clutter", 7), ("clutter", 12), ("gap", 3),
                         ("corridor", 5), ("clutter", 21)])

    print("\n  cost: %d HAA solves, %.1f ms each on average"
          % (sup.n_probe, 1000.0 * sup.probe_s / max(sup.n_probe, 1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
