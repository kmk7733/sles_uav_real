#!/usr/bin/python
# -*- coding: utf-8 -*-
"""ROS adapter for the planar MPPI planner.

    /grid_map ---> PlanarOccupancy ---> PlanarMPPI ---> PlanarReferenceSequence
                                                            |
                                                    lift to z0 (3D)
                                                            |
                                       vicon/world -> FCU ENU (haa_frames)
                                                            |
                                              commander/set_pose (PositionTarget)
                                                            |
                                            setpoint_buffer.py -> mavros -> PX4

This file is the ONLY part of the planar stack that knows about ROS. Everything
it calls -- planar_mppi, planar_dynamics, planar_safety, planar_map,
planar_types -- runs and is tested without it (python3 test_planar.py).

Same contract as haa_planner_node.py, so setpoint_buffer.py is unchanged and
remains the last line of defence: if this node dies, the buffer keeps
republishing the last setpoint and the vehicle hovers.

    Terminal A:  ROS_NAMESPACE=rogx2 python planar_planner_node.py
    Terminal B:  ROS_NAMESPACE=rogx2 python setpoint_buffer.py

FRAMES
Mapper and planner consume the same EKF-derived pose in the configured world
frame. ekf_world_align.py owns the fixed world <- fcu_local transform and
publishes its validity/epoch on /robot/frame_alignment. This node rotates EKF
linear velocity into world and uses the exact inverse for references. It never
re-estimates that transform from the pose it just transformed. Invalid or stale
alignment/state and mismatched-epoch maps withhold all commander setpoints.

ROLL AND PITCH ARE NOT PLANNED
The planner commands a planar acceleration. PX4 turns that into attitude. That
is the whole point of the fixed-altitude planar formulation, and it is why the
reference published here has vz = az = 0: altitude is held by the low-level
controller and was never a planning variable.
"""

import json
import os
import subprocess
import sys
import threading

import numpy as np
import rospy
from scipy.ndimage import distance_transform_edt

from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, TwistStamped, Point
from mavros_msgs.msg import PositionTarget
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import String, Bool
from visualization_msgs.msg import Marker, MarkerArray

from guidance_library import Controller
from haa_frames import SharedFrameAlignment, planar_ekf_velocity, yaw_from_quat

# THE PLANNER COMES FROM `planner/`, WHICH IS THE SIMULATOR'S OWN PACKAGE.
# `catkin_ws/src/planner` is byte-identical to `planner/` in
# sles_uav_planar_sim -- verified with `diff -rq` -- so what flies here is what
# the simulator's tests pin and what its results were produced with. This node
# used to import flat copies sitting beside it (planar_mppi.py, planar_map.py,
# frontier.py, ...) that had drifted a month behind: 142 changed lines in the
# occupancy grid alone and 180 in the frontier cost. Those files are left in
# place but nothing imports them any more.
#
# Found by walking up rather than by a fixed path, so the tree can move.
def _find_src(start):
    d = os.path.dirname(os.path.abspath(start))
    for _ in range(8):
        if os.path.isfile(os.path.join(d, "planner", "__init__.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise ImportError("cannot find planner/ above %s" % start)


_SRC = _find_src(__file__)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from planner.types import PlanarState, PlannerStatus
from planner.dynamics import PlanarDynamics, PlanarLimits
from planner.grid import PlanarOccupancy, FreeSpace
from planner.safety import PlanarSafetyValidator, safe_radius
from planner.mppi import PlanarMPPI, PlanarCostWeights
from planner.haa.capped import CappedDynamics
from planner.haa.cost import FrontierMPPI


# planar_sim/config.py:65-67, and they have to stay the same three numbers.
SIGMA_FRAC_A = 0.48        # of a_max:      clipping ceiling
SIGMA_FRAC_ALPHA = 0.5     # of alpha_max:  clipping ceiling, yaw channel
SIGMA_FRAC_V = 0.54        # of v_max/(dt*sqrt(N)):  rejection ceiling


def _ceilings(lim, dt, horizon):
    spread = float(dt) * np.sqrt(float(horizon))
    return ((SIGMA_FRAC_A * lim.a_max, SIGMA_FRAC_V * lim.v_max / spread),
            (SIGMA_FRAC_ALPHA * lim.alpha_max,
             SIGMA_FRAC_V * lim.omega_max / spread))


def mppi_sigma(lim, dt, horizon):
    """Per-channel MPPI search width (ax, ay, alpha), the smaller ceiling.

    Bounded from two directions at once, which is why it is derived rather
    than configured:

      CLIPPING (the input bound). Samples are clipped to a_max, so a width
      near it piles the proposal onto the +-a_max corners and the weighted
      average degenerates into a vote between bang-bang extremes.

      REJECTION (the state bound). Integrating noise of scale sigma for N
      steps of dt gives a velocity random walk of sigma*dt*sqrt(N), and the
      validator rejects any node over v_max. Too wide and samples die before
      they are ever scored -- which does not look like a tuning problem from
      outside: the solver still returns WEIGHTED, from a handful of
      survivors, and the vehicle simply crawls.

    Only the second couples to the horizon, so it is the one a fixed sigma
    silently gets wrong the moment N, dt or v_max moves.
    """
    (a_clip, a_rej), (w_clip, w_rej) = _ceilings(lim, dt, horizon)
    a, w = min(a_clip, a_rej), min(w_clip, w_rej)
    return (a, a, w)


def sigma_binding(lim, dt, horizon):
    """Which ceiling won, for the log. Rejection is the one to worry about."""
    (a_clip, a_rej), (w_clip, w_rej) = _ceilings(lim, dt, horizon)
    return "a:%s yaw:%s" % ("clip" if a_clip < a_rej else "reject",
                            "clip" if w_clip < w_rej else "reject")


class PlanarPlannerNode(object):

    def __init__(self):
        rospy.init_node("planar_planner_node")
        self.controller = Controller()
        self.lock = threading.RLock()
        self.plan_lock = threading.Lock()

        # ------------------------------------------------------------ params
        self.z0 = rospy.get_param("~z0", self.controller.takeoff_height)
        self.plan_rate = rospy.get_param("~plan_rate", 10.0)
        self.pub_rate = rospy.get_param("~pub_rate", 20.0)
        self.dt = rospy.get_param("~dt", 0.1)
        # K and N are set from a measurement against the LIVE stack, not from an
        # idle benchmark. On this Xavier NX (MODE_15W_6CORE) with the ZED,
        # depth_to_grid, vicon_bridge and foxglove_bridge all running, loadavg
        # sits near 14 on 6 cores and the solve time roughly quadruples --
        # K=256/N=20 that takes 26 ms idle takes ~110 ms here. Sizing off the
        # idle number is how a planner that "runs at 39 Hz on the bench" misses
        # every deadline in flight.
        #
        # The binding constraint on N is NOT solve time, it is the surviving
        # sample fraction. Measured at K=128 against the live grid (70% unknown,
        # therefore unsafe, leaving ~1% of the arena flyable after r_safe):
        #
        #     N=15 dt=0.10  1.5 s   57% valid
        #     N=20 dt=0.10  2.0 s   36% valid   <- default
        #     N=25 dt=0.10  2.5 s   23% valid
        #     N=30 dt=0.10  3.0 s   14% valid
        #     N=20 dt=0.15  3.0 s    7% valid
        #     N=20 dt=0.20  4.0 s    1% valid   <- MPPI has nothing to average
        #
        # A longer rollout has more chances to touch the unsafe set, so raising
        # N or dt trades lookahead against the size of the candidate pool. Note
        # that raising dt is nearly free in CPU (the rollout loop is O(N)) and
        # costs NO integration accuracy -- the planar double integrator is exact
        # under piecewise-constant acceleration -- but it wrecks the valid
        # fraction fastest, so it is the wrong lever here.
        # 30, which is config.yaml's. It was 20, from the table above -- and
        # that table was measured at sigma 1.2, where a longer rollout walks
        # four times further into the unsafe set per node and the valid
        # fraction collapses. At the derived sigma the two are not the same
        # experiment, so the reason for 20 went away with the envelope.
        self.horizon = int(rospy.get_param("~horizon", 30))
        self.num_samples = int(rospy.get_param("~num_samples", 192))

        self.r_quad = rospy.get_param("~r_quad", 0.31)
        self.r_perc = rospy.get_param("~r_perc", 0.10)
        self.r_track = rospy.get_param("~r_track", 0.05)
        self.d_clr = rospy.get_param("~d_clr", 0.05)
        self.r_safe = safe_radius(self.r_quad, self.r_perc, self.r_track,
                                  self.d_clr)
        # MAP inflation, which is NOT r_safe: r_quad is the airframe disc and
        # the validator already applies it against the raw map, so growing the
        # map by it too would draw the vehicle radius twice. Display only.
        self.r_eff = float(rospy.get_param("~r_eff",
                                           self.r_safe - self.r_quad))

        self.pose_topic = rospy.get_param("~state_topic", "/robot/pose_world_epoch")
        self.grid_topic = rospy.get_param("~grid_topic", "/grid_map")
        self.require_map = rospy.get_param("~require_map", True)
        self.unknown_unsafe = rospy.get_param("~unknown_unsafe", True)
        # MARGIN AROUND UNOBSERVED SPACE, and 0.0 is the simulator's value.
        # `unknown_unsafe` makes an unknown cell untraversable; this decides
        # whether its NEIGHBOURS are pushed back as well. r_safe is what it
        # takes to clear a MEASURED surface -- r_perc is the error of a range
        # reading, r_track the tube around a trajectory -- and unobserved space
        # is a hole in the map, not a surface, so it earns none of them. The
        # simulator has always run 0.0 (config.yaml:mapping.unknown_inflate,
        # with the measurement beside the key: inflating unknown by even r_quad
        # pushed the frontier back faster than the camera revealed it and a run
        # that reached the goal in 10.9 s stalled 0.36 m short). This vehicle
        # ran the other rule until 2026-09-06 only because planner/grid.py
        # merged the two sets before its distance transform and could not
        # express the sim's; on flight_20260906_054936 that cost 9.6-13.9
        # percentage points of the arena.
        self.unknown_inflate = float(rospy.get_param("~unknown_inflate", 0.0))
        self.occ_thresh = int(rospy.get_param("~occ_thresh", 50))
        self.clear_footprint = rospy.get_param("~clear_footprint", True)

        # DEFAULT GOAL (2.0, 0.0), and it has to sit clear of the map edge.
        # The grid is x [-4.0, 3.0] grown outward by a 0.1 m always-occupied
        # border ring, so x = 3.0 is ON the wall and x = 2.5 is 0.5 m from it
        # -- inside r_safe (0.51 m), which the validator would refuse. 2.0
        # leaves a metre of standoff, and the vehicle starts near x = -2.9, so
        # it is still a ~4.9 m traverse.
        self.goal = np.array([rospy.get_param("~goal_x", 2.0),
                              rospy.get_param("~goal_y", 0.0)])

        self.plan_timeout = rospy.get_param("~plan_timeout", 0.5)
        self.max_step = rospy.get_param("~max_setpoint_step", 1.0)
        self.dry_run = rospy.get_param("~dry_run", False)
        self.viz_frame = rospy.get_param("~world_frame", rospy.get_param(
            "/robot/world_frame", rospy.get_param("~viz_frame", "vicon/world")))
        # 0. The rollout cloud is 30 x 31 points rebuilt every tick, 2.6 ms,
        # and it is a picture of the SEARCH. What the vehicle actually did is
        # ~nominal_path, which costs 0.03 ms and stays on. Raise it to watch
        # the sampler.
        self.viz_rollouts = int(rospy.get_param("~viz_rollouts", 0))

        self.alignment_topic = rospy.get_param(
            "~alignment_topic", "/robot/frame_alignment")
        self.align = SharedFrameAlignment(expected_world=self.viz_frame)
        self.align_timeout = float(rospy.get_param("~align_timeout", 0.5))
        self.state_timeout = float(rospy.get_param("~state_timeout", 0.5))
        self.state_pair_max_age = float(rospy.get_param("~state_pair_max_age", 0.05))

        # --------------------------------------------------------- planner
        # THE SIMULATOR'S `limits_flown`, NOT ITS `limits:` BLOCK.
        # X_bar = X (-) Z (paper eq. 11): the solver plans against the physical
        # envelope MINUS the tracking tube, so a tracking transient cannot push
        # the true state past the bound the safety argument rests on. The
        # vehicle keeps the untightened numbers; nothing here ever sees them.
        #
        #                 config.yaml limits:   (-) safety:   -> here
        #   v_max               0.35             z_vel 0.04      0.31
        #   omega_max           0.5236           z_omega 0.0349  0.4887
        #   a_max               3.5 (a_max_eff)  a_reserve 0     3.5
        #   alpha_max           3.85             alpha_reserve 0 3.85
        #   j_max               5.5              --              5.5
        #   tilt_max            35 deg           --              0.6109
        #
        # WAS 1.0 / 2.5 / 1.5 / 3.0 / 0.5236 / 8.0, which belonged to nothing:
        # the simulator never flew that envelope, so every result it produced
        # was about a vehicle three times slower in translation and yaw than
        # the one this node was commanding. a_max RISES 2.5 -> 3.5 and that is
        # not a loosening in practice -- it is a clipping bound the sampler
        # comes nowhere near (mean |a| 0.383, 11% of it), while j_max FALLS
        # 8.0 -> 5.5 and that is the bound that actually shapes the trajectory.
        # tilt_max is not binding either way: a_max_eff = min(a_max,
        # g tan(tilt)) picks 3.5 against g tan(35 deg) = 6.87.
        limits = PlanarLimits(
            v_max=rospy.get_param("~v_max", 0.31),
            a_max=rospy.get_param("~a_max", 3.5),
            omega_max=rospy.get_param("~omega_max", 0.4887),
            alpha_max=rospy.get_param("~alpha_max", 3.85),
            tilt_max=rospy.get_param("~tilt_max", 0.610865),
            j_max=rospy.get_param("~j_max", 5.5))

        # INPUT-CHANGE COST, DEFAULTED OFF.
        # R_dnu penalises |nu_k - nu_{k-1}|^2. The intent is "do not thrash the
        # controller", but MPPI's exploration noise IS an input change, so the
        # term bills every perturbed sample for being explored. Measured here
        # against the live map at sigma=1.2: mean slew cost 27.6 per sample
        # while sample 0 -- which IS the unperturbed nominal -- pays exactly 0.
        # Its spread across samples (std 6.0) rivalled the goal terms (6.6),
        # so the cheapest valid sample was consistently "change nothing". The
        # nominal then could not accumulate: dU collapsed toward zero, warm_start
        # shifted what little there was off the front, and the plan decayed from
        # +0.27 m of progress to -0.13 m over 40 closed-loop solves.
        #
        # Smoothness does not depend on this term: clip_inputs() already
        # projects onto |a_k - a_{k-1}| <= j_max*dt as a HARD constraint, so the
        # emitted plan is slew-feasible with R_dnu = 0.
        # BACK ON, at config.yaml's [1.0, 1.0, 0.2]. It was 0 because of a
        # measurement -- mean slew cost 27.6 per sample against goal terms of
        # 6.6, the nominal collapsing, 40 closed-loop solves decaying from
        # +0.27 m of progress to -0.13 m -- and that measurement was made at
        # sigma 1.2. R_dnu bills |nu_k - nu_(k-1)|^2, so it scales with
        # sigma^2: at the derived 0.3056 it costs (0.3056/1.2)^2 = 6.5% of
        # what it cost then. The term was never wrong; the search width it was
        # measured against was.
        R_a = float(rospy.get_param("~r_dnu_a", 1.0))
        R_alpha = float(rospy.get_param("~r_dnu_alpha", 0.2))

        weights = PlanarCostWeights(
            R_dnu=(R_a, R_a, R_alpha),
            w_goal=rospy.get_param("~w_goal", 1.0),
            w_term_pos=rospy.get_param("~w_term_pos", 10.0),
            w_term_vel=rospy.get_param("~w_term_vel", 2.0),
            w_obs=rospy.get_param("~w_obs", 20.0),
            # 0.60, config.yaml's. Passing null here would make it
            # r_safe + 0.35 = 0.86 -- a derivation, never a tuned value, and
            # 0.26 m wider than the one w_frontier and w_obs were swept against.
            d_influence=rospy.get_param("~d_influence", 0.60),
            w_yaw=rospy.get_param("~w_yaw", 0.0),
            yaw_mode=rospy.get_param("~yaw_mode", "velocity"))

        self.occ = None
        self.free = FreeSpace()
        if not self.require_map:
            rospy.logwarn("[planar] ~require_map is FALSE -- flying with NO "
                          "obstacle set. Open-space checks only.")

        # CappedDynamics, matching `mppi.cap_velocity: true` in the
        # simulator's config.yaml -- the velocity limit is enforced by
        # PROJECTION inside the rollout instead of by discarding the sample.
        # The node ran plain PlanarDynamics, which is a different expert: it
        # throws away every sample that exceeds v_max rather than clipping it,
        # so the accepted pool is a different distribution from the one every
        # result in the simulator was produced with.
        # SIGMA IS DERIVED, NOT CONFIGURED, and that is the whole point of
        # having moved it here. It was three hardcoded numbers -- (1.2, 1.2,
        # 1.5) -- which happened to be right for the OLD envelope and would
        # have stayed at their old values while the limits above changed
        # underneath them. Deriving it is what makes the alignment safe:
        # `mppi_sigma` reads limits, dt and horizon, so any future change to
        # any of the three carries the search width with it.
        #
        # Same formula and same constants as planar_sim/config.py:mppi_sigma.
        # Overridable per channel for a sweep; null (the default) derives.
        sigma = mppi_sigma(limits, self.dt, self.horizon)
        sigma = tuple(float(rospy.get_param(k, v)) for k, v in
                      zip(("~sigma_ax", "~sigma_ay", "~sigma_alpha"), sigma))
        rospy.loginfo("[planar] sigma=(%.4f, %.4f, %.4f)  [%s]  "
                      "v_max=%.2f a_max=%.2f omega_max=%.3f j_max=%.2f",
                      sigma[0], sigma[1], sigma[2],
                      sigma_binding(limits, self.dt, self.horizon),
                      limits.v_max, limits.a_max, limits.omega_max,
                      limits.j_max)

        self.cap_velocity = bool(rospy.get_param("~cap_velocity", True))
        dyn_cls = CappedDynamics if self.cap_velocity else PlanarDynamics
        self.dyn = dyn_cls(limits, dt=self.dt)
        self.validator = PlanarSafetyValidator(self.free, self.r_safe)
        # Always FrontierMPPI, never PlanarMPPI. At w_frontier = 0 its frontier
        # term short-circuits to zeros and at use_geodesic = False its goal
        # correction returns 0.0, so the class IS PlanarMPPI in that setting --
        # which keeps an A/B on either feature a change of a weight rather than
        # a change of code path. Mirrors build_planner() in run_planar_sim.py.
        # GEODESIC AND w_frontier 5.0, both config.yaml's. The geodesic field
        # is a Dijkstra over the traversable set rebuilt every solve, measured
        # at 24.8 ms of an 86.7 ms tick here -- affordable, and it is what
        # removes the local minima ||p - goal|| has when the goal sits behind
        # something. w_frontier was 0 only during a retune done against a
        # GROUND-TRUTH map, where the one term that rewards revealing space
        # cannot help and therefore measures as dead weight; this vehicle flies
        # a belief map. On the val split, 11/12 episodes reached at 5 against
        # 10/12 at 0, mean d_goal 0.22 m against 0.54. Tuned together with
        # w_obs 20 and d_influence 0.60, so the three travel together.
        self.use_geodesic = bool(rospy.get_param("~use_geodesic", True))
        self.w_frontier = float(rospy.get_param("~w_frontier", 5.0))
        self.planner = FrontierMPPI(
            self.dyn, self.validator, weights=weights,
            use_geodesic=self.use_geodesic,
            w_frontier=self.w_frontier,
            c_occupied=rospy.get_param("~c_occupied", 2.0),
            c_unknown=rospy.get_param("~c_unknown", -4.0),
            horizon=self.horizon, num_samples=self.num_samples,
            sigma=sigma,
            temperature=rospy.get_param("~temperature", 1.0),
            seed=int(rospy.get_param("~seed", 0)),
            goal_tol=rospy.get_param("~goal_tol", 0.25))
        rospy.loginfo("[planar] cost: geodesic=%s w_frontier=%.2f "
                      "R_dnu=(%.3g,%.3g) num_samples=%d horizon=%d "
                      "dynamics=%s [planner/ from %s]",
                      self.use_geodesic, self.w_frontier, R_a, R_alpha,
                      self.num_samples, self.horizon,
                      dyn_cls.__name__, _SRC)

        # ------------------------------------------------------------ state
        self.pose = None
        self.twist = None
        self.ref = None            # last accepted FixedAltitudeReference
        self.ref_t0 = None
        self.a_prev = np.zeros(2)
        self.hold = None
        self.ref_epoch = None
        self.pose_epoch = None
        self.map_epoch = None
        self._epoch_start = None
        self._needs_planner_reset = False
        self._state_health = {}
        self.arrived = False
        self._goal_sent = None

        # ----------------------------------------------------------- ROS I/O
        rospy.Subscriber(self.pose_topic, TransformStamped, self._pose_cb,
                         queue_size=1)
        rospy.Subscriber(self.alignment_topic, String, self._alignment_cb,
                         queue_size=1)
        rospy.Subscriber("mavros/local_position/velocity_local", TwistStamped,
                         self._twist_cb, queue_size=1)
        rospy.Subscriber(self.grid_topic, OccupancyGrid, self._grid_cb,
                         queue_size=1)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self._goal_cb,
                         queue_size=1)

        self.pub_sp = rospy.Publisher("commander/set_pose", PositionTarget,
                                      queue_size=10)
        self.pub_path = rospy.Publisher("~nominal_path", Path, queue_size=1)
        self.pub_viz = rospy.Publisher("~rollouts", MarkerArray, queue_size=1)
        self.pub_status = rospy.Publisher("~status", String, queue_size=1)
        # WHAT FLEW, latched, as JSON. A bag that does not say which producer
        # made it, under which limits and weights, is not evaluable six months
        # later -- and it is about to matter more, because the same node is
        # meant to fly the HAA, the learned HPA and the DeSimplex supervisor
        # and the three are indistinguishable from their setpoints alone.
        # Latched, so a recorder that starts after the node still gets it.
        self.pub_config = rospy.Publisher("~config", String, queue_size=1,
                                          latch=True)
        self.producer = rospy.get_param("~producer", "haa")
        # Latched so a subscriber that joins late still learns the current
        # answer instead of waiting for the next tick.
        self.arrived_topic = rospy.get_param("~arrived_topic",
                                             "/goal_arrive_tf")
        self.pub_arrived = rospy.Publisher(self.arrived_topic, Bool,
                                           queue_size=1, latch=True)
        self.arrived = False
        self._goal_sent = None
        self.pub_goal = rospy.Publisher("~goal_marker", Marker, queue_size=1,
                                        latch=True)
        # The unsafe set the validator actually uses -- occupied AND unknown AND
        # inflated by r_safe. Publishing it is the only way to see WHY a plan
        # went where it did: /grid_map alone shows neither the unknown-is-unsafe
        # rule nor the 0.59 m inflation, so a path that looks needlessly timid
        # against the raw grid is usually hugging this instead.
        self.pub_unsafe = rospy.Publisher("~inflated", OccupancyGrid,
                                          queue_size=1, latch=True)
        # OFF. It is a POST-FLIGHT question, and it costs 8.5 ms of every plan
        # tick (p95 22.8) on this box: a grid-wide distance_transform_edt plus
        # two int8 rasters of 14976 cells, built INSIDE plan_once on the
        # solve's own thread. Nothing is lost by not sending it -- the
        # inflated set is a pure function of /grid_map and r_safe, both of
        # which are in the bag, so analyze_flight.py reconstructs it exactly.
        # Turn it on to watch live in Foxglove, and expect the plan rate to
        # pay for it.
        self.publish_inflated = rospy.get_param("~publish_inflated", False)
        # The r_quad half of r_safe, drawn as a separate ring so the two parts
        # of the safety radius are distinguishable instead of one blob.
        self.r_viz_expand = float(rospy.get_param("~r_viz_expand",
                                                  self.r_quad))
        self.pub_ring = rospy.Publisher("~inflated_outer", OccupancyGrid,
                                        queue_size=1)

        self._publish_config(limits, sigma, weights, dyn_cls)
        rospy.loginfo("[planar] %s", self.planner.describe().replace("\n", "\n[planar] "))
        rospy.loginfo("[planar] r_safe=%.3f (r_Q %.2f + r_perc %.2f + "
                      "r_track %.2f + d_clr %.2f)", self.r_safe, self.r_quad,
                      self.r_perc, self.r_track, self.d_clr)
        rospy.loginfo("[planar] z0=%.2f m  plan %.1f Hz  publish %.1f Hz",
                      self.z0, self.plan_rate, self.pub_rate)
        if self.dry_run:
            rospy.logwarn("[planar] DRY RUN -- planning and visualising only, "
                          "commander/set_pose will NOT be written")

    # ------------------------------------------------------------- callbacks

    def _discard_epoch_locked(self):
        """Drop frame-dependent state; reset the solver on its own thread."""
        self.pose = self.twist = self.occ = None
        self.pose_epoch = self.map_epoch = self.ref_epoch = None
        self.ref = self.ref_t0 = self.hold = None
        self.a_prev = np.zeros(2)
        self.arrived = False
        self._goal_sent = None
        self._needs_planner_reset = True

    def _alignment_cb(self, msg):
        now = rospy.get_time()
        with self.lock:
            was_ready = self.align.ready
            changed = self.align.update_status(msg.data, now=now)
            if changed or (was_ready and not self.align.ready):
                self._discard_epoch_locked()
            self._epoch_start = self.align.valid_from
            if not self.align.ready:
                self._state_health = {"reason": self.align.reason,
                                      "epoch": self.align.epoch}

    def _pose_cb(self, msg):
        now = rospy.get_time()
        with self.lock:
            if not self.align.is_ready(now=now, max_age=self.align_timeout):
                return
            if (msg.header.frame_id != self.align.world_frame or
                    msg.child_frame_id != "ekf_body/epoch/" + self.align.epoch):
                return
            if (self._epoch_start is None or
                    msg.header.stamp.to_sec() < self._epoch_start):
                return
            # The stamped Transform message carries epoch and measurement in
            # one ROS message; the ordinary PoseStamped output is for legacy
            # visualization/recording, not an authoritative planner input.
            position = msg.transform.translation
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose = Pose(position=Point(x=position.x, y=position.y, z=position.z),
                             orientation=msg.transform.rotation)
            self.pose = pose
            self.pose_epoch = self.align.epoch

    def _twist_cb(self, msg):
        with self.lock:
            self.twist = msg

    def _grid_cb(self, msg):
        with self.lock:
            now = rospy.get_time()
            if not self.align.is_ready(now=now, max_age=self.align_timeout):
                return
            epoch = self.align.epoch
            if (msg.header.frame_id != self.align.world_frame or
                    self._epoch_start is None or
                    msg.header.stamp.to_sec() < self._epoch_start or
                    msg.info.map_load_time != rospy.Time.from_sec(self._epoch_start)):
                return
        try:
            # `occupied` and `unknown` now come from the object itself.
            # This used to recover them from the raw message and assign them
            # onto the instance, because planner/grid.py kept a single merged
            # `unsafe` set and _publish_inflated needed to grow the WALLS
            # without growing the unobserved region with them. That module
            # carries the split now -- and, with unknown_inflate, applies it in
            # clearance() as well, which the patch never could.
            occ = PlanarOccupancy.from_occupancy_grid_msg(
                msg, occ_thresh=self.occ_thresh,
                unknown_unsafe=self.unknown_unsafe,
                unknown_inflate=self.unknown_inflate,
                r_safe=self.r_safe)
        except Exception as e:
            rospy.logwarn_throttle(5.0, "[planar] grid parse failed: %s", e)
            return
        with self.lock:
            if (epoch == self.align.epoch and self.align.is_ready(
                    now=rospy.get_time(), max_age=self.align_timeout)):
                self.occ = occ
                self.map_epoch = epoch

    def _goal_cb(self, msg):
        # Empty legacy headers mean the configured world. Explicit frames
        # must agree; there is no implicit ZED-map or FCU-local goal transform.
        if msg.header.frame_id and msg.header.frame_id != self.viz_frame:
            rospy.logwarn_throttle(2.0, "[planar] goal frame %s is not %s -- ignored",
                                   msg.header.frame_id, self.viz_frame)
            return
        with self.lock:
            self.goal = np.array([msg.pose.position.x, msg.pose.position.y])
            self.arrived = False        # a new goal un-latches the hold
        rospy.loginfo("[planar] new goal (%.2f, %.2f)", self.goal[0],
                      self.goal[1])

    # ------------------------------------------------------------------ state

    def _state_snapshot_locked(self, now):
        """Matched EKF input and a fixed transform snapshot for one operation."""
        if not self.align.is_ready(now=now, max_age=self.align_timeout):
            # A heartbeat loss also retires cached commands until fresh input
            # and a fresh map arrive; static transform creation age is irrelevant.
            self._discard_epoch_locked()
            self._state_health = {"reason": "alignment_unavailable",
                                  "epoch": self.align.epoch}
            return None
        pose, twist = self.pose, self.twist
        if pose is None or twist is None or self.pose_epoch != self.align.epoch:
            self._state_health = {"reason": "waiting_for_matched_ekf_state"}
            return None
        pose_stamp = pose.header.stamp.to_sec()
        twist_stamp = twist.header.stamp.to_sec()
        pose_age, twist_age = now - pose_stamp, now - twist_stamp
        pair_age = abs(pose_stamp - twist_stamp)
        self._state_health = {"epoch": self.align.epoch,
                              "pose_stamp": pose_stamp, "twist_stamp": twist_stamp,
                              "pose_age": pose_age, "twist_age": twist_age,
                              "pair_age": pair_age}
        if (not np.all(np.isfinite([pose_stamp, twist_stamp])) or
                twist_stamp < self._epoch_start or
                min(pose_age, twist_age) < -self.state_pair_max_age or
                max(pose_age, twist_age) > self.state_timeout or
                pair_age > self.state_pair_max_age):
            self._state_health["reason"] = "stale_or_unmatched_ekf_state"
            return None
        align = self.align.snapshot()
        p, q = pose.pose.position, pose.pose.orientation
        linear = twist.twist.linear
        values = [p.x, p.y, p.z, q.x, q.y, q.z, q.w,
                  linear.x, linear.y, linear.z, twist.twist.angular.z]
        if not np.all(np.isfinite(values)):
            self._state_health["reason"] = "nonfinite_ekf_state"
            return None
        vx, vy, omega = planar_ekf_velocity(
            align, [linear.x, linear.y, linear.z], twist.twist.angular.z)
        state = PlanarState(p.x, p.y, vx, vy,
                            yaw_from_quat(q.x, q.y, q.z, q.w), omega)
        self._state_health["reason"] = "ready"
        return state, align

    def _current_state(self):
        """Build a fresh, paired EKF state in the map frame, or None."""
        with self.lock:
            snapshot = self._state_snapshot_locked(rospy.get_time())
            return None if snapshot is None else snapshot[0]

    def _epoch_usable_locked(self, epoch, now):
        return (self.align.epoch == epoch and
                self.align.is_ready(now=now, max_age=self.align_timeout) and
                (not self.require_map or
                 (self.occ is not None and self.map_epoch == epoch)))

    # --------------------------------------------------------------- planning

    def plan_once(self, _evt=None):
        if not self.plan_lock.acquire(False):
            return
        try:
            self._plan_once()
        finally:
            self.plan_lock.release()

    def _plan_once(self):
        with self.lock:
            snapshot = self._state_snapshot_locked(rospy.get_time())
            if snapshot is None:
                rospy.logwarn_throttle(2.0, "[planar] waiting for fresh paired EKF state")
                self._publish_status("WAITING_EKF_STATE")
                return
            state, alignment = snapshot
            epoch = alignment.epoch
            if not self._epoch_usable_locked(epoch, rospy.get_time()):
                rospy.logwarn_throttle(2.0, "[planar] waiting for current-epoch grid")
                self._publish_status("WAITING_EPOCH_GRID")
                return
            if self._needs_planner_reset:
                self.planner.reset()
                self._needs_planner_reset = False
            occ, goal = self.occ, self.goal.copy()
            a_prev = self.a_prev.copy()

        # ---------------------------------------------------- goal arrival
        # Checked BEFORE solving: once the goal is reached there is nothing to
        # plan, and continuing to solve would keep nudging the vehicle around
        # inside goal_tol. Latched, because ||p - goal|| dithers across the
        # tolerance and an unlatched test would flip in and out of hover.
        # Only a NEW goal clears it (see _goal_cb).
        with self.lock:
            if not self._epoch_usable_locked(epoch, rospy.get_time()):
                return
            if self.arrived or self.planner.at_goal(state, goal):
                if not self.arrived:
                    self.arrived = True
                    self.hold = (state.x, state.y, state.psi)
                    rospy.loginfo("[planar] GOAL REACHED (%.2f, %.2f) -- holding",
                                  state.x, state.y)
                self.ref = self.ref_t0 = None
                self.ref_epoch = epoch
                self.pub_arrived.publish(Bool(data=True))
                self._publish_goal(goal)
                self._publish_status("ARRIVED holding (%.2f, %.2f)"
                                     % (self.hold[0], self.hold[1]))
                return
            self.pub_arrived.publish(Bool(data=False))

        if self.require_map:
            if occ is None:
                rospy.logwarn_throttle(2.0, "[planar] waiting for %s (set "
                                            "~require_map:=false for open "
                                            "space)", self.grid_topic)
                return
            if self.clear_footprint:
                # The vehicle's own cells are frequently unknown -- the ZED
                # cannot see underneath itself -- and unknown is unsafe. Without
                # this the first node always collides.
                occ.clear_disc(state.x, state.y, self.r_safe + 0.05)
            self.validator.occ = occ
        else:
            self.validator.occ = self.free

        t0 = rospy.get_time()
        res = self.planner.plan(state, goal, a_prev=a_prev,
                                n_viz=self.viz_rollouts)
        solve_ms = (rospy.get_time() - t0) * 1000.0

        with self.lock:
            if not self._epoch_usable_locked(epoch, rospy.get_time()):
                # Callbacks never mutate the solver while it is running.
                self._needs_planner_reset = True
                return
            if not res.ok:
                rospy.logerr_throttle(1.0, "[planar] %s (%s) -> holding",
                                      res.status, res.reason)
                self.ref = self.ref_t0 = None
                self.ref_epoch = epoch
                self._publish_status("%s %s solve=%.0fms"
                                     % (res.status, res.reason, solve_ms))
                return
            if res.degraded:
                rospy.logwarn_throttle(1.0, "[planar] degraded: %s (%s)",
                                       res.status, res.reason)
            ref = res.reference.lift(self.z0)
            self.ref = ref
            self.ref_t0 = rospy.get_time()
            self.ref_epoch = epoch
            self.a_prev = res.reference.a[0].copy()

        # TIMED, because these run on the solve's own thread and a subscriber
        # appearing -- a rosbag, a Foxglove panel -- is enough to switch them
        # on. Without this in the status line, "the planner got slower when I
        # started recording" is invisible.
        t1 = rospy.get_time()
        self._publish_path(ref)
        self._publish_rollouts(res.X_viz)
        self._publish_goal(goal)
        self._publish_inflated(occ)
        viz_ms = (rospy.get_time() - t1) * 1000.0

        self._publish_status(
            "%s valid=%d/%d beta=%.3g cost=%.1f solve=%.0fms viz=%.0fms | %s %s"
            % (res.status, res.n_valid, res.n_samples, res.beta, res.cost,
               solve_ms, viz_ms, "alignment_epoch=%s" % epoch, res.reason))
        if viz_ms > 0.15 * solve_ms and viz_ms > 5.0:
            rospy.logwarn_throttle(
                10.0, "[planar] visualisation is %.0f ms of a %.0f ms tick -- "
                      "something subscribed to ~rollouts or ~inflated",
                viz_ms, solve_ms + viz_ms)

        if solve_ms > 1000.0 / self.plan_rate:
            rospy.logwarn_throttle(
                5.0, "[planar] solve %.0f ms over the %.0f ms plan period -- "
                     "lower ~num_samples or ~horizon", solve_ms,
                     1000.0 / self.plan_rate)

    # ------------------------------------------------------------- publishing

    def publish_reference(self, _evt=None):
        """Emit a reference only while its state, map and alignment agree."""
        if self.dry_run:
            return
        # This short critical section prevents an epoch reset between the
        # inverse transform and publication. Solves run outside this lock.
        with self.lock:
            now = rospy.get_time()
            snapshot = self._state_snapshot_locked(now)
            if snapshot is None:
                return
            state, alignment = snapshot
            if not self._epoch_usable_locked(alignment.epoch, now):
                return
            if self.ref_epoch is not None and self.ref_epoch != alignment.epoch:
                self.ref = self.ref_t0 = self.hold = None
                return
            if self.hold is None:
                self.hold = (state.x, state.y, state.psi)
            ref, t0 = self.ref, self.ref_t0
            pt = None
            if ref is not None and t0 is not None:
                age = now - t0
                if 0.0 <= age <= self.plan_timeout:
                    pt = ref.sample(age)
                else:
                    rospy.logwarn_throttle(2.0, "[planar] plan stale (%.2fs) -> holding", age)
            if pt is None:
                p = np.array([self.hold[0], self.hold[1], self.z0])
                v, a = np.zeros(3), np.zeros(3)
                yaw, yaw_rate = self.hold[2], 0.0
            else:
                p, v, a = pt.p.copy(), pt.v.copy(), pt.a.copy()
                yaw, yaw_rate = pt.psi, pt.psi_dot
                self.hold = (p[0], p[1], yaw)
            p = self._limit(p, state)
            p_fcu, yaw_fcu = alignment.world_to_local(p, yaw)
            v_fcu = alignment.rotate_to_local(v)
            a_fcu = alignment.rotate_to_local(a)
            command = self.controller.construct_target_full(
                p_fcu, v_fcu, a_fcu, yaw_fcu, yaw_rate)
            # Internal commander contract: the versioned local frame identifies
            # the transform used for this command even across callback ordering.
            # mission_node validates it and restores fcu_local before MAVROS.
            command.header.frame_id = (alignment.local_frame + "/epoch/" +
                                       alignment.epoch)
            self.pub_sp.publish(command)

    def _limit(self, p, state):
        """Bound how far the emitted setpoint may sit from the vehicle."""
        dx, dy = p[0] - state.x, p[1] - state.y
        d = float(np.hypot(dx, dy))
        if d > self.max_step and d > 1e-9:
            s = self.max_step / d
            rospy.logwarn_throttle(2.0, "[planar] setpoint %.2fm away -> "
                                        "clamped to %.2fm", d, self.max_step)
            return np.array([state.x + dx * s, state.y + dy * s, self.z0])
        return np.array([p[0], p[1], self.z0])

    def _publish_path(self, ref):
        if self.pub_path.get_num_connections() == 0:
            return
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self.viz_frame
        for row in ref.p:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(row[0])
            ps.pose.position.y = float(row[1])
            ps.pose.position.z = float(row[2])
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.pub_path.publish(path)

    def _publish_rollouts(self, X_viz):
        if self.pub_viz.get_num_connections() == 0 or X_viz is None:
            return
        arr = MarkerArray()
        stamp = rospy.Time.now()

        wipe = Marker()
        wipe.header.frame_id = self.viz_frame
        wipe.header.stamp = stamp
        wipe.ns = "planar_rollouts"
        wipe.action = Marker.DELETEALL
        arr.markers.append(wipe)

        for i, Xi in enumerate(X_viz):
            m = Marker()
            m.header.frame_id = self.viz_frame
            m.header.stamp = stamp
            m.ns = "planar_rollouts"
            m.id = i
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.01
            m.color.r, m.color.g, m.color.b, m.color.a = 0.3, 0.6, 1.0, 0.25
            m.pose.orientation.w = 1.0
            m.points = [Point(float(r[0]), float(r[1]), float(self.z0))
                        for r in Xi]
            m.lifetime = rospy.Duration(0.5)
            arr.markers.append(m)
        self.pub_viz.publish(arr)

    def _publish_goal(self, goal):
        # LATCHED, so re-sending an unchanged goal every tick tells nobody
        # anything -- a late subscriber gets the latched copy regardless.
        g = (float(goal[0]), float(goal[1]))
        if self._goal_sent == g:
            return
        self._goal_sent = g
        m = Marker()
        m.header.frame_id = self.viz_frame
        m.header.stamp = rospy.Time.now()
        m.ns = "planar_goal"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(goal[0])
        m.pose.position.y = float(goal[1])
        m.pose.position.z = float(self.z0)
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 2.0 * self.planner.goal_tol
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 1.0, 0.2, 0.6
        self.pub_goal.publish(m)

    def _publish_outer_ring(self, occ, d_occ, inner):
        """One more inflation on top of the r_eff band. Visualisation only.

        Only the RING is published -- cells inside the outer inflation that are
        not already in the inner band. Publishing the filled disc would cover
        the inner band and the two would be indistinguishable.

            inner band  r_eff  = 0.28 m   map uncertainty
            + this ring r_quad = 0.31 m   the airframe's own disc
            = r_safe             0.59 m   what the validator gates on

        Colour is a per-topic setting in the Foxglove 3D panel; it is
        deliberately not baked into the message.
        """
        if self.pub_ring.get_num_connections() == 0:
            return
        ring = (d_occ < (self.r_eff + self.r_viz_expand)) & ~inner

        g = OccupancyGrid()
        g.header.stamp = rospy.Time.now()
        g.header.frame_id = occ.frame_id or self.viz_frame
        g.info.resolution = occ.res
        g.info.width = occ.W
        g.info.height = occ.H
        g.info.origin.position.x = occ.origin[0]
        g.info.origin.position.y = occ.origin[1]
        g.info.origin.position.z = 0.0
        g.info.origin.orientation.w = 1.0
        g.data = np.where(ring, 100, -1).astype(np.int8).reshape(-1).tolist()
        self.pub_ring.publish(g)

    def _publish_inflated(self, occ):
        """The obstacle set grown by r_eff, as a transparent overlay.

        WHAT CHANGED, AND WHY
        This used to publish `clearance(...) < r_safe`, which was wrong twice:

          1. clearance() runs its EDT over `unsafe`, which is obstacle OR
             unknown, so it grew the UNKNOWN region too. Early in a flight
             nearly everything is unknown, so the layer was a near-solid block
             that said nothing about where the walls are.
          2. it grew obstacles by r_safe = 0.59 m. r_safe is the centre-to-
             obstacle gate and already contains r_quad; the airframe disc is
             not part of what the MAP should be inflated by. Growing the map by
             it double-draws the vehicle radius.

        What the map should absorb is only the uncertainty about where the
        obstacle actually is:

            r_eff = r_perc + r_track + d_clr = r_safe - r_quad = 0.28 m

        The remaining r_quad is drawn separately by _publish_outer_ring, so
        inner + ring still adds up to the 0.59 m the validator gates on.

        Everything outside the band is published as -1, which Foxglove draws as
        nothing. Publishing 0 for free cells paints an opaque sheet over the
        whole extent and hides /grid_map underneath -- which is what this did
        before. The raw occupied cells are excluded too, so the obstacle being
        inflated stays visible from /grid_map instead of vanishing under its
        own inflation.

        A VIEW, NOT THE GATE. The validator still refuses unknown space, so
        cells that look free here can still be rejected.
        """
        if not self.publish_inflated or occ is None:
            return
        if (self.pub_unsafe.get_num_connections() == 0
                and self.pub_ring.get_num_connections() == 0):
            return
        if not hasattr(occ, "occupied"):
            return
        d_occ = distance_transform_edt(~occ.occupied) * occ.res
        blocked = (d_occ < self.r_eff) & ~occ.occupied
        self._publish_outer_ring(occ, d_occ, d_occ < self.r_eff)

        g = OccupancyGrid()
        g.header.stamp = rospy.Time.now()
        g.header.frame_id = occ.frame_id or self.viz_frame
        g.info.resolution = occ.res
        g.info.width = occ.W
        g.info.height = occ.H
        g.info.origin.position.x = occ.origin[0]
        g.info.origin.position.y = occ.origin[1]
        g.info.origin.position.z = 0.0
        g.info.origin.orientation.w = 1.0
        g.data = np.where(blocked, 100, -1).astype(np.int8).reshape(-1).tolist()
        self.pub_unsafe.publish(g)

    def _publish_config(self, limits, sigma, weights, dyn_cls):
        """One latched JSON blob describing everything that decides a solve."""
        try:
            sha = subprocess.check_output(
                ["git", "-C", _SRC, "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL).decode().strip()
            dirty = bool(subprocess.check_output(
                ["git", "-C", _SRC, "status", "--porcelain", "planner"],
                stderr=subprocess.DEVNULL).strip())
        except Exception:
            sha, dirty = "unknown", None
        cfg = {
            # WHICH PLANNER. The one field a comparison run cannot do without.
            "producer": self.producer,
            "planner_class": type(self.planner).__name__,
            "dynamics": dyn_cls.__name__,
            "src": _SRC, "git": sha, "planner_dirty": dirty,
            "limits": {"v_max": limits.v_max, "a_max": limits.a_max,
                       "omega_max": limits.omega_max,
                       "alpha_max": limits.alpha_max,
                       "tilt_max": limits.tilt_max, "j_max": limits.j_max},
            "sigma": list(sigma),
            "mppi": {"horizon": self.horizon, "num_samples": self.num_samples,
                     "dt": self.dt, "temperature": self.planner.temperature,
                     "goal_tol": self.planner.goal_tol,
                     "cap_velocity": self.cap_velocity,
                     "seed": int(rospy.get_param("~seed", 0)),
                     "use_geodesic": self.use_geodesic,
                     "w_frontier": self.w_frontier,
                     # The frontier term's two costs. Left out of the first
                     # version of this blob, which made a w_frontier sweep
                     # unattributable: the weight was recorded but not what
                     # it scaled.
                     "c_occupied": rospy.get_param("~c_occupied", 2.0),
                     "c_unknown": rospy.get_param("~c_unknown", -4.0)},
            "weights": {"w_goal": weights.w_goal,
                        "w_term_pos": weights.w_term_pos,
                        "w_term_vel": weights.w_term_vel,
                        "w_obs": weights.w_obs,
                        "d_influence": weights.d_influence,
                        "R_dnu": list(weights.R_dnu),
                        "w_yaw": weights.w_yaw,
                        "yaw_mode": weights.yaw_mode},
            "safety": {"r_quad": self.r_quad, "r_perc": self.r_perc,
                       "r_track": self.r_track, "d_clr": self.d_clr,
                       "r_safe": self.r_safe,
                       "unknown_unsafe": self.unknown_unsafe,
                       "unknown_inflate": self.unknown_inflate},
            "goal": [float(self.goal[0]), float(self.goal[1])],
            "rates": {"plan_rate": self.plan_rate, "pub_rate": self.pub_rate,
                      "plan_timeout": self.plan_timeout},
            "z0": self.z0, "dry_run": self.dry_run,
            "grid_topic": self.grid_topic, "pose_topic": self.pose_topic,
            "state_adapter": {"source": "mavros_ekf", "angular_z": "preserved",
                              "alignment_topic": self.alignment_topic,
                              "alignment": "fixed_shared_W_from_L",
                              "state_timeout": self.state_timeout,
                              "pair_max_age": self.state_pair_max_age,
                              "alignment_timeout": self.align_timeout},
        }
        self.pub_config.publish(String(data=json.dumps(cfg, sort_keys=True)))
        rospy.loginfo("[planar] producer=%s planner=%s git=%s%s",
                      self.producer, type(self.planner).__name__, sha,
                      " (planner/ DIRTY)" if dirty else "")

    def _publish_status(self, text):
        with self.lock:
            health = dict(self._state_health)
        self.pub_status.publish(String(data=text + " | ekf=" +
                                      json.dumps(health, sort_keys=True)))

    # --------------------------------------------------------------- run loop

    def start(self):
        rospy.loginfo("[planar] waiting for EKF pose, velocity, shared alignment and grid")
        rospy.Timer(rospy.Duration(1.0 / self.plan_rate), self.plan_once)
        rospy.Timer(rospy.Duration(1.0 / self.pub_rate), self.publish_reference)
        rospy.spin()


if __name__ == "__main__":
    PlanarPlannerNode().start()
