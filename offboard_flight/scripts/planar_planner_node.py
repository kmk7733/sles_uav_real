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
The planner works in the grid's frame (vicon/world), because a planner must be
in the same frame as the obstacles it is avoiding. mavros consumes the FCU's
local ENU frame, which does not coincide with vicon/world -- EKF2 here is fed by
the ZED, not by Vicon. The conversion happens at the last possible moment, in
publish_reference(), and if the alignment is not established NO setpoint is
emitted at all. Note that velocity and acceleration are rotated only
(rotate_to_fcu), never translated.

ROLL AND PITCH ARE NOT PLANNED
The planner commands a planar acceleration. PX4 turns that into attitude. That
is the whole point of the fixed-altitude planar formulation, and it is why the
reference published here has vz = az = 0: altitude is held by the low-level
controller and was never a planning variable.
"""

import threading

import numpy as np
import rospy

from geometry_msgs.msg import PoseStamped, TwistStamped, Point
from mavros_msgs.msg import PositionTarget
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from guidance_library import Controller
from haa_frames import WorldToFcu, yaw_from_quat
from planar_types import PlanarState, PlannerStatus
from planar_dynamics import PlanarDynamics, PlanarLimits
from planar_map import PlanarOccupancy, FreeSpace
from planar_safety import PlanarSafetyValidator, safe_radius
from planar_mppi import PlanarMPPI, PlanarCostWeights


class PlanarPlannerNode(object):

    def __init__(self):
        rospy.init_node("planar_planner_node")
        self.controller = Controller()
        self.lock = threading.Lock()

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
        self.horizon = int(rospy.get_param("~horizon", 20))
        self.num_samples = int(rospy.get_param("~num_samples", 128))

        self.r_quad = rospy.get_param("~r_quad", 0.31)
        self.r_perc = rospy.get_param("~r_perc", 0.18)
        self.r_track = rospy.get_param("~r_track", 0.05)
        self.d_clr = rospy.get_param("~d_clr", 0.05)
        self.r_safe = safe_radius(self.r_quad, self.r_perc, self.r_track,
                                  self.d_clr)

        self.pose_topic = rospy.get_param("~pose_topic", "/robot/pose_world")
        self.grid_topic = rospy.get_param("~grid_topic", "/grid_map")
        self.require_map = rospy.get_param("~require_map", True)
        self.unknown_unsafe = rospy.get_param("~unknown_unsafe", True)
        self.occ_thresh = int(rospy.get_param("~occ_thresh", 50))
        self.clear_footprint = rospy.get_param("~clear_footprint", True)

        self.goal = np.array([rospy.get_param("~goal_x", 2.5),
                              rospy.get_param("~goal_y", 0.0)])

        self.plan_timeout = rospy.get_param("~plan_timeout", 0.5)
        self.max_step = rospy.get_param("~max_setpoint_step", 1.0)
        self.dry_run = rospy.get_param("~dry_run", False)
        self.viz_frame = rospy.get_param("~viz_frame", "vicon/world")
        self.viz_rollouts = int(rospy.get_param("~viz_rollouts", 30))

        self.align = WorldToFcu(
            alpha=rospy.get_param("~align_alpha", 0.05),
            min_updates=rospy.get_param("~align_min_updates", 10))
        self.align_timeout = rospy.get_param("~align_timeout", 1.0)

        # --------------------------------------------------------- planner
        limits = PlanarLimits(
            v_max=rospy.get_param("~v_max", 1.5),
            a_max=rospy.get_param("~a_max", 2.5),
            omega_max=rospy.get_param("~omega_max", 1.5),
            alpha_max=rospy.get_param("~alpha_max", 3.0),
            tilt_max=rospy.get_param("~tilt_max", 0.5236),
            j_max=rospy.get_param("~j_max", 8.0))

        weights = PlanarCostWeights(
            w_goal=rospy.get_param("~w_goal", 1.0),
            w_term_pos=rospy.get_param("~w_term_pos", 10.0),
            w_term_vel=rospy.get_param("~w_term_vel", 2.0),
            w_obs=rospy.get_param("~w_obs", 20.0),
            w_yaw=rospy.get_param("~w_yaw", 0.0),
            yaw_mode=rospy.get_param("~yaw_mode", "velocity"))

        self.occ = None
        self.free = FreeSpace()
        if not self.require_map:
            rospy.logwarn("[planar] ~require_map is FALSE -- flying with NO "
                          "obstacle set. Open-space checks only.")

        self.dyn = PlanarDynamics(limits, dt=self.dt)
        self.validator = PlanarSafetyValidator(self.free, self.r_safe)
        self.planner = PlanarMPPI(
            self.dyn, self.validator, weights=weights,
            horizon=self.horizon, num_samples=self.num_samples,
            sigma=(rospy.get_param("~sigma_ax", 1.2),
                   rospy.get_param("~sigma_ay", 1.2),
                   rospy.get_param("~sigma_alpha", 1.5)),
            temperature=rospy.get_param("~temperature", 1.0),
            seed=int(rospy.get_param("~seed", 0)),
            goal_tol=rospy.get_param("~goal_tol", 0.25))

        # ------------------------------------------------------------ state
        self.pose = None
        self.fcu_pose = None
        self.twist = None
        self.ref = None            # last accepted FixedAltitudeReference
        self.ref_t0 = None
        self.a_prev = np.zeros(2)
        self.hold = None

        # ----------------------------------------------------------- ROS I/O
        rospy.Subscriber(self.pose_topic, PoseStamped, self._pose_cb,
                         queue_size=1)
        rospy.Subscriber("mavros/local_position/pose", PoseStamped,
                         self._fcu_pose_cb, queue_size=1)
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
        self.pub_goal = rospy.Publisher("~goal_marker", Marker, queue_size=1,
                                        latch=True)
        # The unsafe set the validator actually uses -- occupied AND unknown AND
        # inflated by r_safe. Publishing it is the only way to see WHY a plan
        # went where it did: /grid_map alone shows neither the unknown-is-unsafe
        # rule nor the 0.59 m inflation, so a path that looks needlessly timid
        # against the raw grid is usually hugging this instead.
        self.pub_unsafe = rospy.Publisher("~inflated", OccupancyGrid,
                                          queue_size=1, latch=True)
        self.publish_inflated = rospy.get_param("~publish_inflated", True)

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

    def _pose_cb(self, msg):
        with self.lock:
            self.pose = msg
        self._try_align()

    def _fcu_pose_cb(self, msg):
        with self.lock:
            self.fcu_pose = msg
        self._try_align()

    def _twist_cb(self, msg):
        with self.lock:
            self.twist = msg

    def _try_align(self):
        with self.lock:
            w, f = self.pose, self.fcu_pose
        if w is None or f is None:
            return
        qw, qf = w.pose.orientation, f.pose.orientation
        self.align.update(
            [w.pose.position.x, w.pose.position.y, w.pose.position.z],
            yaw_from_quat(qw.x, qw.y, qw.z, qw.w), w.header.stamp.to_sec(),
            [f.pose.position.x, f.pose.position.y, f.pose.position.z],
            yaw_from_quat(qf.x, qf.y, qf.z, qf.w), f.header.stamp.to_sec())

    def _grid_cb(self, msg):
        try:
            occ = PlanarOccupancy.from_occupancy_grid_msg(
                msg, occ_thresh=self.occ_thresh,
                unknown_unsafe=self.unknown_unsafe)
        except Exception as e:
            rospy.logwarn_throttle(5.0, "[planar] grid parse failed: %s", e)
            return
        with self.lock:
            self.occ = occ

    def _goal_cb(self, msg):
        with self.lock:
            self.goal = np.array([msg.pose.position.x, msg.pose.position.y])
        rospy.loginfo("[planar] new goal (%.2f, %.2f)", self.goal[0],
                      self.goal[1])

    # ------------------------------------------------------------------ state

    def _current_state(self):
        """Build a PlanarState from the latest pose and twist, or None."""
        with self.lock:
            pose, twist = self.pose, self.twist
        if pose is None:
            return None
        p = pose.pose.position
        q = pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        if twist is not None:
            vx, vy = twist.twist.linear.x, twist.twist.linear.y
            omega = twist.twist.angular.z
        else:
            vx = vy = omega = 0.0
        return PlanarState(p.x, p.y, vx, vy, yaw, omega)

    # --------------------------------------------------------------- planning

    def plan_once(self, _evt=None):
        state = self._current_state()
        if state is None:
            rospy.logwarn_throttle(2.0, "[planar] waiting for %s",
                                   self.pose_topic)
            return

        with self.lock:
            occ, goal = self.occ, self.goal.copy()

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
        res = self.planner.plan(state, goal, a_prev=self.a_prev,
                                n_viz=self.viz_rollouts)
        solve_ms = (rospy.get_time() - t0) * 1000.0

        if not res.ok:
            rospy.logerr_throttle(1.0, "[planar] %s (%s) -> holding",
                                  res.status, res.reason)
            self.ref = None
            self._publish_status("%s %s solve=%.0fms"
                                 % (res.status, res.reason, solve_ms))
            return

        if res.degraded:
            rospy.logwarn_throttle(1.0, "[planar] degraded: %s (%s)",
                                   res.status, res.reason)

        self.ref = res.reference.lift(self.z0)
        self.ref_t0 = rospy.get_time()
        self.a_prev = res.reference.a[0].copy()

        self._publish_path(self.ref)
        self._publish_rollouts(res.X_viz)
        self._publish_goal(goal)
        self._publish_inflated(occ)
        self._publish_status(
            "%s valid=%d/%d beta=%.3g cost=%.1f solve=%.0fms | %s %s"
            % (res.status, res.n_valid, res.n_samples, res.beta, res.cost,
               solve_ms, self.align.describe(), res.reason))

        if solve_ms > 1000.0 / self.plan_rate:
            rospy.logwarn_throttle(
                5.0, "[planar] solve %.0f ms over the %.0f ms plan period -- "
                     "lower ~num_samples or ~horizon", solve_ms,
                     1000.0 / self.plan_rate)

    # ------------------------------------------------------------- publishing

    def publish_reference(self, _evt=None):
        """Sample the plan at the elapsed time and emit it to the controller."""
        if self.dry_run:
            return
        state = self._current_state()
        if state is None:
            return

        if self.hold is None:
            self.hold = (state.x, state.y, state.psi)

        ref, t0 = self.ref, self.ref_t0
        pt = None
        if ref is not None and t0 is not None:
            age = rospy.get_time() - t0
            if age <= self.plan_timeout:
                pt = ref.sample(age)
            else:
                rospy.logwarn_throttle(2.0, "[planar] plan stale (%.2fs) -> "
                                            "holding", age)

        if pt is None:
            # No usable plan: hold position, zero feedforward. Not the braking
            # trajectory -- that is the planner's job and it already tried.
            p = np.array([self.hold[0], self.hold[1], self.z0])
            v = np.zeros(3)
            a = np.zeros(3)
            yaw, yaw_rate = self.hold[2], 0.0
        else:
            p, v, a = pt.p.copy(), pt.v.copy(), pt.a.copy()
            yaw, yaw_rate = pt.psi, pt.psi_dot
            self.hold = (p[0], p[1], yaw)

        p = self._limit(p, state)

        if not self.align.ready:
            rospy.logwarn_throttle(2.0, "[planar] no world->FCU alignment "
                                        "(%d pairs) -- withholding setpoint",
                                   self.align.n_updates)
            return
        if self.align.age(rospy.get_time()) > self.align_timeout:
            rospy.logwarn_throttle(2.0, "[planar] alignment stale -- "
                                        "withholding setpoint")
            return

        p_fcu, yaw_fcu = self.align.to_fcu(p, yaw)
        v_fcu = self.align.rotate_to_fcu(v)       # free vectors: rotate only,
        a_fcu = self.align.rotate_to_fcu(a)       # never translate

        self.pub_sp.publish(
            self.controller.construct_target_full(p_fcu, v_fcu, a_fcu,
                                                  yaw_fcu, yaw_rate))

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

    def _publish_inflated(self, occ):
        """Republish the validator's unsafe set as an OccupancyGrid.

        100 where a trajectory node would be rejected (clearance < r_safe),
        0 where it is flyable. Overlay this on /grid_map in Foxglove and the
        planner's behaviour stops being mysterious.
        """
        if not self.publish_inflated or occ is None:
            return
        if self.pub_unsafe.get_num_connections() == 0:
            return
        xs = occ.origin[0] + (np.arange(occ.W) + 0.5) * occ.res
        ys = occ.origin[1] + (np.arange(occ.H) + 0.5) * occ.res
        gx, gy = np.meshgrid(xs, ys)
        blocked = occ.clearance(gx, gy) < self.r_safe

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
        g.data = np.where(blocked, 100, 0).astype(np.int8).reshape(-1).tolist()
        self.pub_unsafe.publish(g)

    def _publish_status(self, text):
        self.pub_status.publish(String(data=text))

    # --------------------------------------------------------------- run loop

    def start(self):
        rospy.loginfo("[planar] waiting for %s ...", self.pose_topic)
        while not rospy.is_shutdown() and self.pose is None:
            rospy.sleep(0.1)
        rospy.loginfo("[planar] pose acquired")
        if not self.dry_run:
            rospy.loginfo("[planar] waiting for world->FCU alignment ...")
            while not rospy.is_shutdown() and not self.align.ready:
                rospy.sleep(0.1)
            rospy.loginfo("[planar] %s", self.align.describe())

        rospy.Timer(rospy.Duration(1.0 / self.plan_rate), self.plan_once)
        rospy.Timer(rospy.Duration(1.0 / self.pub_rate), self.publish_reference)
        rospy.spin()


if __name__ == "__main__":
    PlanarPlannerNode().start()
