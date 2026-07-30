#!/usr/bin/python
# -*- coding: utf-8 -*-
"""HAA planner node: MPPI over the full quadrotor model -> PX4 setpoints.

Drop-in replacement for path_generation.py. Same contract:

    setpoint_buffer.py  arms, takes off to takeoff_height, then relays whatever
                        arrives on  commander/set_pose  to
                        mavros/setpoint_raw/local  at 20 Hz.
    this node           publishes the HAA nominal reference on
                        commander/set_pose.

So setpoint_buffer.py is unchanged, and it stays the last line of defence: if
this node dies, setpoint_buffer keeps republishing the last setpoint and the
vehicle hovers.

    Terminal A:  ROS_NAMESPACE=rogx2 python setpoint_buffer.py
    Terminal B:  ROS_NAMESPACE=rogx2 python haa_planner_node.py

FRAME
The planner works in the GRID's frame -- vicon/world -- not the FCU's. That is
forced by two things: /grid_map is published in vicon/world, and the planner's
state must be in the same frame as the obstacles it is avoiding (mpc.launch
makes the same point about its own pose_topic). HPA also needs a global frame.

So the state comes from ~pose_topic (/robot/pose_world, bridged into
vicon/world by pose_to_world.py), and the plan is produced in vicon/world.

But mavros/setpoint_raw/local consumes the FCU's local ENU frame, which is
anchored wherever EKF2 started -- and EKF2 here is fed by the ZED, not by
Vicon, so it does NOT coincide with vicon/world. The conversion happens at the
last possible moment, in publish_setpoint(), using the online alignment in
haa_frames.WorldToFcu. If that alignment is not established, no setpoint is
emitted at all: a vicon/world coordinate sent raw to the FCU would command a
point metres from where it was meant to be.

PX4 is the ancillary feedback controller kappa of eq (6): we hand it the
nominal trajectory and it closes the loop. The tube Z that its tracking error
induces is what ~r_track inflates obstacles by.
"""

import numpy as np
import rospy
import threading

from geometry_msgs.msg import PoseStamped, TwistStamped, Point
from sensor_msgs.msg import Imu
from mavros_msgs.msg import PositionTarget
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String

from attitude_library import Attitude
from guidance_library import Controller
from haa_dynamics import (QuadrotorDynamics, ThrustRateQuadrotor,
                          AttitudeStabilizedQuadrotor,
                          ROGX_MASS, ROGX_INERTIA, state_from_pose)
from haa_obstacles import InflatedGrid, effective_radius
from haa_mppi import HAAMPPI
from haa_frames import WorldToFcu, yaw_from_quat


class _FreeSpace(object):
    """Stand-in obstacle set for runs without a map: nothing is ever blocked.

    Only selected when ~require_map is false. It exists so that a no-map run
    fails open deliberately and visibly (we log it loudly) instead of by
    accident -- InflatedGrid on purpose reports everything blocked when it has
    no map, so it can never silently behave like free space.
    """

    ready = True
    r_eff = 0.0

    def collides(self, x, y, **kw):
        return np.zeros(np.asarray(x).shape, dtype=bool)

    def distance_to_blocked(self, x, y):
        return float("inf")


def _pt(row):
    """State row -> geometry_msgs/Point for a marker line strip."""
    p = Point()
    p.x, p.y, p.z = float(row[0]), float(row[1]), float(row[2])
    return p


class HAAPlannerNode(object):

    def __init__(self):
        rospy.init_node("haa_planner_node")

        self.attitude = Attitude()
        self.controller = Controller()
        self.lock = threading.Lock()

        # ---------------------------------------------------------- params
        self.z_hold = rospy.get_param("~z_hold", self.controller.takeoff_height)
        self.plan_rate = rospy.get_param("~plan_rate", 10.0)
        self.pub_rate = rospy.get_param("~pub_rate", 20.0)
        self.dt = rospy.get_param("~mppi_dt", 0.1)
        self.horizon = rospy.get_param("~horizon", 15)
        self.num_samples = rospy.get_param("~num_samples", 256)

        # obstacle inflation, eq (79): r_eff = r_Q + r_track + r_perc
        self.r_quad = rospy.get_param("~robot_radius", 0.31)
        self.r_track = rospy.get_param("~r_track", 0.05)   # today's 5 cm margin
        self.r_perc = rospy.get_param("~r_perc", 0.0)
        self.r_eff = effective_radius(self.r_quad, self.r_track, self.r_perc)

        self.pose_topic = rospy.get_param("~pose_topic", "/robot/pose_world")
        self.require_map = rospy.get_param("~require_map", True)
        # Goal is in the PLANNING frame (vicon/world), same as the grid -- no
        # conversion here, only the emitted setpoint gets mapped to the FCU
        # frame. Defaults match mpc.launch.
        self.goal = np.array([rospy.get_param("~goal_x", 2.5),
                              rospy.get_param("~goal_y", 0.0)])
        self.use_default_goal = rospy.get_param("~use_default_goal", True)

        # safety limits applied to the emitted setpoint
        self.max_step = rospy.get_param("~max_setpoint_step", 1.0)
        self.plan_timeout = rospy.get_param("~plan_timeout", 0.5)
        self.yaw_mode = rospy.get_param("~yaw_mode", "hold")   # hold | plan

        # dry_run plans and visualises but never emits a setpoint, so the node
        # can be exercised on the bench with no risk of commanding the vehicle.
        self.dry_run = rospy.get_param("~dry_run", False)
        self.viz_rollouts = int(rospy.get_param("~viz_rollouts", 40))
        self.viz_frame = rospy.get_param("~viz_frame", "vicon/world")

        # Bench visualisation without a vehicle: synthesise a hovering state
        # instead of waiting for mavros. Refused unless ~dry_run is also set,
        # so a planted pose can never drive a real setpoint.
        self.fake_pose = rospy.get_param("~fake_pose", False)
        self.fake_state = None

        self.align = WorldToFcu(
            alpha=rospy.get_param("~align_alpha", 0.05),
            min_updates=rospy.get_param("~align_min_updates", 10))
        self.align_timeout = rospy.get_param("~align_timeout", 1.0)
        if self.fake_pose and not self.dry_run:
            raise rospy.ROSInitException(
                "~fake_pose requires ~dry_run:=true -- refusing to plan from a "
                "synthetic state while setpoints are live")

        # ------------------------------------------------------- dynamics
        mass = rospy.get_param("~mass", ROGX_MASS)
        inertia = rospy.get_param("~inertia", None)
        J = ROGX_INERTIA if inertia is None else np.asarray(inertia, dtype=float)
        if J.size == 9:
            J = J.reshape(3, 3)

        self.base_dyn = QuadrotorDynamics(
            mass=mass, inertia=J,
            tilt_max=rospy.get_param("~tilt_max", 0.5236),
            omega_max=rospy.get_param("~omega_max", 3.0),
            vel_max=rospy.get_param("~vel_max", 1.5))

        # CTBR rollouts, as in PA-MPPI: u = [collective thrust, body rates].
        # Raw [f, M] sampling is not an option -- integrated open-loop, only
        # 0.9% of samples stay inside the tilt limit over a 4 s horizon.
        # ~model selects the alternative acceleration-input model if wanted.
        self.model = rospy.get_param("~model", "ctbr")     # ctbr | accel
        if self.model == "accel":
            self.dyn = AttitudeStabilizedQuadrotor(
                self.base_dyn,
                k_att=rospy.get_param("~k_att", 5.0),
                k_rate=rospy.get_param("~k_rate", 10.0),
                substeps=rospy.get_param("~substeps", 1))
        else:
            self.dyn = ThrustRateQuadrotor(
                self.base_dyn,
                tau_rate=rospy.get_param("~tau_rate", 0.03))

        self.grid = InflatedGrid(
            occ_thresh=rospy.get_param("~occ_thresh", 50),
            unknown_is_obstacle=rospy.get_param("~unknown_is_obstacle", True))
        if not self.require_map:
            rospy.logwarn("[haa] ~require_map is FALSE -- flying with NO "
                          "obstacle set. Only valid for open-space checks.")
            self.obstacles = _FreeSpace()
        else:
            self.obstacles = self.grid

        self.mppi = HAAMPPI(
            self.dyn, self.obstacles,
            horizon=self.horizon, num_samples=self.num_samples, dt=self.dt,
            lam=rospy.get_param("~lambda", 0.02),
            sigma_thrust=rospy.get_param("~sigma_thrust", 3.0),
            sigma_rate_xy=rospy.get_param("~sigma_rate_xy", 0.35),
            sigma_rate_z=rospy.get_param("~sigma_rate_z", 0.2),
            w_goal=rospy.get_param("~w_goal", 10.0),
            w_terminal=rospy.get_param("~w_terminal", 100.0),
            w_obstacle=rospy.get_param("~w_obstacle", 3.0),
            w_z=rospy.get_param("~w_z", 50.0),
            w_vel=rospy.get_param("~w_vel", 0.5),
            w_smooth=rospy.get_param("~w_smooth", 1.0),
            z_hold=self.z_hold,
            goal_tol=rospy.get_param("~goal_tol", 0.25))

        # ------------------------------------------------------------ state
        self.pose = None          # planner state, in vicon/world
        self.fcu_pose = None      # FCU local ENU, only used for alignment
        self.vel = None
        self.imu = None
        self.plan_X = None
        self.plan_t0 = None
        self.last_sp = None          # last emitted (x, y, z, yaw)
        self.hold_pos = None         # where to hover when we have no plan

        # --------------------------------------------------------- ROS I/O
        rospy.Subscriber(self.pose_topic, PoseStamped,
                         self._pose_cb, queue_size=1)
        rospy.Subscriber("mavros/local_position/pose", PoseStamped,
                         self._fcu_pose_cb, queue_size=1)
        rospy.Subscriber("mavros/local_position/velocity_local", TwistStamped,
                         self._vel_cb, queue_size=1)
        rospy.Subscriber("mavros/imu/data", Imu, self._imu_cb, queue_size=1)
        rospy.Subscriber("/grid_map", OccupancyGrid, self._grid_cb, queue_size=1)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped,
                         self._goal_cb, queue_size=1)

        self.pub_sp = rospy.Publisher("commander/set_pose", PositionTarget,
                                      queue_size=10)
        self.pub_path = rospy.Publisher("~nominal_path", Path, queue_size=1)
        self.pub_viz = rospy.Publisher("~rollouts", MarkerArray, queue_size=1)
        self.pub_goal = rospy.Publisher("~goal_marker", Marker, queue_size=1)
        self.pub_status = rospy.Publisher("~status", String, queue_size=1)

        rospy.loginfo("[haa] m=%.3f kg  J=%s%s", self.dyn.m,
                      np.round(np.diag(self.base_dyn.J), 5).tolist(),
                      "  (J unused under CTBR)" if self.model != "accel" else "")
        rospy.loginfo("[haa] model=%s  hover thrust %.2f N  thrust max %.2f N",
                      self.model, self.dyn.hover_thrust, self.dyn.thrust_max)
        rospy.loginfo("[haa] plan frame=%s  pose=%s", self.viz_frame,
                      self.pose_topic)
        rospy.loginfo("[haa] r_eff = %.3f m  (r_Q %.2f + r_track %.2f + r_perc %.2f)",
                      self.r_eff, self.r_quad, self.r_track, self.r_perc)
        rospy.loginfo("[haa] z_hold=%.2f m  H=%d  K=%d  dt=%.3f",
                      self.z_hold, self.horizon, self.num_samples, self.dt)
        if self.dry_run:
            rospy.logwarn("[haa] DRY RUN -- planning and publishing viz only, "
                          "commander/set_pose will NOT be written")

    # ------------------------------------------------------------ callbacks

    def _pose_cb(self, msg):
        with self.lock:
            self.pose = msg
        self._try_align()

    def _fcu_pose_cb(self, msg):
        with self.lock:
            self.fcu_pose = msg
        self._try_align()

    def _try_align(self):
        """Refresh vicon/world -> FCU ENU from the current matched pose pair."""
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

    def _vel_cb(self, msg):
        with self.lock:
            self.vel = msg

    def _imu_cb(self, msg):
        with self.lock:
            self.imu = msg

    def _grid_cb(self, msg):
        try:
            self.grid.update(msg, self.r_eff)
        except Exception as e:
            rospy.logwarn_throttle(5.0, "[haa] grid update failed: %s", e)

    def _goal_cb(self, msg):
        with self.lock:
            self.goal = np.array([msg.pose.position.x, msg.pose.position.y])
        rospy.loginfo("[haa] new goal: (%.2f, %.2f)", self.goal[0], self.goal[1])

    # ---------------------------------------------------------------- state

    def _current_state(self):
        """Build the 12-vector from the latest mavros messages, or None."""
        with self.lock:
            pose, vel, imu = self.pose, self.vel, self.imu
        if pose is None:
            if self.fake_pose:
                if self.fake_state is None:
                    self.fake_state = state_from_pose(
                        0.0, 0.0, self.z_hold, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                return self.fake_state.copy()
            return None

        p = pose.pose.position
        q = pose.pose.orientation
        rpy = self.attitude.quatToEuler([q.x, q.y, q.z, q.w])

        if vel is not None:
            v = (vel.twist.linear.x, vel.twist.linear.y, vel.twist.linear.z)
        else:
            v = (0.0, 0.0, 0.0)

        if imu is not None:
            om = (imu.angular_velocity.x, imu.angular_velocity.y,
                  imu.angular_velocity.z)
        else:
            om = (0.0, 0.0, 0.0)

        return state_from_pose(p.x, p.y, p.z, v[0], v[1], v[2],
                               rpy[0], rpy[1], rpy[2], om[0], om[1], om[2])

    # ------------------------------------------------------------- planning

    def plan_once(self, _evt=None):
        x0 = self._current_state()
        if x0 is None:
            rospy.logwarn_throttle(2.0, "[haa] waiting for mavros pose")
            return

        if self.hold_pos is None:
            self.hold_pos = (x0[0], x0[1], self.z_hold, x0[8])

        if self.require_map and not self.grid.ready:
            rospy.logwarn_throttle(2.0, "[haa] waiting for /grid_map "
                                        "(set ~require_map:=false to fly open space)")
            return

        with self.lock:
            goal = self.goal.copy()

        t_start = rospy.get_time()
        res = self.mppi.plan(x0, goal, warm_shift=1, n_viz=self.viz_rollouts)
        solve_ms = (rospy.get_time() - t_start) * 1000.0

        if not res.feasible:
            # HAA infeasibility. Hold position; braking/recovery is a separate
            # piece of work, and hovering is the conservative thing to do here.
            rospy.logwarn_throttle(1.0, "[haa] INFEASIBLE (%s) -> holding",
                                   res.reason)
            self.plan_X = None
            self.mppi.reset()
            self._publish_status("INFEASIBLE %s" % res.reason)
            return

        self.plan_X = res.X
        self.plan_t0 = rospy.get_time()

        # Bench mode is a closed loop: step the synthetic vehicle one dt along
        # the plan it just produced. Without this the state never advances, the
        # goal never gets closer, and MPPI winds the nominal up until every
        # sample breaks the velocity limit -- which is exactly what a static
        # fake pose produced before this was added.
        if self.fake_pose and res.X is not None and res.X.shape[0] > 1:
            self.fake_state = res.X[1].copy()
        self._publish_path(res.X)
        self._publish_rollouts(res.X_viz)
        self._publish_goal(goal)
        self._publish_status(
            "OK valid=%d/%d cost=%.1f solve=%.0fms | %s %s"
            % (res.n_valid, res.n_samples, res.best_cost, solve_ms,
               self.align.describe(), res.reason))

        if solve_ms > 1000.0 / self.plan_rate:
            rospy.logwarn_throttle(
                5.0, "[haa] solve %.0f ms exceeds the %.0f ms plan period -- "
                     "lower ~num_samples or ~horizon", solve_ms,
                     1000.0 / self.plan_rate)

    # ------------------------------------------------------------ publishing

    def publish_setpoint(self, _evt=None):
        """Sample the nominal trajectory at the elapsed time and emit it."""
        if self.dry_run:
            return
        x0 = self._current_state()
        if x0 is None:
            return

        target = None
        if self.plan_X is not None and self.plan_t0 is not None:
            age = rospy.get_time() - self.plan_t0
            if age <= self.plan_timeout:
                i = int(np.clip(round(age / self.dt), 1, self.plan_X.shape[0] - 1))
                row = self.plan_X[i]
                yaw = row[8] if self.yaw_mode == "plan" else x0[8]
                target = (row[0], row[1], self.z_hold, yaw)
            else:
                rospy.logwarn_throttle(2.0, "[haa] plan stale (%.2fs) -> holding",
                                       age)

        if target is None:
            target = self.hold_pos if self.hold_pos is not None else \
                (x0[0], x0[1], self.z_hold, x0[8])

        target = self._limit(target, x0)

        # target is in vicon/world; mavros wants the FCU frame. No alignment,
        # no setpoint -- publishing raw world coordinates would command a point
        # metres away from the intended one.
        if not self.align.ready:
            rospy.logwarn_throttle(2.0, "[haa] no world->FCU alignment yet "
                                        "(%d pairs) -- withholding setpoint",
                                   self.align.n_updates)
            return
        if self.align.age(rospy.get_time()) > self.align_timeout:
            rospy.logwarn_throttle(2.0, "[haa] alignment stale (%.1fs) -- "
                                        "withholding setpoint",
                                   self.align.age(rospy.get_time()))
            return

        p_fcu, yaw_fcu = self.align.to_fcu(
            [target[0], target[1], target[2]], target[3])

        self.last_sp = target
        self.pub_sp.publish(
            self.controller.construct_target(p_fcu[0], p_fcu[1],
                                             p_fcu[2], yaw_fcu))

    def _limit(self, target, x0):
        """Clamp how far the setpoint may sit from the vehicle.

        A setpoint far from the current position is a large step command to
        PX4's position controller. This bounds it, so a bad plan or a frame
        mistake cannot turn into a violent lunge.
        """
        x, y, z, yaw = target
        dx, dy = x - x0[0], y - x0[1]
        d = float(np.hypot(dx, dy))
        if d > self.max_step and d > 1e-9:
            s = self.max_step / d
            x, y = x0[0] + dx * s, x0[1] + dy * s
            rospy.logwarn_throttle(2.0, "[haa] setpoint %.2fm away -> clamped "
                                        "to %.2fm", d, self.max_step)
        return (x, y, float(self.z_hold), yaw)

    def _publish_path(self, X):
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self.viz_frame
        for row in X:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(row[0])
            ps.pose.position.y = float(row[1])
            ps.pose.position.z = float(row[2])
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.pub_path.publish(path)

    def _publish_rollouts(self, X_viz):
        """Draw the surviving samples as thin lines and the chosen plan thick.

        Foxglove 3D renders these against the map->odom->base_link TF tree, so
        you can see what MPPI explored, not just what it picked.
        """
        if self.pub_viz.get_num_connections() == 0:
            return

        arr = MarkerArray()
        stamp = rospy.Time.now()

        wipe = Marker()
        wipe.header.frame_id = self.viz_frame
        wipe.header.stamp = stamp
        wipe.ns = "rollouts"
        wipe.action = Marker.DELETEALL
        arr.markers.append(wipe)

        if X_viz is not None:
            for i, Xi in enumerate(X_viz):
                m = Marker()
                m.header.frame_id = self.viz_frame
                m.header.stamp = stamp
                m.ns = "rollouts"
                m.id = i
                m.type = Marker.LINE_STRIP
                m.action = Marker.ADD
                m.scale.x = 0.01
                m.color.r, m.color.g, m.color.b, m.color.a = 0.3, 0.6, 1.0, 0.25
                m.pose.orientation.w = 1.0
                m.points = [_pt(r) for r in Xi]
                m.lifetime = rospy.Duration(0.5)
                arr.markers.append(m)

        if self.plan_X is not None:
            m = Marker()
            m.header.frame_id = self.viz_frame
            m.header.stamp = stamp
            m.ns = "nominal"
            m.id = 0
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.04
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.35, 0.0, 1.0
            m.pose.orientation.w = 1.0
            m.points = [_pt(r) for r in self.plan_X]
            m.lifetime = rospy.Duration(0.5)
            arr.markers.append(m)

        self.pub_viz.publish(arr)

    def _publish_goal(self, goal):
        if self.pub_goal.get_num_connections() == 0:
            return
        m = Marker()
        m.header.frame_id = self.viz_frame
        m.header.stamp = rospy.Time.now()
        m.ns = "goal"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(goal[0])
        m.pose.position.y = float(goal[1])
        m.pose.position.z = float(self.z_hold)
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 2.0 * self.mppi.goal_tol
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 1.0, 0.2, 0.6
        self.pub_goal.publish(m)

    def _publish_status(self, text):
        self.pub_status.publish(String(data=text))

    # -------------------------------------------------------------- run loop

    def start(self):
        if self.fake_pose:
            rospy.logwarn("[haa] ~fake_pose -- planning from a synthetic hover "
                          "state at (0, 0, %.2f), NOT from the vehicle",
                          self.z_hold)
        else:
            rospy.loginfo("[haa] waiting for %s ...", self.pose_topic)
            while not rospy.is_shutdown() and self.pose is None:
                rospy.sleep(0.1)
            rospy.loginfo("[haa] pose acquired")
            if not self.dry_run:
                rospy.loginfo("[haa] waiting for world->FCU alignment ...")
                while not rospy.is_shutdown() and not self.align.ready:
                    rospy.sleep(0.1)
                rospy.loginfo("[haa] %s", self.align.describe())
        rospy.loginfo("[haa] planning at %.1f Hz, publishing at %.1f Hz",
                      self.plan_rate, self.pub_rate)

        rospy.Timer(rospy.Duration(1.0 / self.plan_rate), self.plan_once)
        rospy.Timer(rospy.Duration(1.0 / self.pub_rate), self.publish_setpoint)
        rospy.spin()


if __name__ == "__main__":
    HAAPlannerNode().start()
