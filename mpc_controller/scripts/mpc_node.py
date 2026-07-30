#!/usr/bin/env python3
"""MPC (MPPI) trajectory node.

Consumes the ZED 2D occupancy grid (``nav_msgs/OccupancyGrid``, e.g. ``/grid_map``)
and the vehicle pose, runs the sampling-based MPPI planner ported from simulation,
and publishes the planned trajectory as a ``nav_msgs/Path`` for visualization in
Foxglove / RViz (Display frame = the grid frame, usually ``map``).

This first stage is visualization-only: it does NOT yet command MAVROS. It proves
the perception -> planner -> trajectory loop end to end so the path can be seen in
the 3D map. Control-to-MAVROS is the next stage.

Planning happens in the planar surrogate ``PlanarHolonomic`` (state
[x, y, yaw, vx, vy, w_yaw], control [ax, ay, alpha_yaw]); altitude is held at a
fixed ``z_hold`` purely so the Path renders at flight height in 3D.

Inputs
    ~grid_topic   (nav_msgs/OccupancyGrid)   default /grid_map
    ~pose_topic   (geometry_msgs/PoseStamped) default /mavros/local_position/pose
    ~goal_topic   (geometry_msgs/PoseStamped) default /move_base_simple/goal
                  -> click "2D Goal Pose" / publish a goal in Foxglove/RViz.
Outputs
    ~path_topic   (nav_msgs/Path)            default /mpc/trajectory
    ~goal_marker  (visualization_msgs/Marker) default /mpc/goal_marker
"""
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker

from mpc_controller.mppi import MPPI, MPPIPlanner
from mpc_controller.quadrotor import PlanarHolonomic, planar_state
from mpc_controller.obstacle_map import OccupancyGridMap
from mpc_controller.cost_to_go import compute_cost_to_go


def yaw_from_quat(q):
    """Yaw (Z) from a geometry_msgs Quaternion."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(np.arctan2(siny, cosy))


def quat_from_yaw(yaw):
    """(x, y, z, w) for a pure yaw rotation."""
    return (0.0, 0.0, float(np.sin(yaw * 0.5)), float(np.cos(yaw * 0.5)))


class MPCNode:
    def __init__(self):
        # -- topics ---------------------------------------------------------
        self.grid_topic = rospy.get_param("~grid_topic", "/grid_map")
        self.pose_topic = rospy.get_param("~pose_topic", "/mavros/local_position/pose")
        self.goal_topic = rospy.get_param("~goal_topic", "/move_base_simple/goal")
        self.path_topic = rospy.get_param("~path_topic", "/mpc/trajectory")
        self.marker_topic = rospy.get_param("~goal_marker", "/mpc/goal_marker")

        # -- planner horizon / sampling ------------------------------------
        self.N = int(rospy.get_param("~horizon", 30))
        self.K = int(rospy.get_param("~num_rollouts", 256))
        self.dt = float(rospy.get_param("~mppi_dt", 0.1))
        self.rate_hz = float(rospy.get_param("~plan_rate", 10.0))
        self.sigma = float(rospy.get_param("~sigma", 0.7))
        # With cost_norm the temperature is relative (units of cost std).
        self.temperature = float(rospy.get_param("~temperature", 0.3))
        self.noise_beta = float(rospy.get_param("~noise_beta", 0.8))
        self.mppi_iters = int(rospy.get_param("~mppi_iters", 1))
        self.ctg_period = float(rospy.get_param("~ctg_period", 0.5))  # cost-to-go refresh [s]

        # -- anti-chatter ----------------------------------------------------
        self.cost_norm = bool(rospy.get_param("~cost_norm", True))
        self.consistency_w = float(rospy.get_param("~consistency_w", 200.0))
        self.path_consistency_w = float(rospy.get_param("~path_consistency_w", 100.0))
        self.smoothing_w = float(rospy.get_param("~smoothing_w", 1.0))
        self.revalidate_mean = bool(rospy.get_param("~revalidate_mean", True))
        self.retry_keep_warm = bool(rospy.get_param("~retry_keep_warm", True))
        self.near_obs_soft = bool(rospy.get_param("~near_obs_soft", True))
        self.near_obs_falloff = float(rospy.get_param("~near_obs_falloff", 0.30))
        self.vel_ema_alpha = float(rospy.get_param("~vel_ema_alpha", 0.3))

        # -- limits / weights ----------------------------------------------
        self.a_max = float(rospy.get_param("~a_max", 0.7))
        self.alpha_max = float(rospy.get_param("~alpha_max", 3.0))
        self.v_max = float(rospy.get_param("~v_max", 1.5))       # hard state limit
        self.v_cap = float(rospy.get_param("~v_cap", 0.8))       # soft cruise target
        self.robot_radius = float(rospy.get_param("~robot_radius", 0.31))
        self.safe_margin = float(rospy.get_param("~safe_margin", 0.05))
        self.progress_w = float(rospy.get_param("~progress_w", 10.0))
        self.terminal_w = float(rospy.get_param("~terminal_w", 100.0))
        self.near_obs_w = float(rospy.get_param("~near_obs_w", 3.0))
        self.use_cost_to_go = bool(rospy.get_param("~use_cost_to_go", True))

        # -- grid interpretation / viz -------------------------------------
        self.occ_thresh = int(rospy.get_param("~occ_thresh", 50))
        self.unknown_is_obstacle = bool(rospy.get_param("~unknown_is_obstacle", False))
        self.z_hold = float(rospy.get_param("~z_hold", 1.0))
        self.goal_tol = float(rospy.get_param("~goal_tol", 0.25))
        # Clear a disc of this radius [m] around the current robot position in
        # every incoming grid, so the robot's own footprint (self-noise / unknown
        # / ground seen at takeoff) never blocks the first rollout point.
        self.robot_clear_radius = float(rospy.get_param("~robot_clear_radius", 0.30))

        # -- default goal ---------------------------------------------------
        # If enabled, the node plans toward (goal_x, goal_y) immediately without
        # waiting for a /move_base_simple/goal message. A published goal still
        # overrides it at runtime.
        self.use_default_goal = bool(rospy.get_param("~use_default_goal", True))
        self.default_goal = (float(rospy.get_param("~goal_x", 3.0)),
                             float(rospy.get_param("~goal_y", 0.0)))

        # -- state ----------------------------------------------------------
        self.occ = None            # OccupancyGridMap
        self.grid_frame = "map"
        self.planner = None
        self.pose = None           # (x, y, yaw)
        self.vel = np.zeros(3, dtype=np.float32)   # (vx, vy, w_yaw)
        self._last_pose = None     # (x, y, yaw, stamp) for finite-diff velocity
        self.goal = None           # (gx, gy)
        self._ctg_goal = None      # goal the cost-to-go field was last built for
        self._ctg_time = -1e9      # wall time of last cost-to-go recompute

        # -- pubs / subs ----------------------------------------------------
        self.pub_path = rospy.Publisher(self.path_topic, Path, queue_size=1)
        # latched: the marker is published once per goal change, and late
        # subscribers (Foxglove connecting after startup) must still get it
        self.pub_marker = rospy.Publisher(self.marker_topic, Marker,
                                          queue_size=1, latch=True)
        rospy.Subscriber(self.grid_topic, OccupancyGrid, self._grid_cb, queue_size=1)
        rospy.Subscriber(self.pose_topic, PoseStamped, self._pose_cb, queue_size=1)
        rospy.Subscriber(self.goal_topic, PoseStamped, self._goal_cb, queue_size=1)

        if self.use_default_goal:
            self.goal = self.default_goal
            self._publish_goal_marker()

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self._plan_cb)
        rospy.loginfo("[mpc] up. grid=%s pose=%s goal=%s -> path=%s",
                      self.grid_topic, self.pose_topic, self.goal_topic, self.path_topic)
        if self.use_default_goal:
            rospy.loginfo("[mpc] default goal (%.2f, %.2f); publish %s to override",
                          self.default_goal[0], self.default_goal[1], self.goal_topic)
        else:
            rospy.loginfo("[mpc] waiting for grid, pose, and a goal (publish %s)",
                          self.goal_topic)

    # -- planner construction ----------------------------------------------

    def _build_planner(self):
        planar = PlanarHolonomic()
        channel_scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        mppi = MPPI(num_nodes=self.N, num_rollouts=self.K,
                    channel_scale=channel_scale,
                    sigma=self.sigma, temperature=self.temperature,
                    use_noise_ramp=False, noise_beta=self.noise_beta,
                    cost_norm=self.cost_norm)
        ctrl_lo = np.array([-self.a_max, -self.a_max, -self.alpha_max], dtype=np.float32)
        ctrl_hi = np.array([+self.a_max, +self.a_max, +self.alpha_max], dtype=np.float32)
        # State limits: only cap velocities (catastrophic bound); position is free.
        state_lims = [None, None, None,
                      (-self.v_max, self.v_max), (-self.v_max, self.v_max),
                      (-3.0, 3.0)]
        planner = MPPIPlanner(
            mppi, planar, self.occ,
            ctrl_lo=ctrl_lo, ctrl_hi=ctrl_hi,
            robot_radius=self.robot_radius, safe_margin=self.safe_margin,
            state_lims=state_lims,
            progress_w=self.progress_w, terminal_w=self.terminal_w,
            near_obs_w=self.near_obs_w, z_track_w=0.0,
            v_cap=self.v_cap, v_cap_w=50.0,
            smoothing_w=self.smoothing_w, dt=self.dt, mppi_iters=self.mppi_iters,
            consistency_w=self.consistency_w,
            path_consistency_w=self.path_consistency_w,
            near_obs_soft=self.near_obs_soft,
            near_obs_falloff=self.near_obs_falloff,
            revalidate_mean=self.revalidate_mean,
            retry_keep_warm=self.retry_keep_warm)
        return planner

    # -- callbacks ----------------------------------------------------------

    def _grid_cb(self, msg):
        occ = OccupancyGridMap.from_msg(
            msg, occ_thresh=self.occ_thresh,
            unknown_is_obstacle=self.unknown_is_obstacle)
        # Clear the robot's own footprint so it never reads as an obstacle.
        if self.robot_clear_radius > 0.0 and self.pose is not None:
            occ.clear_disc(self.pose[0], self.pose[1], self.robot_clear_radius)
        self.occ = occ
        prev_frame = self.grid_frame
        self.grid_frame = msg.header.frame_id or "map"
        if self.grid_frame != prev_frame and self.goal is not None:
            # the startup marker was latched before the first grid arrived,
            # stamped with the placeholder frame -- restamp it correctly
            self._publish_goal_marker()
        if self.planner is None:
            self.planner = self._build_planner()
        else:
            self.planner.update_obstacle_map(occ)
        # cost-to-go refreshes on its own throttle (ctg_period); no reset here.

    def _pose_cb(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        yaw = yaw_from_quat(msg.pose.orientation)
        self.pose = (x, y, yaw)
        # Finite-difference velocity (no velocity topic assumed for stage 1).
        t = msg.header.stamp.to_sec() if msg.header.stamp else rospy.get_time()
        if self._last_pose is not None:
            px, py, pyaw, pt = self._last_pose
            d = t - pt
            if d > 1e-3:
                vx = (x - px) / d
                vy = (y - py) / d
                dyaw = np.arctan2(np.sin(yaw - pyaw), np.cos(yaw - pyaw))
                a = self.vel_ema_alpha  # EMA to tame differentiation noise
                self.vel = (1 - a) * self.vel + a * np.array(
                    [vx, vy, dyaw / d], dtype=np.float32)
        self._last_pose = (x, y, yaw, t)

    def _goal_cb(self, msg):
        new_goal = (msg.pose.position.x, msg.pose.position.y)
        # A materially different goal invalidates the previous plan as a
        # hysteresis anchor -- reset the warm start so the consistency cost
        # never fights the new objective.
        if (self.goal is not None and self.planner is not None and
                np.hypot(new_goal[0] - self.goal[0],
                         new_goal[1] - self.goal[1]) > self.goal_tol):
            self.planner.reset_warm_start()
        self.goal = new_goal
        rospy.loginfo("[mpc] new goal: (%.2f, %.2f)", self.goal[0], self.goal[1])
        self._publish_goal_marker()

    # -- planning tick ------------------------------------------------------

    def _plan_cb(self, _evt):
        if self.planner is None or self.pose is None or self.goal is None:
            return
        x, y, yaw = self.pose
        gx, gy = self.goal
        if np.hypot(gx - x, gy - y) < self.goal_tol:
            return  # reached; nothing to plan

        x0 = planar_state(np.array([x, y], dtype=np.float32), yaw=yaw)
        x0[3:6] = self.vel
        goal = np.array([gx, gy], dtype=np.float32)

        # (Re)compute geodesic cost-to-go. This is a ~35 ms pure-Python Dijkstra,
        # so we do NOT rebuild it every tick: recompute only when the goal moves
        # or at most every ``ctg_period`` seconds (the grid refreshes obstacles at
        # ~10 Hz, but re-routing that fast is unnecessary and dominates runtime).
        if self.use_cost_to_go:
            now = rospy.get_time()
            goal_moved = (self._ctg_goal is None or
                          np.hypot(gx - self._ctg_goal[0], gy - self._ctg_goal[1]) > 0.10)
            stale = (now - self._ctg_time) >= self.ctg_period
            if goal_moved or stale:
                try:
                    field = compute_cost_to_go(self.occ, goal, self.robot_radius)
                    self.planner.set_cost_to_go(field)
                    self._ctg_goal = (gx, gy)
                    self._ctg_time = now
                except Exception as e:  # keep planning on Euclidean if CTG fails
                    rospy.logwarn_throttle(5.0, "[mpc] cost_to_go failed: %s", e)
                    self.planner.set_cost_to_go(None)

        X_opt, U_opt = self.planner.plan(x0, goal, warm_shift=1)
        if X_opt is None:
            rospy.logwarn_throttle(2.0, "[mpc] no valid trajectory (blocked?)")
            return
        if self.planner.last_mean_fallback:
            rospy.logwarn_throttle(
                2.0, "[mpc] mean plan infeasible -> best-rollout fallback "
                "(%d total)", self.planner.n_mean_fallbacks)
        self._publish_path(X_opt)

    # -- publishers ---------------------------------------------------------

    def _publish_path(self, X_opt):
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self.grid_frame
        for row in X_opt:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(row[0])
            ps.pose.position.y = float(row[1])
            ps.pose.position.z = self.z_hold
            qx, qy, qz, qw = quat_from_yaw(float(row[2]))
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            path.poses.append(ps)
        self.pub_path.publish(path)

    def _publish_goal_marker(self):
        if self.goal is None:
            return
        m = Marker()
        m.header.frame_id = self.grid_frame
        m.header.stamp = rospy.Time.now()
        m.ns = "mpc_goal"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position = Point(self.goal[0], self.goal[1], self.z_hold)
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.3
        m.color.r = 1.0
        m.color.g = 0.2
        m.color.b = 0.2
        m.color.a = 1.0
        self.pub_marker.publish(m)


def main():
    rospy.init_node("mpc_node")
    MPCNode()
    rospy.spin()


if __name__ == "__main__":
    main()
