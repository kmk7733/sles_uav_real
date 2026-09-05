#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure MPPI solve time on the LIVE /grid_map, without flying anything.

    rosrun offboard_flight bench_planar_mppi.py _goal_x:=0.0 _goal_y:=3.0

WHAT THIS MEASURES, AND WHAT IT DOES NOT
It subscribes to the same /grid_map the planner node consumes, builds the same
PlanarOccupancy, and calls FrontierMPPI.plan() in a loop, timing each call. It
publishes NOTHING -- no setpoints, no mode changes -- so it is safe to run with
the vehicle sitting on the floor and no FCU link.

It does not measure the node's end-to-end tick (callback + publish + TF), only
the solve. The solve is what the "solve N ms over the plan period" warning is
about, so it is the number that decides whether ~num_samples/~horizon fit.

WHY THE START STATE IS A PARAMETER
Solve cost is not state-independent: the geodesic field is rebuilt per solve
over the traversable set, and how much of the arena is traversable depends on
where the vehicle is and what has been observed. With no pose available it
falls back to ~x0/~y0 so the benchmark still runs with the drone powered off.
"""

import sys
import time

import numpy as np
import rospy
from scipy.ndimage import distance_transform_edt

from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker, MarkerArray

from planar_types import PlanarState
from planar_dynamics import PlanarDynamics, PlanarLimits
from planar_map import PlanarOccupancy
from planar_safety import PlanarSafetyValidator, safe_radius
from planar_mppi import PlanarCostWeights
from frontier import FrontierMPPI


def percentile(a, q):
    return float(np.percentile(np.asarray(a, dtype=np.float64), q))


class Bench(object):
    def __init__(self):
        self.grid_topic = rospy.get_param("~grid_topic", "/grid_map")
        self.pose_topic = rospy.get_param("~pose_topic", "/robot/pose_world")
        self.n_solves = int(rospy.get_param("~n_solves", 100))
        self.warmup = int(rospy.get_param("~warmup", 5))

        self.goal = np.array([float(rospy.get_param("~goal_x", 0.0)),
                              float(rospy.get_param("~goal_y", 3.0))])
        self.x0 = float(rospy.get_param("~x0", 0.0))
        self.y0 = float(rospy.get_param("~y0", 0.0))

        self.horizon = int(rospy.get_param("~horizon", 20))
        self.num_samples = int(rospy.get_param("~num_samples", 192))
        self.dt = float(rospy.get_param("~dt", 0.1))
        self.use_geodesic = bool(rospy.get_param("~use_geodesic", True))
        self.w_frontier = float(rospy.get_param("~w_frontier", 0.0))

        self.r_safe = safe_radius(rospy.get_param("~r_quad", 0.31),
                                  rospy.get_param("~r_perc", 0.18),
                                  rospy.get_param("~r_track", 0.05),
                                  rospy.get_param("~d_clr", 0.05))
        # The MAP inflation radius, which is NOT r_safe: r_quad is the
        # vehicle's own disc and is accounted for by the validator's footprint
        # check, not by growing the map. What the map must absorb is the
        # uncertainty in where the obstacle really is:
        #     r_eff = r_perc + r_track + d_clr = r_safe - r_quad = 0.280 m
        # Same name and same value as config.yaml:22 in the simulation repo.
        self.r_quad = float(rospy.get_param("~r_quad", 0.31))
        self.r_eff = float(rospy.get_param("~r_eff",
                                              self.r_safe - self.r_quad))
        self.occ_thresh = int(rospy.get_param("~occ_thresh", 50))
        self.unknown_unsafe = bool(rospy.get_param("~unknown_unsafe", True))

        # ------------------------------------------------------- viz output
        # Published on the SAME topic names and frame the planner node uses, so
        # a Foxglove layout built against one works unchanged against the other.
        self.viz_frame = rospy.get_param("~viz_frame", "vicon/world")
        self.z0 = float(rospy.get_param("~z0", 1.0))
        self.viz_rollouts = int(rospy.get_param("~viz_rollouts", 30))
        self.hold = float(rospy.get_param("~hold", 30.0))

        self.pub_path = rospy.Publisher("~nominal_path", Path, queue_size=1)
        self.pub_viz = rospy.Publisher("~rollouts", MarkerArray, queue_size=1)
        self.pub_goal = rospy.Publisher("~goal_marker", Marker, queue_size=1,
                                        latch=True)
        self.pub_start = rospy.Publisher("~start_marker", Marker, queue_size=1,
                                         latch=True)
        self.pub_inflated = rospy.Publisher("~inflated_obstacles",
                                            OccupancyGrid, queue_size=1)
        # Visualisation-only second inflation, drawn on top of the r_eff band.
        # r_quad by default, so inner + ring adds up to r_safe on screen.
        self.r_viz_expand = float(rospy.get_param("~r_viz_expand", 0.31))
        self.pub_ring = rospy.Publisher("~inflated_outer", OccupancyGrid,
                                        queue_size=1)

        self.occ = None
        self.pose = None
        rospy.Subscriber(self.grid_topic, OccupancyGrid, self._grid_cb,
                         queue_size=1)
        rospy.Subscriber(self.pose_topic, PoseStamped, self._pose_cb,
                         queue_size=1)

        limits = PlanarLimits(
            v_max=rospy.get_param("~v_max", 1.0),
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
            d_influence=rospy.get_param("~d_influence", None),
            w_yaw=rospy.get_param("~w_yaw", 0.0),
            yaw_mode=rospy.get_param("~yaw_mode", "velocity"))

        self.dyn = PlanarDynamics(limits, dt=self.dt)
        self.validator = PlanarSafetyValidator(None, self.r_safe)
        self.planner = FrontierMPPI(
            self.dyn, self.validator, weights=weights,
            use_geodesic=self.use_geodesic, w_frontier=self.w_frontier,
            c_occupied=rospy.get_param("~c_occupied", 2.0),
            c_unknown=rospy.get_param("~c_unknown", -4.0),
            horizon=self.horizon, num_samples=self.num_samples,
            temperature=rospy.get_param("~temperature", 1.0),
            seed=int(rospy.get_param("~seed", 0)),
            goal_tol=rospy.get_param("~goal_tol", 0.25))

    def _grid_cb(self, msg):
        try:
            self.occ = PlanarOccupancy.from_occupancy_grid_msg(
                msg, occ_thresh=self.occ_thresh,
                unknown_unsafe=self.unknown_unsafe)
        except Exception as e:
            rospy.logwarn("[bench] grid parse failed: %s", e)

    def _pose_cb(self, msg):
        self.pose = msg

    # --------------------------------------------------------------- viz

    def _publish_path(self, ref):
        """The accepted nominal plan, lifted to the flight height."""
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self.viz_frame
        for row in ref.p:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(row[0])
            ps.pose.position.y = float(row[1])
            ps.pose.position.z = self.z0
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.pub_path.publish(path)
        return path

    def _publish_rollouts(self, X_viz):
        # Building 30 LINE_STRIPs costs the same whether or not anyone is
        # listening. The planner node guards this the same way.
        if X_viz is None or self.pub_viz.get_num_connections() == 0:
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
            m.points = [Point(float(r[0]), float(r[1]), self.z0) for r in Xi]
            arr.markers.append(m)
        self.pub_viz.publish(arr)

    def _publish_outer_ring(self, occ, d_occ, inner):
        """One more inflation on top of the r_eff band.

        PURELY A PICTURE. Nothing reads this -- not the validator, not the cost,
        not the geodesic field. It exists so the r_quad part of the safety
        radius is visible on screen instead of only living inside the
        validator's arithmetic:

            inner band (r_eff  = 0.28 m)  map uncertainty, already published
            + this ring (r_quad = 0.31 m)  the airframe's own disc
            = r_safe            0.59 m     what the validator actually gates on

        Only the RING is published -- the cells in the outer inflation that are
        not already in the inner one. Publishing the filled outer disc instead
        would cover the inner band and the two would be indistinguishable.

        Same -1 overlay convention as _publish_inflated: everything outside the
        ring is -1 so /grid_map and the inner band stay visible underneath.
        Colour is a per-topic setting in the Foxglove 3D panel (Color mode ->
        custom); it is deliberately not baked into the message.
        """
        if self.pub_ring.get_num_connections() == 0:
            return
        outer = d_occ < (self.r_eff + self.r_viz_expand)
        ring = outer & ~inner

        cells = np.full(occ.occupied.shape, -1, dtype=np.int8)
        cells[ring] = 100

        g = OccupancyGrid()
        g.header.stamp = rospy.Time.now()
        g.header.frame_id = getattr(occ, "frame_id", None) or self.viz_frame
        g.info.resolution = occ.res
        g.info.width = occ.W
        g.info.height = occ.H
        g.info.origin.position.x = occ.origin[0]
        g.info.origin.position.y = occ.origin[1]
        g.info.origin.position.z = 0.0
        g.info.origin.orientation.w = 1.0
        g.data = cells.reshape(-1).tolist()
        self.pub_ring.publish(g)

    def _publish_inflated(self, occ):
        """The obstacle set grown by r_eff -- MEASURED obstacles only.

        WHY NOT occ.clearance()
        clearance() runs its EDT over `unsafe`, which is obstacle OR unknown, so
        inflating with it grows the unknown region too. Early in a flight almost
        everything is unknown, so that view is a nearly solid block and says
        nothing about where the walls are.

        This inflates `occupied` alone, which is what the simulation shows and
        what the question "will the vehicle fit past that wall" actually asks.
        Unknown is NOT grown and is not drawn at all -- see the overlay note
        below.

        WHY r_eff AND NOT r_safe
        r_safe = r_quad + r_perc + r_track + d_clr, and r_quad is the vehicle's
        own disc -- the validator already applies that against the raw map, so
        baking it into the map too would count it twice. Growing by r_safe here
        turned 2686 occupied cells into 11270 of 14976, three quarters of the
        arena. What the MAP should absorb is only the uncertainty about where
        the obstacle actually is: r_perc + r_track + d_clr.

        This is a VIEW, not the gate. PlanarSafetyValidator still refuses to
        enter unknown space, so free-looking cells here can still be rejected --
        that is the intended difference, not a bug.
        """
        if occ is None or not hasattr(occ, "occupied"):
            return
        if (self.pub_inflated.get_num_connections() == 0
                and self.pub_ring.get_num_connections() == 0):
            return                      # the EDT below is the expensive part
        d_occ = distance_transform_edt(~occ.occupied) * occ.res
        inflated = d_occ < self.r_eff
        self._publish_outer_ring(occ, d_occ, inflated)

        # OVERLAY, NOT A LAYER, and only this layer's OWN annulus. Everything
        # else is -1 (unknown), which Foxglove draws as nothing:
        #   * publishing 0 for free cells paints an opaque sheet over the whole
        #     extent and hides /grid_map underneath
        #   * publishing the filled disc covers the measured obstacle itself, so
        #     the thing being inflated disappears under its own inflation
        # Excluding `occupied` leaves the real obstacle visible from /grid_map,
        # exactly as the outer ring excludes this band.
        cells = np.full(occ.occupied.shape, -1, dtype=np.int8)
        cells[inflated & ~occ.occupied] = 100

        g = OccupancyGrid()
        g.header.stamp = rospy.Time.now()
        g.header.frame_id = getattr(occ, "frame_id", None) or self.viz_frame
        g.info.resolution = occ.res
        g.info.width = occ.W
        g.info.height = occ.H
        g.info.origin.position.x = occ.origin[0]
        g.info.origin.position.y = occ.origin[1]
        g.info.origin.position.z = 0.0
        g.info.origin.orientation.w = 1.0
        g.data = cells.reshape(-1).tolist()
        self.pub_inflated.publish(g)

    def _sphere(self, pub, ns, xy, rgba, diameter):
        m = Marker()
        m.header.frame_id = self.viz_frame
        m.header.stamp = rospy.Time.now()
        m.ns = ns
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(xy[0])
        m.pose.position.y = float(xy[1])
        m.pose.position.z = self.z0
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = diameter
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        pub.publish(m)

    def run(self):
        rospy.loginfo("[bench] waiting for %s ...", self.grid_topic)
        t_wait = time.time()
        while self.occ is None and not rospy.is_shutdown():
            if time.time() - t_wait > 20.0:
                rospy.logerr("[bench] no %s in 20 s -- is the grid running?",
                             self.grid_topic)
                return 1
            time.sleep(0.2)

        if self.pose is not None:
            x0 = self.pose.pose.position.x
            y0 = self.pose.pose.position.y
            src = self.pose_topic
        else:
            x0, y0 = self.x0, self.y0
            src = "~x0/~y0 fallback (no pose)"

        print("")
        print("start   (%.2f, %.2f)   from %s" % (x0, y0, src))
        print("goal    (%.2f, %.2f)   frame %s"
              % (self.goal[0], self.goal[1],
                 getattr(self.occ, "frame_id", "?")))
        print("grid    %s" % self.occ.describe())
        print("occupied %d   unknown %d   of %d cells"
              % (int(self.occ.occupied.sum()), int(self.occ.unknown.sum()),
                 self.occ.H * self.occ.W))
        print("planner %s" % self.planner.describe())
        print("r_safe  %.3f m   r_eff %.3f m (= r_safe - r_quad %.2f)"
              % (self.r_safe, self.r_eff, self.r_quad))
        print("        K=%d  N=%d  dt=%.3f"
              % (self.num_samples, self.horizon, self.dt))
        print("")

        state = PlanarState(x=x0, y=y0, psi=0.0, vx=0.0, vy=0.0, omega=0.0)

        last_ref = None
        last_viz = None
        ms, statuses = [], {}
        a_prev = None
        n = self.n_solves + self.warmup
        for i in range(n):
            if rospy.is_shutdown():
                break
            # Freshest map, exactly as the node would see it.
            occ = self.occ
            occ.clear_disc(state.x, state.y, self.r_safe + 0.05)
            self.validator.occ = occ
            self.planner.validator = self.validator

            # Only plan() is timed; publishing is deliberately outside the
            # clock so the number stays comparable to the node's solve warning.
            t0 = time.time()
            res = self.planner.plan(state, self.goal, a_prev=a_prev,
                                    n_viz=self.viz_rollouts)
            dt_ms = (time.time() - t0) * 1e3

            st = str(getattr(res, "status", "?"))
            statuses[st] = statuses.get(st, 0) + 1
            if i >= self.warmup:          # discard JIT/cache warm-up
                ms.append(dt_ms)

            if getattr(res, "ok", False) and res.reference is not None:
                last_ref = res.reference
                last_viz = res.X_viz
                self._publish_path(last_ref)
                self._publish_rollouts(last_viz)
            self._sphere(self.pub_goal, "planar_goal", self.goal,
                         (0.1, 1.0, 0.2, 0.6), 2.0 * self.planner.goal_tol)
            self._sphere(self.pub_start, "planar_start", (state.x, state.y),
                         (1.0, 0.6, 0.1, 0.9), 0.20)
            self._publish_inflated(occ)

        if not ms:
            print("no solves completed")
            return 1

        ms_sorted = sorted(ms)
        print("=== solve time over %d solves (%d warm-up discarded) ==="
              % (len(ms), self.warmup))
        print("  mean   %7.1f ms" % (sum(ms) / len(ms)))
        print("  median %7.1f ms" % percentile(ms_sorted, 50))
        print("  p95    %7.1f ms" % percentile(ms_sorted, 95))
        print("  max    %7.1f ms" % max(ms))
        print("  min    %7.1f ms" % min(ms))
        print("")
        print("  => sustainable rate  %.1f Hz (mean)   %.1f Hz (p95)"
              % (1000.0 / (sum(ms) / len(ms)), 1000.0 / percentile(ms_sorted, 95)))
        budget = 1000.0 / float(rospy.get_param("~plan_rate", 10.0))
        over = sum(1 for v in ms if v > budget)
        print("  => %d/%d solves (%.0f%%) exceed the %.0f ms plan period"
              % (over, len(ms), 100.0 * over / len(ms), budget))
        print("")
        print("status counts: %s" % statuses)

        # The measurement loop finishes in seconds, which is not long enough to
        # find the topics in Foxglove. Keep the last plan on the wire.
        if self.hold > 0.0 and last_ref is not None:
            ns = rospy.get_name()
            print("")
            print("holding %.0f s -- add these in Foxglove (frame %s):"
                  % (self.hold, self.viz_frame))
            print("    %s/nominal_path   nav_msgs/Path        the plan" % ns)
            print("    %s/rollouts       MarkerArray          %d samples"
                  % (ns, self.viz_rollouts))
            print("    %s/goal_marker    Marker               goal" % ns)
            print("    %s/start_marker   Marker               start" % ns)
            print("    %s/inflated_obstacles  OccupancyGrid   grown by r_eff=%.2f m"
                  % (ns, self.r_eff))
            print("    %s/inflated_outer   OccupancyGrid  +%.2f m ring (viz only,"
                  " set its colour in the panel)" % (ns, self.r_viz_expand))
            print("    /grid_map         OccupancyGrid        the raw map")
            t_end = time.time() + self.hold
            while time.time() < t_end and not rospy.is_shutdown():
                self._publish_path(last_ref)
                self._publish_rollouts(last_viz)
                self._sphere(self.pub_goal, "planar_goal", self.goal,
                             (0.1, 1.0, 0.2, 0.6), 2.0 * self.planner.goal_tol)
                self._sphere(self.pub_start, "planar_start", (x0, y0),
                             (1.0, 0.6, 0.1, 0.9), 0.20)
                self._publish_inflated(self.occ)
                time.sleep(0.2)
        return 0


if __name__ == "__main__":
    # NOT anonymous: the topic names must be stable or a saved Foxglove layout
    # stops resolving them between runs.
    rospy.init_node("planar_mppi_bench", disable_signals=True)
    sys.exit(Bench().run())
