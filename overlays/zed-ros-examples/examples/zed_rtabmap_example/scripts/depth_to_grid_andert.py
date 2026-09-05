#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""depth_to_grid_andert.py -- 2D occupancy grid from the ZED depth image, by
Andert's inverse measurement model (IROS 2009, eqs. 8-11).

Publishes the SAME contract as `depth_to_grid.py` -- nav_msgs/OccupancyGrid,
-1 unknown / 0 free / 100 occupied, 144x104 @ 0.05 m in `vicon/world` -- so it
is a drop-in for `planar_planner_node.py`, which needs no change. The two nodes
can run side by side on different topics; `mapper:=both` in the launch file.

WHAT IS DIFFERENT FROM depth_to_grid.py, AND WHY
    input        the depth IMAGE + camera_info, not the point cloud. The
                 inverse model is parameterised by image geometry -- one ray
                 per pixel, sigma from disparity -- and a cloud has thrown that
                 away. It is also 0.92 MB/frame against 7.4 MB/frame.
    evidence     a range-dependent profile, not two constants. Stereo has
                 sigma_Z ~ Z^2, so a measurement at 6 m must vote less than one
                 at 1 m; l_occ = 0.85 regardless of range is the thing this
                 node exists to stop doing.
    free space   an exact Amanatides-Woo cell walk, not 16 jittered samples
                 along every ninth ray. At 6 m those samples are 0.37 m apart
                 against 0.05 m cells, so seven cells in eight are never
                 touched and convergence relies on the jitter averaging over
                 frames -- which it cannot while the vehicle is moving.
    combination  per-frame MAXIMUM over pixels (eq. 11, "obstacle priority"),
                 not a sum. A hundred pixels on one wall are one observation of
                 that wall, not a hundred.
    behind       a surface's far side gets P = 0.5, i.e. log-odds 0, i.e. NO
                 EVIDENCE -- neither free nor occupied.
    unknown      `-1` means "no ray has ever touched this cell", and nothing
                 else. The node this replaces published `-1` for everything in
                 (-0.4, +0.4), so a cell observed a hundred times but ambiguous
                 was indistinguishable from one the camera never pointed at.
                 Downstream both are unsafe, so the distinction decides whether
                 ambiguity is flyable.
    pose         the transform at the DEPTH FRAME'S OWN STAMP, not the latest.
                 At 1 m/s and ~100 ms of pipeline latency, `Time(0)` displaces
                 every ray by 10 cm -- two cells, in the direction of travel.

THE MODEL IS NOT IMPLEMENTED HERE. `perception/inverse_sensor.py` is a
byte-identical copy of the simulator's, whose tests pin eqs. 8-11 and whose
`docs/HAA_MPPI.md` §3a justifies p_occ, p_min and eta against the paper and
against measurement. This file is the ROS shell around it: subscribe, look up a
pose, hand over an image, publish. See perception/VENDOR.md.

Subscribes:  ~depth_in  (sensor_msgs/Image, 32FC1, .../depth/depth_registered)
             ~info_in   (sensor_msgs/CameraInfo, .../depth/camera_info)
Publishes:   ~grid_out  (nav_msgs/OccupancyGrid, in <world_frame>)
             ~wedge_out (visualization_msgs/Marker, the FOV wedge -- see below)
             ~seeded    (std_msgs/Bool, latched)
"""
import math
import os
import sys
import time

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool
from tf2_msgs.msg import TFMessage
from tf.transformations import quaternion_matrix
from visualization_msgs.msg import Marker, MarkerArray

# `perception/` is a plain package under catkin_ws/src, reached the same way
# `planner/` is -- no package.xml, nothing to build. Found by walking up rather
# than by a fixed number of dirnames, because this file also exists in the
# `overlays/` mirror at a different depth and a hard-coded count would resolve
# to the wrong tree there without saying so.
def _find_src(start):
    d = os.path.dirname(os.path.abspath(start))
    for _ in range(8):
        if os.path.isfile(os.path.join(d, "perception", "__init__.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise ImportError(
        "cannot find perception/ above %s -- it is deployed to "
        "catkin_ws/src/perception (see its VENDOR.md)" % start)


_SRC = _find_src(__file__)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from perception import verify as verify_provenance
from perception.fuse import (band_rows_per_col, decode_depth, fuse_frame,
                             ray_spacing_limit, stereo_from_camera_info)
from perception.grid import AndertGrid
from perception.inverse_sensor import ProfileParams, validate_profile_params


class RateClock(object):
    """Fires at a fixed rate in MESSAGE time, without drift and without catch-up.

    `perception/pipeline.py:RateClock` in the simulator, same semantics. The
    next slot is `t_next += dt` from the previous DUE time, never from the fire
    time -- resetting to the arrival stamp makes a 10 Hz gate fed at 15 Hz run
    at something slightly slower and nobody notices. Missed slots are counted
    and never fired retroactively: a 300 ms-old obstacle position is worse than
    none.
    """

    def __init__(self, hz):
        self.dt = 1.0 / float(hz) if hz and hz > 0 else 0.0
        self.t_next = None
        self.n_fired = 0
        self.n_skipped = 0

    def due(self, t):
        if self.dt <= 0.0:
            self.n_fired += 1
            return True
        if self.t_next is None:
            self.t_next = t + self.dt
            self.n_fired += 1
            return True
        if t < self.t_next:
            return False
        missed = int((t - self.t_next) / self.dt)
        self.n_skipped += missed
        self.t_next += (missed + 1) * self.dt
        self.n_fired += 1
        return True


class DepthToGridAndert(object):

    def __init__(self):
        # ------------------------------------------------------------ frames
        self.world_frame = rospy.get_param('~world_frame', 'map')
        self.world_frame_tf = self.world_frame.lstrip('/')   # tf2 rejects '/'
        self.map_frame = rospy.get_param('~map_frame', 'map').lstrip('/')
        self.tf_timeout = float(rospy.get_param('~tf_timeout', 0.03))
        self.tf_max_age = float(rospy.get_param('~tf_max_age', 0.25))
        # `vicon_map_align.py` computes world->map ONCE and re-broadcasts the
        # same constant on /tf at 10 Hz. Latching it keeps a 10 Hz publisher off
        # the critical path of a 15 Hz consumer without changing the answer.
        self.latch_world_map = bool(rospy.get_param('~latch_world_map', True))
        self._M_wm = None
        # WHERE THE POSE COMES FROM, AND WHY IT IS NOT tf2 BY DEFAULT.
        # The mapper needs one thing per frame: the camera's world pose at that
        # frame's stamp. `tf2_ros.TransformListener` will provide it, and on
        # this vehicle it costs more than the mapping does -- it subscribes to
        # /tf, which runs at ~458 Hz here, and deserialises every message in
        # Python inside this process, holding the GIL against the fusion.
        # MEASURED: identical fusion work is 66 ms in a process without that
        # listener and 151 ms inside the node with it.
        #
        # Nothing it is being asked for needs a live listener:
        #     vicon/world -> map     constant, latched once by vicon_map_align
        #     map -> base_link       /robot/pose_world, 30 Hz, ALREADY in
        #                            vicon/world (pose_to_world.py)
        #     base_link -> optical   /tf_static, three latched messages
        # `perception/check_pose_source.py` compares the two paths against each
        # other: 60 samples agreed to 0.0000 m and 0.014 deg, and the static
        # chain it recovers is the mount transform `bag_grid_map` hard-codes.
        self.pose_source = str(rospy.get_param('~pose_source', 'topic'))
        self.pose_topic = str(rospy.get_param('~pose_topic',
                                              '/robot/pose_world'))
        self.pose_frame = str(rospy.get_param('~pose_frame',
                                              'base_link')).lstrip('/')
        self.pose_max_dt = float(rospy.get_param('~pose_max_dt', 0.05))
        self._T_bo = None          # pose_frame -> camera optical, static
        self._depth_frame = None
        self._pose_t = []          # ring buffer of stamps
        self._pose_M = []          # and their 4x4 world <- pose_frame
        self.n_dropped_pose = 0
        self._pose_dt = []
        self.require_optical = not bool(
            rospy.get_param('~allow_non_optical_frame', False))

        # -------------------------------------------------------------- grid
        self.res = float(rospy.get_param('~resolution', 0.05))
        self.min_x = float(rospy.get_param('~grid_min_x', -5.0))
        self.max_x = float(rospy.get_param('~grid_max_x', 5.0))
        self.min_y = float(rospy.get_param('~grid_min_y', -2.0))
        self.max_y = float(rospy.get_param('~grid_max_y', 10.0))
        self.wall_enable = bool(rospy.get_param('~border_wall_enable', True))
        self.wall_t = float(rospy.get_param('~border_wall_thickness', 0.1))

        # Identical geometry to depth_to_grid.py, wall growth included, so
        # /grid_map stays a drop-in and the two can be diffed cell for cell.
        wall_c = int(np.ceil(self.wall_t / self.res)) if self.wall_enable else 0
        self.wall_cells = max(wall_c, 0)
        if self.wall_cells:
            self.min_x -= self.wall_cells * self.res
            self.max_x += self.wall_cells * self.res
            self.min_y -= self.wall_cells * self.res
            self.max_y += self.wall_cells * self.res
        nx = int(round((self.max_x - self.min_x) / self.res))
        ny = int(round((self.max_y - self.min_y) / self.res))

        # ------------------------------------------------------------- model
        self.plane_height = float(rospy.get_param('~plane_height', 1.0))
        self.plane_tol = float(rospy.get_param('~plane_tol', 0.25))
        self.decim = max(1, int(rospy.get_param('~decim', 2)))
        # COLUMNS ARE BEARINGS AND ROWS ARE NOT. The horizontal direction of a
        # ray depends on the image column alone, so `decim` sets the map's
        # angular resolution; rows only decide which rays pass the height-band
        # test, and every surviving row in a column redraws the same line into
        # the same cells. Rows can therefore be thinned much harder than
        # columns, which cuts the per-pixel back-projection -- the part of the
        # frame cost that `rays_per_col` does NOT touch. 0 means "same as
        # decim", i.e. the isotropic behaviour.
        rd = int(rospy.get_param('~row_decim', 0))
        self.row_decim = self.decim if rd <= 0 else max(1, rd)
        self.rays_per_col = int(rospy.get_param('~rays_per_col', 8))
        self.z_min = float(rospy.get_param('~z_min', 0.4))
        self.z_max = float(rospy.get_param('~z_max', 6.0))
        self.baseline_m = float(rospy.get_param('~baseline_m', 0.12))
        sd = rospy.get_param('~sigma_disp_px', None)
        self.sigma_disp_px = None if sd is None else float(sd)
        self.profile = ProfileParams(
            p_min=float(rospy.get_param('~p_min', 0.35)),
            eta=float(rospy.get_param('~eta', 0.025)))
        self.map_hz = float(rospy.get_param('~map_hz', 10.0))
        self.publish_every_n = max(1, int(rospy.get_param('~publish_every_n', 1)))

        self.grid = AndertGrid(
            nx, ny, self.res, self.min_x, self.min_y,
            log_odds_clip=float(rospy.get_param('~log_odds_clip', 10.0)),
            p_occ=float(rospy.get_param('~p_occ', 0.5)))

        self._wall = None
        if self.wall_cells:
            w = self.wall_cells
            self._wall = np.zeros((ny, nx), dtype=bool)
            self._wall[:w, :] = True
            self._wall[-w:, :] = True
            self._wall[:, :w] = True
            self._wall[:, -w:] = True
            self.grid.L[self._wall] = self.grid.l_clip
            # ASSERTED, so it must also count as observed -- otherwise the wall
            # publishes as -1, which downstream is unsafe for a different
            # reason and hides the fact that it is a prior and not a
            # measurement.
            self.grid.assert_seen(self._wall)

        # ------------------------------------------ the one asserted free disc
        # ONE SHOT, unlike depth_to_grid.py's per-frame forced-free disc. The
        # camera cannot see around itself, so without an initial assertion the
        # vehicle is boxed in by unknown, the first solve fails, and it never
        # moves or observes anything. But re-asserting it every frame is not an
        # observation: it paints a disc of free space travelling with the
        # vehicle over whatever the camera actually measured. The honest claim
        # is made once -- "the vehicle is physically here, so this is not a
        # wall" -- and everything after it is measurement.
        self.seed_radius = float(rospy.get_param('~initial_free_radius', 1.0))
        self.seed_on = str(rospy.get_param('~seed_on', 'first_pose'))
        self.seed_alt = float(rospy.get_param('~seed_alt', 0.5))
        sp = rospy.get_param('~seed_pose', None)
        self.seed_pose = None if sp is None else (float(sp[0]), float(sp[1]))

        # ------------------------------------------------------------- state
        self.stereo = None
        self.clock = RateClock(self.map_hz)
        self.n_in = 0
        self.n_fused = 0
        self.n_dropped_tf = 0
        self.n_dropped_stale = 0
        self.n_dropped_rate = 0
        self.n_published = 0
        self._ms = []
        self._t_log = 0.0
        self.log_period = float(rospy.get_param('~log_period', 5.0))
        self.publish_wedge = bool(rospy.get_param('~publish_wedge', True))
        # The airframe disc, drawn on the VEHICLE rather than on the camera.
        # r_quad = 0.31 m (arm 0.21 + rotor 0.10) is what the validator's
        # r_safe is built on top of, and it is the one radius you cannot read
        # off the map: the grid is never inflated by it -- `clearance()` returns
        # a distance and the caller compares it against r_safe. Drawing it makes
        # "would this pose fit" answerable by eye.
        self.publish_footprint = bool(rospy.get_param('~publish_footprint',
                                                      True))
        # WHICH HEIGHT TO DRAW IT AT. The three planes in this scene are not
        # the same one and the difference is a metre: the vehicle sits at its
        # own z (0.14 m on the ground, 1.0 m in the air), /grid_map renders at
        # z = 0 because its origin says so, and the mapped SLICE is at
        # plane_height. A ring drawn on the slice floats a metre above both the
        # drone and the map while it is on the ground, which looks like a
        # missing marker rather than a misplaced one.
        #   vehicle (default) -- on the drone, physically true in 3D
        #   grid              -- on the occupancy grid, easiest to read against
        #   slice             -- on the mapped plane
        #   <a number>        -- that height
        self.footprint_z = str(rospy.get_param('~footprint_z', 'vehicle'))
        self.footprint_radii = [
            float(v) for v in str(
                rospy.get_param('~footprint_radii', '0.31')).replace(
                    ',', ' ').split() if v]
        self._last_base = None

        if self.pose_source == 'tf':
            self.tf_buf = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
            self.tf_lis = tf2_ros.TransformListener(self.tf_buf)
        else:
            self.tf_buf = self.tf_lis = None
            # /tf_static is latched and three messages long; read it once and
            # let the subscriber go.
            self._static_links = {}
            self._sub_static = rospy.Subscriber('/tf_static', TFMessage,
                                                self.cb_static, queue_size=50)
            self.sub_pose = rospy.Subscriber(self.pose_topic, PoseStamped,
                                             self.cb_pose, queue_size=50)
        self.pub = rospy.Publisher('~grid_out', OccupancyGrid, queue_size=1)
        self.pub_wedge = rospy.Publisher('~wedge_out', Marker, queue_size=1)
        self.pub_foot = rospy.Publisher('~footprint_out', MarkerArray,
                                        queue_size=1)
        self.pub_seeded = rospy.Publisher('~seeded', Bool, queue_size=1,
                                          latch=True)
        self.pub_seeded.publish(Bool(data=False))

        self.sub_info = rospy.Subscriber('~info_in', CameraInfo, self.cb_info,
                                         queue_size=1)
        self.sub = rospy.Subscriber('~depth_in', Image, self.cb_depth,
                                    queue_size=1, buff_size=2 ** 22)

        try:
            verify_provenance(quiet=True)
            prov = "vendored model verified"
        except RuntimeError as exc:
            rospy.logerr("perception provenance: %s", exc)
            prov = "PROVENANCE MISMATCH"
        rospy.loginfo(
            "depth_to_grid_andert: %dx%d cells @ %.3fm, x[%.2f,%.2f] "
            "y[%.2f,%.2f], world=%s, slice %.2f +- %.2f m, decim col/row "
            "%s, rays/col %d, map %.1f Hz  [%s]",
            nx, ny, self.res, self.min_x, self.max_x, self.min_y, self.max_y,
            self.world_frame, self.plane_height, self.plane_tol,
            "%d/%d" % (self.decim, self.row_decim),
            self.rays_per_col, self.map_hz, prov)

    # ------------------------------------------------------------ intrinsics

    def cb_info(self, msg):
        """Build StereoParams from the camera's OWN calibration, once.

        The intrinsics cannot change without restarting the wrapper, which
        restarts this node, so pairing them per-frame with message_filters buys
        nothing. They are logged loudly instead, because THE SIMULATOR'S
        CONSTANTS ARE WRONG FOR THIS CAMERA: NATIVE_F = 350 at 640x360 is an
        84.9 deg HFOV and this ZED 2i is 101.5 deg. Anyone reading the log
        should be able to see which one is in force.
        """
        if self.stereo is not None:
            return
        st = stereo_from_camera_info(
            msg.K, msg.width, msg.height, decim=self.decim,
            row_decim=self.row_decim,
            baseline_m=self.baseline_m, sigma_disp_px=self.sigma_disp_px,
            z_min_m=self.z_min, z_max_m=self.z_max)
        try:
            sigma_l_min = validate_profile_params(self.profile, st, self.res)
        except ValueError as exc:
            rospy.logfatal("inverse model parameters are not a probability at "
                           "these intrinsics: %s", exc)
            rospy.signal_shutdown("bad inverse model parameters")
            return
        hfov = math.degrees(2.0 * math.atan(0.5 * st.width / st.fx))
        lim = ray_spacing_limit(self.res, st.fx)
        rospy.loginfo(
            "camera_info(%s): %dx%d fx %.2f fy %.2f cx %.2f cy %.2f  "
            "HFOV %.1f deg  B %.3f m  sigma_disp %.4f px  range [%.2f, %.2f] m"
            "  sigma_l floor %.4f m",
            msg.header.frame_id, st.width, st.height, st.fx, st.fy, st.cx,
            st.cy, hfov, st.baseline_m, st.sigma_disp_px, st.z_min_m,
            st.z_max_m, sigma_l_min)
        if self.rays_per_col > 0:
            # The row stride is only free while the band still offers at least
            # rays_per_col rows to choose from. Past that the cap stops binding
            # and the stride starts costing rays rather than time -- and it
            # does so silently, because the map simply fills in a little less.
            for z in (2.0, 4.0, 6.0):
                n = band_rows_per_col(st.fy, self.plane_tol, z)
                if n < self.rays_per_col:
                    rospy.logwarn(
                        "at %.0f m the height band keeps %.1f rows per column, "
                        "below rays_per_col %d: the row stride %d is costing "
                        "rays beyond that range, not just time",
                        z, n, self.rays_per_col, self.row_decim)
                    break
        if self.z_max > lim:
            # Ray spacing grows with range; past res*fx adjacent rays skip
            # cells and a far wall comes back with holes in it, which reads as
            # a gap rather than as distance.
            rospy.logwarn(
                "z_max %.1f m exceeds the ray-spacing limit res*fx = %.1f m at "
                "decim %d -- surfaces beyond %.1f m will be perforated",
                self.z_max, lim, self.decim, lim)
        self.stereo = st
        self.sub_info.unregister()

    # ------------------------------------------------------------------ pose

    def cb_static(self, msg):
        """Collect /tf_static once and build pose_frame -> camera optical.

        A depth-first parent->child walk, not tf2: the ZED chain is a simple
        path (base_link -> zed2i_base_link -> camera_center -> left_camera_frame
        -> left_camera_optical_frame) and resolving it costs three latched
        messages instead of a listener on a 458 Hz topic.
        """
        for tr in msg.transforms:
            par = tr.header.frame_id.lstrip('/')
            chi = tr.child_frame_id.lstrip('/')
            t, r = tr.transform.translation, tr.transform.rotation
            M = quaternion_matrix([r.x, r.y, r.z, r.w])
            M[:3, 3] = [t.x, t.y, t.z]
            self._static_links[(par, chi)] = M
        if self.stereo is None or self._T_bo is not None:
            return
        self._resolve_static()

    def _resolve_static(self):
        target = self._depth_frame
        if target is None:
            return
        stack, seen = [(self.pose_frame, np.eye(4))], {self.pose_frame}
        while stack:
            node, M = stack.pop()
            if node == target:
                self._T_bo = M
                rospy.loginfo("static %s <- %s from /tf_static: t %s",
                              target, self.pose_frame,
                              np.array2string(M[:3, 3], precision=4))
                self._sub_static.unregister()
                return
            for (par, chi), T in self._static_links.items():
                if par == node and chi not in seen:
                    seen.add(chi)
                    stack.append((chi, M.dot(T)))

    def cb_pose(self, msg):
        """Ring-buffer the vehicle pose, already in the world frame."""
        got = msg.header.frame_id.lstrip('/')
        if got and got != self.world_frame_tf:
            rospy.logwarn_throttle(
                10.0, "%s is stamped %r, not the map's %r -- the pose and the "
                "grid are in different frames", self.pose_topic, got,
                self.world_frame_tf)
        p_, q_ = msg.pose.position, msg.pose.orientation
        M = quaternion_matrix([q_.x, q_.y, q_.z, q_.w])
        M[:3, 3] = [p_.x, p_.y, p_.z]
        self._pose_t.append(msg.header.stamp.to_sec())
        self._pose_M.append(M)
        if len(self._pose_t) > 128:          # ~4 s at 30 Hz
            del self._pose_t[:64]
            del self._pose_M[:64]

    def lookup_topic(self, stamp):
        """world <- camera optical, from the pose topic and the static chain.

        Nearest stamp, not interpolation: the pose runs at 30 Hz against a
        15 Hz depth stream, so the worst pairing error is half a pose period,
        17 ms -- 1.7 cm at 1 m/s, a third of a cell. `~pose_max_dt` drops
        anything worse rather than fusing it, and the ages are reported so the
        assumption is on the record instead of assumed.
        """
        if self._T_bo is None or not self._pose_t:
            return None
        t = stamp.to_sec()
        ts = np.asarray(self._pose_t)
        i = int(np.argmin(np.abs(ts - t)))
        dt = abs(ts[i] - t)
        if dt > self.pose_max_dt:
            rospy.logwarn_throttle(3.0, "nearest pose is %.0f ms from the "
                                   "frame; dropping", dt * 1e3)
            return None
        self._pose_dt.append(dt)
        self._last_base = self._pose_M[i][:3, 3]
        M = self._pose_M[i].dot(self._T_bo)
        return M[:3, 3], M[:3, :3]

    def lookup(self, frame, stamp):
        """world <- camera optical, AT THE FRAME'S OWN STAMP.

        Returns (t_vec, R_3x3) or None. Never falls back to Time(0): a pose
        from the wrong instant is not a degraded measurement, it is a
        measurement of somewhere else, and it looks perfectly plausible.
        """
        try:
            if self.latch_world_map and self.world_frame_tf != self.map_frame:
                if self._M_wm is None:
                    tw = self.tf_buf.lookup_transform(
                        self.world_frame_tf, self.map_frame, rospy.Time(0),
                        rospy.Duration(self.tf_timeout))
                    self._M_wm = _mat(tw)
                    rospy.loginfo("latched %s <- %s (constant, re-broadcast at "
                                  "10 Hz by vicon_map_align)",
                                  self.world_frame_tf, self.map_frame)
                tm = self.tf_buf.lookup_transform(
                    self.map_frame, frame, stamp,
                    rospy.Duration(self.tf_timeout))
                M = self._M_wm.dot(_mat(tm))
            else:
                tw = self.tf_buf.lookup_transform(
                    self.world_frame_tf, frame, stamp,
                    rospy.Duration(self.tf_timeout))
                M = _mat(tw)
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException) as exc:
            rospy.logwarn_throttle(3.0, "tf %s <- %s at %.3f: %s",
                                   self.world_frame_tf, frame,
                                   stamp.to_sec(), exc)
            return None
        return M[:3, 3], M[:3, :3]

    # ------------------------------------------------------------------ main

    def cb_depth(self, msg):
        self.n_in += 1
        if self.stereo is None:
            return                              # camera_info not in yet

        if self.require_optical and not msg.header.frame_id.endswith(
                "_optical_frame"):
            # optical vs non-optical is a silent 90 deg rotation: the map fills
            # in at right angles to where the camera is pointing and looks
            # entirely plausible while doing it. Fail loudly, once.
            rospy.logfatal(
                "depth frame_id is %r, which is not an optical frame. The "
                "image data is in optical convention regardless, so this "
                "would rotate the whole map. Set ~allow_non_optical_frame if "
                "you know better.", msg.header.frame_id)
            rospy.signal_shutdown("depth image is not in an optical frame")
            return

        if self._depth_frame is None:
            self._depth_frame = msg.header.frame_id.lstrip('/')
            if self.pose_source != 'tf':
                self._resolve_static()

        t_msg = msg.header.stamp.to_sec()
        try:
            age = (rospy.Time.now() - msg.header.stamp).to_sec()
        except Exception:
            age = 0.0
        if self.tf_max_age > 0.0 and age > self.tf_max_age:
            # A backlog is better dropped than fused against a pose the buffer
            # had to reach for.
            self.n_dropped_stale += 1
            return
        if not self.clock.due(t_msg):
            self.n_dropped_rate += 1
            return

        if self.pose_source == 'tf':
            pose = self.lookup(msg.header.frame_id, msg.header.stamp)
        else:
            pose = self.lookup_topic(msg.header.stamp)
        if pose is None:
            self.n_dropped_tf += 1
            return
        cam_xyz, R_world_opt = pose

        # PER STAGE, because the total on its own cannot tell a slow model
        # from a slow publish, and the two want opposite fixes. The bench
        # (perception/bench_fuse.py) measures the fuse stage ALONE on a
        # synthetic room, so this is also the only way to see how far that
        # room is from the one the camera is actually looking at.
        t0 = time.time()
        depth = decode_depth(msg)
        if self.decim > 1 or self.row_decim > 1:
            # A STRIDE, not an average: averaging across a depth discontinuity
            # invents a surface between the foreground and the background,
            # standing at neither range. Rows first, columns second, matching
            # the order stereo_from_camera_info divided the intrinsics in.
            depth = depth[::self.row_decim, ::self.decim]

        if not self.grid.seeded:
            self._maybe_seed(cam_xyz)

        t1 = time.time()
        fuse_frame(depth, cam_xyz, R_world_opt, self.stereo, self.profile,
                   self.grid, self.plane_height, self.plane_tol,
                   rays_per_col=self.rays_per_col)
        t2 = time.time()
        self.n_fused += 1

        if self.n_fused % self.publish_every_n == 0:
            self.publish(msg.header.stamp)
            if self.publish_wedge:
                self.publish_wedge_marker(msg.header.stamp, cam_xyz,
                                          R_world_opt)
            if self.publish_footprint:
                self.publish_footprint_marker(msg.header.stamp, cam_xyz)
        t3 = time.time()
        self._ms.append(((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3))
        self._log(t_msg)

    def _maybe_seed(self, cam_xyz):
        """Assert the free disc once. See the note in __init__."""
        if self.seed_radius <= 0.0:
            self.grid.seeded = True
            return
        if self.seed_pose is not None:
            cx, cy = self.seed_pose
        elif self.seed_on == 'altitude':
            if float(cam_xyz[2]) < self.seed_alt:
                return
            cx, cy = float(cam_xyz[0]), float(cam_xyz[1])
        else:                                    # 'first_pose'
            # The disc is HORIZONTAL, in the slice plane, and takeoff is
            # vertical -- so the ground (x, y) is the takeoff (x, y) and no
            # takeoff detection is needed.
            cx, cy = float(cam_xyz[0]), float(cam_xyz[1])
        n = self.grid.seed_free_disc(cx, cy, self.seed_radius)
        rospy.loginfo("asserted a %.2f m free disc at (%.2f, %.2f): %d cells "
                      "(%s)", self.seed_radius, cx, cy, n, self.seed_on
                      if self.seed_pose is None else "explicit ~seed_pose")
        if n == 0:
            rospy.logwarn("the free disc landed entirely outside the grid -- "
                          "the vehicle is not where the map thinks it is")
        self.pub_seeded.publish(Bool(data=True))

    # --------------------------------------------------------------- publish

    def publish(self, stamp):
        data = self.grid.to_int8()
        if self._wall is not None:
            # After to_int8, so precedence between "occupied" and "unknown" is
            # decided here explicitly rather than by the order of two boolean
            # assignments somewhere else.
            data[self._wall] = 100
        g = OccupancyGrid()
        g.header.stamp = stamp
        g.header.frame_id = self.world_frame
        g.info.resolution = self.res
        g.info.width = self.grid.nx
        g.info.height = self.grid.ny
        g.info.origin.position.x = self.min_x
        g.info.origin.position.y = self.min_y
        g.info.origin.orientation.w = 1.0
        g.data = data.reshape(-1).tolist()
        self.pub.publish(g)
        self.n_published += 1

    def publish_wedge_marker(self, stamp, cam_xyz, R_world_opt):
        """The camera's field of view, drawn in the map's own frame.

        THIS IS THE CHEAPEST FRAME-BUG DETECTOR THERE IS. A 90 deg rotation
        error -- optical frame confused for body frame, or a yaw convention
        crossed -- produces a grid that is entirely plausible and simply
        turned. Overlay this on the grid and the failure is visible on the
        first frame with the vehicle standing still: the wedge points one way
        and the newly filled cells appear at right angles to it.
        """
        if self.pub_wedge.get_num_connections() == 0:
            # A rospy publisher serialises its message whether or not anyone is
            # listening, so "nobody is looking" has to be checked here rather
            # than left to the transport. Same idiom as the planner's
            # _publish_rollouts.
            return
        R = np.asarray(R_world_opt)
        fwd = R[:, 2]                            # optical +z is the boresight
        yaw = math.atan2(fwd[1], fwd[0])
        half = math.atan(0.5 * self.stereo.width / self.stereo.fx)
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = self.world_frame
        m.ns = "andert_fov"
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.02
        m.color.a, m.color.r, m.color.g, m.color.b = 0.9, 0.1, 0.9, 0.3
        m.pose.orientation.w = 1.0
        pts = [(cam_xyz[0], cam_xyz[1])]
        for k in range(9):
            a = yaw - half + 2.0 * half * k / 8.0
            pts.append((cam_xyz[0] + self.z_max * math.cos(a),
                        cam_xyz[1] + self.z_max * math.sin(a)))
        pts.append((cam_xyz[0], cam_xyz[1]))
        for x, y in pts:
            m.points.append(Point(x=x, y=y, z=self.plane_height))
        self.pub_wedge.publish(m)

    def publish_footprint_marker(self, stamp, cam_xyz):
        """Rings of `~footprint_radii` about the VEHICLE, in the map's frame.

        Centred on base_link, not on the camera: the camera sits 0.096 m
        forward and 0.06 m to the side of it, which is a fifth of r_quad and
        exactly the error that makes a gap look flyable when it is not. With
        `~pose_source: topic` the vehicle pose is what arrives and the camera
        pose is derived from it, so the right centre is the one already in hand.
        """
        if self.pub_foot.get_num_connections() == 0 or not self.footprint_radii:
            return
        base = self._last_base
        cx, cy = ((base[0], base[1]) if base is not None
                  else (float(cam_xyz[0]), float(cam_xyz[1])))
        mode = self.footprint_z
        if mode == 'grid':
            cz = 0.0
        elif mode == 'slice':
            cz = self.plane_height
        elif mode == 'vehicle':
            cz = float(base[2]) if base is not None else float(cam_xyz[2])
        else:
            try:
                cz = float(mode)
            except ValueError:
                cz = float(base[2]) if base is not None else 0.0
        arr = MarkerArray()
        for k, r in enumerate(self.footprint_radii):
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = self.world_frame
            m.ns = "airframe"
            m.id = k
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.015
            m.color.a = 0.9
            m.color.r, m.color.g, m.color.b = (0.95, 0.75, 0.1) if k == 0 \
                else (0.4, 0.7, 1.0)
            m.pose.orientation.w = 1.0
            for j in range(41):
                a = 2.0 * math.pi * j / 40.0
                m.points.append(Point(x=cx + r * math.cos(a),
                                      y=cy + r * math.sin(a),
                                      z=cz))
            arr.markers.append(m)
        self.pub_foot.publish(arr)

    # ------------------------------------------------------------ diagnostics

    def _log(self, t_msg):
        """A mapper that has quietly stopped fusing looks exactly like one that
        can see nothing, so the drop reasons are separated and always printed."""
        now = time.time()
        if self._t_log == 0.0:
            self._t_log = now
            return
        if now - self._t_log < self.log_period:
            return
        self._t_log = now
        ms = (np.array(self._ms[-200:]) if self._ms
              else np.zeros((1, 3), dtype=float))
        tot = ms.sum(axis=1)
        cov = self.grid.coverage()
        rospy.loginfo(
            "frame %.0f/%.0f ms p50/p95 (decode %.0f fuse %.0f publish %.0f) "
            "| in %d fused %d pub %d | dropped rate %d tf %d stale %d "
            "| seen %.1f%% occ %.1f%% free %.1f%%",
            np.percentile(tot, 50), np.percentile(tot, 95),
            np.percentile(ms[:, 0], 50), np.percentile(ms[:, 1], 50),
            np.percentile(ms[:, 2], 50), self.n_in, self.n_fused,
            self.n_published, self.n_dropped_rate, self.n_dropped_tf,
            self.n_dropped_stale, cov["seen_pct"], cov["occ_pct"],
            cov["free_pct"])
        if self._pose_dt:
            d = np.asarray(self._pose_dt[-200:]) * 1e3
            rospy.loginfo("  pose pairing: p50 %.1f ms p95 %.1f ms max %.1f ms "
                          "(source %s)", np.percentile(d, 50),
                          np.percentile(d, 95), d.max(), self.pose_source)


def _mat(tf_stamped):
    q = tf_stamped.transform.rotation
    tr = tf_stamped.transform.translation
    M = quaternion_matrix([q.x, q.y, q.z, q.w])
    M[:3, 3] = [tr.x, tr.y, tr.z]
    return M


if __name__ == '__main__':
    import gc
    gc.disable()          # avoid periodic GC pauses that spike callback latency
    if hasattr(gc, "freeze"):
        gc.freeze()       # keep startup allocations out of the young generation
    rospy.init_node('depth_to_grid_andert')
    DepthToGridAndert()
    rospy.spin()
