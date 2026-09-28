#!/usr/bin/env python3
"""
depth_to_grid.py  -- 2D horizontal occupancy grid directly from a ZED point cloud + pose.

No SLAM. Each cloud uses the nearest EKF world body pose and static camera
extrinsics (full orientation), with the same alignment epoch as the planner.
Points in a height band [floor+z_min, floor+z_max] become obstacles; the space along
each beam is cleared to free. Log-odds accumulation, published at cloud rate.

Grid is a fixed rectangular window in the world frame, given by [min_x,max_x] x
[min_y,max_y]. Defaults match zed_depth_grid.launch: a 7x5 m interior,
x[-1,6], y[-2.5,2.5], relative to the initial EKF body. The Vicon launch passes
its fixed-room origin while retaining the same 7x5 m dimensions.

Subscribes:  ~cloud_in  (sensor_msgs/PointCloud2, registered, in a camera frame)
Publishes:   ~grid_out  (nav_msgs/OccupancyGrid, in <world_frame>)
"""
import time
import threading
import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
from tf.transformations import quaternion_matrix
from ekf_alignment import SharedFrameAlignment

_PF = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}


def cloud_to_xyz(msg):
    """Vectorized PointCloud2 -> (N,3) float32 array (no ros_numpy dependency)."""
    dt = np.dtype({
        'names':   [f.name for f in msg.fields],
        'formats': [_PF[f.datatype] for f in msg.fields],
        'offsets': [f.offset for f in msg.fields],
        'itemsize': msg.point_step,
    })
    arr = np.frombuffer(msg.data, dtype=dt, count=msg.width * msg.height)
    xyz = np.empty((arr.shape[0], 3), dtype=np.float32)
    xyz[:, 0] = arr['x']
    xyz[:, 1] = arr['y']
    xyz[:, 2] = arr['z']
    return xyz


class DepthToGrid(object):
    def __init__(self):
        self.world_frame = rospy.get_param('~world_frame', 'plan_world')
        # tf2 rejects leading '/' in lookups; messages keep the name verbatim
        self.world_frame_tf = self.world_frame.lstrip('/')
        self.res   = float(rospy.get_param('~resolution', 0.05))
        # rectangular world-frame extents (m)
        self.min_x = float(rospy.get_param('~grid_min_x', -1.0))
        self.max_x = float(rospy.get_param('~grid_max_x',  6.0))
        self.min_y = float(rospy.get_param('~grid_min_y', -2.5))
        self.max_y = float(rospy.get_param('~grid_max_y', 2.5))
        self.z_min = float(rospy.get_param('~obstacle_min_height', 0.15))
        self.z_max = float(rospy.get_param('~obstacle_max_height', 2.0))
        self.auto_floor = bool(rospy.get_param('~auto_floor', True))
        self.floor_z = float(rospy.get_param('~floor_z', 0.0))
        self.floor_ema = float(rospy.get_param('~floor_ema', 0.1))
        self.stride = int(rospy.get_param('~point_stride', 4))
        self.max_range = float(rospy.get_param('~max_range', 6.0))
        self.min_hits = int(rospy.get_param('~min_cell_hits', 1))
        self.clear_stride = int(rospy.get_param('~clear_stride', 9))
        self.clear_samples = int(rospy.get_param('~clear_samples', 16))
        self.l_occ = float(rospy.get_param('~l_occ', 0.85))
        self.l_free = float(rospy.get_param('~l_free', 0.4))
        self.l_min = float(rospy.get_param('~l_min', -2.0))
        self.l_max = float(rospy.get_param('~l_max', 3.5))
        self.occ_thr = float(rospy.get_param('~occ_logodds_thr', 0.4))
        self.free_thr = float(rospy.get_param('~free_logodds_thr', -0.4))
        self.timing = bool(rospy.get_param('~timing', True))
        # free-space prior: cells within this radius of the sensor are forced
        # free each frame (ZED min-range blind zone is otherwise never observed;
        # also gives the planner room at startup when unknown == obstacle).
        # Never overwrites confident obstacles or cells hit this frame.
        self.near_free_enable = bool(rospy.get_param('~near_free_enable', True))
        self.near_free_radius = float(rospy.get_param('~near_free_radius', 1.0))
        # border wall prior: an always-occupied ring AROUND the configured
        # extents (the interior keeps the requested size; the grid is grown
        # outward by the wall thickness). Keeps the planner inside the arena.
        self.wall_enable = bool(rospy.get_param('~border_wall_enable', True))
        self.wall_t = float(rospy.get_param('~border_wall_thickness', 0.1))

        wall_c = int(np.ceil(self.wall_t / self.res)) if self.wall_enable else 0
        self.wall_cells = max(wall_c, 0)
        if self.wall_cells:
            self.min_x -= self.wall_cells * self.res
            self.max_x += self.wall_cells * self.res
            self.min_y -= self.wall_cells * self.res
            self.max_y += self.wall_cells * self.res

        self.nx = int(round((self.max_x - self.min_x) / self.res))
        self.ny = int(round((self.max_y - self.min_y) / self.res))
        self.L = np.zeros((self.ny, self.nx), dtype=np.float32)   # row=y, col=x
        self._wall = None
        if self.wall_cells:
            w = self.wall_cells
            self._wall = np.zeros((self.ny, self.nx), dtype=bool)
            self._wall[:w, :] = True
            self._wall[-w:, :] = True
            self._wall[:, :w] = True
            self._wall[:, -w:] = True
            self.L[self._wall] = self.l_max
        self._floor = None
        self._warned = 0

        self._lock = threading.RLock()
        self.alignment = SharedFrameAlignment(
            self.world_frame_tf, rospy.get_param('~local_frame', 'fcu_local'))
        self.alignment_max_age = float(rospy.get_param('~alignment_max_age', 0.5))
        self.pose_max_dt = float(rospy.get_param('~pose_max_dt', 0.05))
        self.frame_max_age = float(rospy.get_param('~frame_max_age', 0.25))
        self.pose_frame = rospy.get_param('~pose_frame', 'base_link').lstrip('/')
        self._ready = False
        self._generation = 0
        self._poses = []
        self._static = {}
        self._extrinsics = {}
        self.pub = rospy.Publisher('~grid_out', OccupancyGrid, queue_size=1)
        self.sub_alignment = rospy.Subscriber(
            rospy.get_param('~alignment_topic', '/robot/frame_alignment'),
            String, self.cb_alignment, queue_size=10)
        self.sub_pose = rospy.Subscriber(
            rospy.get_param('~pose_topic', '/robot/pose_world_epoch'),
            TransformStamped, self.cb_pose, queue_size=50)
        self.sub_static = rospy.Subscriber('/tf_static', TFMessage,
                                            self.cb_static, queue_size=50)
        self._alignment_timer = rospy.Timer(rospy.Duration(0.1), self.check_alignment)
        self.sub = rospy.Subscriber('~cloud_in', PointCloud2, self.cb,
                                    queue_size=1, buff_size=2 ** 24)
        rospy.loginfo("depth_to_grid: %dx%d cells @ %.3fm, x[%.1f,%.1f] y[%.1f,%.1f], world=%s",
                      self.nx, self.ny, self.res, self.min_x, self.max_x,
                      self.min_y, self.max_y, self.world_frame)

    def _clear_epoch(self):
        self._generation += 1
        self.L = np.zeros((self.ny, self.nx), dtype=np.float32)
        if self._wall is not None:
            self.L[self._wall] = self.l_max
        self._floor = None
        self._poses.clear()

    def _ready_locked(self):
        ready = self.alignment.is_ready(rospy.Time.now().to_sec(), self.alignment_max_age)
        if self._ready and not ready:
            self._clear_epoch()
        self._ready = ready
        return ready

    def cb_alignment(self, msg):
        with self._lock:
            if self.alignment.update_status(msg.data, rospy.Time.now().to_sec()):
                self._clear_epoch()
                self._ready = False
            self._ready_locked()

    def check_alignment(self, _event):
        with self._lock:
            self._ready_locked()

    def cb_pose(self, msg):
        if msg.header.frame_id.lstrip('/') != self.world_frame_tf:
            return
        p, q = msg.transform.translation, msg.transform.rotation
        M = quaternion_matrix([q.x, q.y, q.z, q.w])
        M[:3, 3] = [p.x, p.y, p.z]
        if not np.isfinite(M).all():
            return
        with self._lock:
            stamp = msg.header.stamp.to_sec()
            if (not self._ready_locked() or stamp < self.alignment.valid_from or
                    msg.child_frame_id != 'ekf_body/epoch/' + self.alignment.epoch):
                return
            self._poses.append((stamp, M))
            if len(self._poses) > 128:
                del self._poses[:64]

    def cb_static(self, msg):
        with self._lock:
            for tr in msg.transforms:
                t, q = tr.transform.translation, tr.transform.rotation
                M = quaternion_matrix([q.x, q.y, q.z, q.w])
                M[:3, 3] = [t.x, t.y, t.z]
                self._static[(tr.header.frame_id.lstrip('/'), tr.child_frame_id.lstrip('/'))] = M
            self._extrinsics.clear()

    def lookup(self, frame, stamp):
        """Use the measured EKF pose nearest cloud time and static body extrinsics."""
        frame = frame.lstrip('/')
        T = self._extrinsics.get(frame)
        if T is None:
            stack, seen = [(self.pose_frame, np.eye(4))], {self.pose_frame}
            while stack:
                node, M = stack.pop()
                if node == frame:
                    T = self._extrinsics[frame] = M
                    break
                for (parent, child), edge in self._static.items():
                    if parent == node and child not in seen:
                        seen.add(child)
                        stack.append((child, M.dot(edge)))
        if T is None or not self._poses:
            return None
        t, pose = min(self._poses, key=lambda pair: abs(pair[0] - stamp))
        if abs(t-stamp) > self.pose_max_dt:
            return None
        return pose.dot(T).astype(np.float32)

    def cb(self, msg):
        t0 = time.time()
        stamp = msg.header.stamp.to_sec()
        with self._lock:
            now = rospy.Time.now().to_sec()
            if (not self._ready_locked() or stamp < self.alignment.valid_from or
                    stamp > now + 0.05 or now - stamp > self.frame_max_age):
                return
            generation, epoch = self._generation, self.alignment.epoch
            valid_from = self.alignment.valid_from
            L, floor_previous = self.L, self._floor
            M = self.lookup(msg.header.frame_id, stamp)
        if M is None:
            rospy.logwarn_throttle(5.0, 'depth_to_grid: no matching EKF pose/static camera transform')
            return
        xyz = cloud_to_xyz(msg)
        if self.stride > 1:
            xyz = xyz[::self.stride]
        r2 = xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + xyz[:, 2] ** 2
        keep = np.isfinite(xyz).all(axis=1) & (r2 <= self.max_range ** 2)
        xyz = xyz[keep]
        if xyz.shape[0] == 0:
            return
        sensor = M[:3, 3]
        pw = xyz.dot(M[:3, :3].T) + sensor
        zc = pw[:, 2]
        if self.auto_floor:
            f_now = float(np.percentile(zc[::8], 5.0))
            floor = f_now if floor_previous is None else \
                (1.0 - self.floor_ema) * floor_previous + self.floor_ema * f_now
        else:
            floor = self.floor_z

        nx, ny, res, ox, oy = self.nx, self.ny, self.res, self.min_x, self.min_y
        m = nx * ny
        obs = (zc >= floor + self.z_min) & (zc <= floor + self.z_max)
        t1 = time.time()

        # --- free-space clearing: jittered samples along sensor->point (decimated set) ---
        # Only ray samples whose height is INSIDE the slice band clear cells, so a ray
        # sweeping the floor doesn't write free/occupied into the z-slice it never passes.
        ex = pw[::self.clear_stride, 0]
        ey = pw[::self.clear_stride, 1]
        ez = pw[::self.clear_stride, 2]
        jit = float(np.random.random()) / self.clear_samples
        fr = (np.linspace(0.0, 0.92, self.clear_samples, dtype=np.float32) + jit).clip(0.0, 0.95)
        fx = sensor[0] + np.outer(fr, ex - sensor[0])
        fy = sensor[1] + np.outer(fr, ey - sensor[1])
        fz = (sensor[2] + np.outer(fr, ez - sensor[2])).ravel()
        fcx = np.floor((fx - ox) / res).astype(np.int32).ravel()
        fcy = np.floor((fy - oy) / res).astype(np.int32).ravel()
        fin = ((fcx >= 0) & (fcx < nx) & (fcy >= 0) & (fcy < ny)
               & (fz >= floor + self.z_min) & (fz <= floor + self.z_max))
        free_cnt = np.bincount((fcy[fin] * nx + fcx[fin]), minlength=m).astype(np.float32)
        t2 = time.time()

        # --- obstacles: cells with >= min_hits band points this frame ---
        cx = np.floor((pw[obs, 0] - ox) / res).astype(np.int32)
        cy = np.floor((pw[obs, 1] - oy) / res).astype(np.int32)
        oin = (cx >= 0) & (cx < nx) & (cy >= 0) & (cy < ny)
        occ_cnt = np.bincount((cy[oin] * nx + cx[oin]), minlength=m)

        L -= (self.l_free * np.minimum(free_cnt, 1.0)).reshape(ny, nx)
        L += (self.l_occ * (occ_cnt >= self.min_hits)).astype(np.float32).reshape(ny, nx)
        np.clip(L, self.l_min, self.l_max, out=L)

        if self._wall is not None:
            # border wall prior: always occupied, immune to ray clearing
            L[self._wall] = self.l_max

        if self.near_free_enable:
            occ_hit = (occ_cnt >= self.min_hits).reshape(ny, nx)
            self._seed_near_free(sensor[0], sensor[1], occ_hit, L)

        with self._lock:
            if (not self._ready_locked() or generation != self._generation or
                    epoch != self.alignment.epoch):
                return
            if self.auto_floor:
                self._floor = floor
            self.publish(msg.header.stamp, valid_from)
        if self.timing:
            t3 = time.time()
            rospy.loginfo_throttle(
                3.0, "depth_to_grid: N=%d prep=%.0f clear=%.0f obs+pub=%.0f total=%.0fms (%.1f Hz)"
                % (xyz.shape[0], (t1 - t0) * 1e3, (t2 - t1) * 1e3,
                   (t3 - t2) * 1e3, (t3 - t0) * 1e3, 1.0 / max(t3 - t0, 1e-3)))

    def _seed_near_free(self, sx, sy, occ_hit, L):
        """Force cells within near_free_radius of (sx, sy) to strong free,
        except confident obstacles (L >= occ_thr) and cells hit this frame."""
        r = self.near_free_radius
        x0 = max(int(np.floor((sx - r - self.min_x) / self.res)), 0)
        x1 = min(int(np.floor((sx + r - self.min_x) / self.res)) + 1, self.nx)
        y0 = max(int(np.floor((sy - r - self.min_y) / self.res)), 0)
        y1 = min(int(np.floor((sy + r - self.min_y) / self.res)) + 1, self.ny)
        if x0 >= x1 or y0 >= y1:
            return
        xs = self.min_x + (np.arange(x0, x1) + 0.5) * self.res
        ys = self.min_y + (np.arange(y0, y1) + 0.5) * self.res
        mask = ((xs[None, :] - sx) ** 2 + (ys[:, None] - sy) ** 2) <= r * r
        patch = L[y0:y1, x0:x1]
        protect = (patch >= self.occ_thr) | occ_hit[y0:y1, x0:x1]
        patch[mask & ~protect] = self.l_min

    def publish(self, stamp, valid_from):
        data = np.full(self.L.shape, -1, dtype=np.int8)
        data[self.L >= self.occ_thr] = 100
        data[self.L <= self.free_thr] = 0
        g = OccupancyGrid()
        g.header.stamp = stamp
        g.header.frame_id = self.world_frame
        g.info.map_load_time = rospy.Time.from_sec(valid_from)
        g.info.resolution = self.res
        g.info.width = self.nx
        g.info.height = self.ny
        g.info.origin.position.x = self.min_x
        g.info.origin.position.y = self.min_y
        g.info.origin.orientation.w = 1.0
        g.data = data.reshape(-1).tolist()
        self.pub.publish(g)


if __name__ == '__main__':
    import gc
    gc.disable()   # avoid periodic GC pauses that spike callback latency
    rospy.init_node('depth_to_grid')
    DepthToGrid()
    rospy.spin()
