"""ROGX V4 stamped-message adapter. No ROS publishers, services or model weights.

Records are (header_stamp_seconds, ROS_message, monotonic_receipt_seconds).
Buffer ownership/locking and ROS subscriptions belong to the shadow/node caller.
The pose policy is explicit: direct linear interpolation in timestamp and
unwrapped Euler angles, or causal zero-order hold. This is NOT a claim to
reproduce the offline 20 ms regridding, whose phase is absent from the bundle.
Velocity and body angular-z are always original causal samples, never smoothed.
"""
import math
import numpy as np

TIME_EPS = 1e-9


class InputRejected(ValueError):
    pass


class InputPending(InputRejected):
    """Waiting for the bounded future odometry pose bracket; no inference yet."""


class InputExpired(InputRejected):
    """Terminal pose-wait failure: the caller must consume this depth anchor."""


TOPIC_FIELDS = {
    "depth": "/rogx2/zed2i/zed_node/depth/depth_registered: data (32FC1 metres)",
    "camera_info": "/rogx2/zed2i/zed_node/depth/camera_info: K,width,height",
    "pose": "/rogx2/mavros/local_position/odom: pose.pose.position/orientation",
    "angular": "/rogx2/mavros/local_position/odom: twist.twist.angular (state uses z)",
    "velocity": "/rogx2/mavros/local_position/velocity_local: twist.linear (ENU)",
}


def vector3(value):
    out = np.array([value.x, value.y, value.z], dtype=np.float64)
    if not np.isfinite(out).all():
        raise InputRejected("nonfinite vector")
    return out


def odom_pose(message):
    # No PoseStamped or odom.twist.linear fallback: these are different contracts.
    p = vector3(message.pose.pose.position)
    q = message.pose.pose.orientation
    q = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
    quaternion_rpy(q)
    return p, q


def quaternion_rpy(q):
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > .01:
        raise InputRejected("invalid/nonunit odometry quaternion")
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                     math.asin(float(np.clip(2*(w*y-z*x), -1, 1))),
                     math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))])


def rpy_quaternion(rpy):
    r, p, y = np.asarray(rpy, dtype=float)/2
    cr, cp, cy, sr, sp, sy = np.cos(r), np.cos(p), np.cos(y), np.sin(r), np.sin(p), np.sin(y)
    return np.array([sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
                     cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy])


def angle_delta(new, old):
    return np.arctan2(np.sin(np.asarray(new)-old), np.cos(np.asarray(new)-old))


def latest_causal(records, anchor, max_gap, name):
    # Reverse iteration makes equal-stamp replacement use the last original.
    record = next((r for r in reversed(records) if r[0] <= anchor), None)
    if record is None:
        raise InputRejected("missing causal " + name + " at depth anchor")
    if anchor-record[0] > max_gap + TIME_EPS:
        raise InputRejected("operational gap exceeded: " + name)
    return record


def align_inputs(anchor, buffers, *, policy="interpolate", pose_wait_s=.04,
                 waited_s=0., max_gap=.10, camera_max_gap=.25,
                 depth_received=None):
    """Frozen depth anchor -> aligned pose + original causal velocity/angular.

    ``max_gap`` is an operational rejection threshold for dynamic inputs, not
    a training sensor-gap threshold. CameraInfo is fixed session calibration:
    use the first cached record without causal-header or acquisition-age
    selection. The caller validates and caches that calibration once and must
    exclude it from dynamic freshness checks. ``camera_max_gap`` remains in
    the signature for compatibility but is ignored. Waiting uses monotonic
    arrival elapsed time only; dynamic selection uses header timestamps.
    ``depth_received`` is the depth callback's monotonic receipt time. When
    supplied, a future pose must have arrived by its bounded receipt deadline,
    even if inference processing starts later. Interpolation has no automatic
    causal fallback. ``InputExpired`` is terminal for the depth anchor: callers
    must not retry it when a later message arrives. Omitting receipt time leaves
    that deadline unverified. Causal velocity/angular inputs have no receipt
    deadline here; their acquisition age is bounded by the gap settings
    and the caller's freshness checks.
    """
    if policy not in ("interpolate", "causal"):
        raise InputRejected("unknown pose policy")
    if not all(math.isfinite(v) for v in (anchor, pose_wait_s, waited_s, max_gap)) or anchor <= 0 or pose_wait_s < 0 or waited_s < 0 or max_gap <= 0:
        raise InputRejected("invalid timing settings")
    if depth_received is not None and (not math.isfinite(depth_received) or depth_received < 0):
        raise InputRejected("invalid depth monotonic receipt time")
    odom = latest_causal(buffers["pose"], anchor, max_gap, "odom angular/pose")
    velocity = latest_causal(buffers["velocity"], anchor, max_gap, "velocity_local.linear")
    camera = next(iter(buffers["camera_info"]), None)
    if camera is None:
        raise InputRejected("missing fixed camera_info calibration")
    p0, q0 = odom_pose(odom[1]); rp0 = quaternion_rpy(q0)
    records = {"pose_before": odom, "angular": odom, "velocity": velocity, "camera_info": camera}
    p, q, rpy = p0.copy(), q0.copy(), rp0.copy()
    fraction = 0.
    pose_stamp = odom[0]
    if policy == "interpolate" and odom[0] < anchor:
        upper = next((r for r in buffers["pose"] if r[0] >= anchor), None)
        if upper is None:
            if waited_s + TIME_EPS < pose_wait_s:
                raise InputPending("waiting for odometry pose bracket")
            raise InputExpired("pose interpolation wait expired (no causal fallback)")
        if upper[0]-odom[0] > max_gap + TIME_EPS:
            raise InputRejected("operational gap exceeded: pose bracket")
        if depth_received is not None:
            if not math.isfinite(upper[2]) or upper[2] < 0:
                raise InputRejected("invalid future odometry monotonic receipt time")
            if upper[2] - depth_received > pose_wait_s + TIME_EPS:
                raise InputExpired("pose bracket arrived after receipt deadline (no causal fallback)")
        p1, q1 = odom_pose(upper[1]); rp1 = quaternion_rpy(q1)
        fraction = (anchor-odom[0])/(upper[0]-odom[0])
        p = p0 + fraction*(p1-p0)
        rpy = rp0 + fraction*angle_delta(rp1, rp0)
        q = rpy_quaternion(rpy)
        pose_stamp = anchor
        records["pose_after"] = upper
    v = vector3(velocity[1].twist.linear)
    angular = vector3(odom[1].twist.twist.angular)
    return dict(position=p, quaternion=q, rpy=rpy, velocity=v, angular=angular,
                records=records, camera_info=camera[1], pose_stamp=pose_stamp,
                alignment=dict(pose_policy=policy, interpolation_fraction=float(fraction),
                               pose_uses_future_odom="pose_after" in records,
                               pose_wait_s=pose_wait_s,
                               receipt_deadline_checked=depth_received is not None,
                               future_pose_receipt_offset_s=(records["pose_after"][2]-depth_received)
                                  if "pose_after" in records and depth_received is not None else None,
                               velocity_policy="latest original header.stamp <= depth stamp; no smoothing",
                               angular_policy="latest original odom.twist.twist.angular.z at stamp <= depth; no smoothing",
                               camera_policy="first cached fixed-session calibration; no causal-header or age requirement",
                               camera_max_gap_ignored=True,
                               causal_position_delta_m=(p-p0).tolist(),
                               causal_rpy_delta_rad=angle_delta(rpy, rp0).tolist(),
                               original_angular_stamp=odom[0], original_velocity_stamp=velocity[0],
                               pose_effective_stamp=pose_stamp,
                               offline_pose_match="direct timestamp interpolation; offline 0.02s grid phase/interpolation implementation unavailable"))


def quantize_scan(scan, mode="training-fp16"):
    raw = np.asarray(scan, dtype=np.float32)
    if raw.shape != (2, 512) or not np.isfinite(raw).all():
        raise InputRejected("scan must be finite (2,512)")
    if not ((raw >= 0) & (raw <= 1)).all() or not np.isin(raw[1], [0, 1]).all():
        raise InputRejected("scan must contain normalized range and binary validity")
    if mode == "training-fp16":
        out = raw.astype(np.float16).astype(np.float32)
    elif mode == "float32":
        out = raw.copy()
    else:
        raise InputRejected("unknown scan precision policy")
    return out, float(np.max(np.abs(out-raw)))


def planner_state(position, velocity, rpy, angular):
    """Same Euler yaw-rate conversion as planar_sim/adapters/state.py.

    The network state still receives original odom angular.z, independently.
    This helper is for the acceleration-integrated planar reference only.
    """
    position, velocity, rpy, angular = [np.asarray(value, dtype=np.float64)
                                       for value in (position, velocity, rpy, angular)]
    if any(value.shape != (3,) or not np.isfinite(value).all()
           for value in (position, velocity, rpy, angular)):
        raise InputRejected("planner projection needs finite position/velocity/rpy/angular (3,)")
    roll, pitch, yaw = rpy
    cp = math.cos(pitch)
    omega = angular[2] if abs(cp) < 1e-6 else (angular[1]*math.sin(roll)+angular[2]*math.cos(roll))/cp
    return np.array([position[0], position[1], velocity[0], velocity[1], yaw, omega], dtype=np.float64)
