"""ROS-independent validation for the HPA subscriber-only shadow runner."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np


BUNDLE_MANIFEST_SHA256 = "a91d533dd5ba9e84c5545fa91a545c5a4a4e3fd95cdea518bfefaab60487f6b1"
CANDIDATE = "v4_lr1e3_wd1_b64_es10_s1_e40"


class Rejected(ValueError):
    """An input must not produce a prediction."""


def verify_bundle(directory):
    directory = Path(directory).resolve(strict=True)
    manifest_bytes = (directory / "SHA256.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != BUNDLE_MANIFEST_SHA256:
        raise Rejected("bundle manifest differs from the approved V4 bundle")
    manifest = json.loads(manifest_bytes)
    for name, expected in manifest.items():
        if Path(name).name != name or (directory / name).is_symlink():
            raise Rejected("bundle member must be a regular local file: " + name)
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != expected:
            raise Rejected("checksum mismatch: " + name)
    info = json.loads((directory / "model_info.json").read_text())
    if info["candidate"] != CANDIDATE or info["best_epoch"] != 35:
        raise Rejected("unexpected model candidate or epoch")
    return directory, manifest, info


def stamp_seconds(message):
    stamp = float(message.header.stamp.to_sec())
    if not math.isfinite(stamp) or stamp <= 0:
        raise Rejected("missing/nonfinite/nonpositive acquisition timestamp")
    return stamp


def check_age(stamp, now, max_age, future_tolerance):
    if not math.isfinite(stamp) or stamp <= 0:
        raise Rejected("invalid acquisition timestamp")
    if not math.isfinite(now) or now <= 0:
        raise Rejected("invalid ROS clock")
    age = now - stamp
    if age < -future_tolerance:
        raise Rejected("future input")
    if age > max_age:
        raise Rejected("stale input")
    return age


def decode_depth(message):
    if (message.height, message.width) != (360, 640):
        raise Rejected("depth must be exactly 360x640; resizing is forbidden")
    if message.encoding != "32FC1":
        raise Rejected("depth must be 32FC1 optical-axis metres")
    if message.is_bigendian not in (0, 1, False, True):
        raise Rejected("invalid depth endian flag")
    if message.step < 640 * 4 or message.step % 4:
        raise Rejected("invalid depth row stride")
    if len(message.data) != message.height * message.step:
        raise Rejected("depth payload length does not match stride and height")
    dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
    depth = np.ndarray((360, 640), dtype=dtype, buffer=message.data,
                       strides=(message.step, 4))
    # Keep NaN/Inf/zero invalid returns for the packaged projection to handle.
    return np.array(depth, dtype=np.float32, copy=True)


def camera_intrinsics(info, depth, expected_depth_frame):
    if (info.height, info.width) != (360, 640):
        raise Rejected("camera_info dimensions do not match depth")
    if not expected_depth_frame or depth.header.frame_id != expected_depth_frame:
        raise Rejected("depth frame differs from the explicitly configured optical frame")
    if info.header.frame_id != depth.header.frame_id:
        raise Rejected("depth and camera_info frames differ")
    if info.binning_x not in (0, 1) or info.binning_y not in (0, 1):
        raise Rejected("binned camera_info requires a separately verified pixel contract")
    roi = info.roi
    if roi.x_offset or roi.y_offset or roi.width not in (0, 640) or roi.height not in (0, 360):
        raise Rejected("camera_info ROI changes the pixel contract")
    k = np.asarray(info.K, dtype=float)
    if k.shape != (9,) or not np.isfinite(k).all() or k[0] <= 0 or k[4] <= 0:
        raise Rejected("invalid camera_info K")
    if not np.allclose(k[[1, 3, 6, 7, 8]], [0, 0, 0, 0, 1], rtol=0, atol=1e-8):
        raise Rejected("K has unsupported skew or projective terms")
    if not (0 <= k[2] < 640 and 0 <= k[5] < 360):
        raise Rejected("K principal point lies outside the raw depth image")
    return k


def camera_calibration_record(message, received, expected_frame):
    """Validate fixed session calibration without applying dynamic input age.

    A CameraInfo acquisition stamp identifies the calibration's provenance;
    old or zero stamps are valid for a user-confirmed fixed camera session.
    The caller owns first-valid caching. This helper neither modifies the
    message nor relaxes freshness checks for depth or PX4 measurements.
    """
    camera_intrinsics(message, message, expected_frame)
    try:
        source = message.header.stamp
        stamp = float(source.to_sec() if hasattr(source, "to_sec") else
                      source.sec + source.nanosec * 1e-9)
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise Rejected("invalid camera calibration timestamp") from error
    if not math.isfinite(stamp) or stamp < 0:
        raise Rejected("camera calibration timestamp must be finite and nonnegative")
    return stamp, message, received


def quaternion_rpy(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > .01:
        raise Rejected("invalid/nonunit PX4 quaternion")
    x, y, z, w = q / np.linalg.norm(q)
    roll = math.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    pitch = math.asin(float(np.clip(2 * (w*y - z*x), -1, 1)))
    yaw = math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    return np.array([roll, pitch, yaw])


def nearest_synchronized(anchor, buffers, slop):
    selected = {}
    for name, records in buffers.items():
        if not records:
            raise Rejected("missing " + name)
        record = min(records, key=lambda record: abs(record[0] - anchor))
        if abs(record[0] - anchor) > slop:
            raise Rejected("unsynchronized " + name)
        selected[name] = record
    stamps = [anchor] + [record[0] for record in selected.values()]
    if max(stamps) - min(stamps) > slop:
        raise Rejected("synchronized inputs exceed total timestamp span")
    return selected


def pose_discontinuity(previous, current, position_margin, speed_bound, yaw_margin, yaw_rate_bound):
    """Heuristic discontinuity detector; cannot observe all EKF resets."""
    if previous is None:
        return None
    old_stamp, old_position, old_yaw = previous
    stamp, position, yaw = current
    dt = stamp - old_stamp
    if dt < 0:
        return "pose timestamp reversal"
    position_jump = float(np.linalg.norm(np.asarray(position) - old_position))
    yaw_jump = abs(math.atan2(math.sin(yaw - old_yaw), math.cos(yaw - old_yaw)))
    if position_jump > position_margin + speed_bound * dt:
        return "local-position discontinuity (possible estimator reset)"
    if yaw_jump > yaw_margin + yaw_rate_bound * dt:
        return "yaw discontinuity (possible estimator reset)"
    return None


def percentiles(values):
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=float)
    return {"count": len(values), "mean": float(array.mean()), "min": float(array.min()),
            "p50": float(np.percentile(array, 50)), "p95": float(np.percentile(array, 95)),
            "p99": float(np.percentile(array, 99)), "max": float(array.max())}
