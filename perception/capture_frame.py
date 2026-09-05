#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grab ONE live depth frame with its intrinsics and its TF pose, to an npz.

    rosrun ... capture_frame.py            # or just: python3 capture_frame.py
    python3 capture_frame.py --out /tmp/frame.npz --world vicon/world

Exists so the profiler and the bench can be pointed at the scene the camera is
ACTUALLY looking at. The synthetic room in bench_fuse.py was measured at half
the cost of the real one (115 ms vs 214 ms at the same settings), so tuning
against it is tuning against the wrong scene -- and a profile taken on it would
apportion the time wrongly too.

Saves exactly what `fuse_frame` needs and nothing else: the depth image as it
arrived (NaN and inf preserved), K, the raster size, the camera's world
position, and the world <- optical rotation.
"""
import argparse
import sys

import numpy as np
import rospy
import tf2_ros
from sensor_msgs.msg import CameraInfo, Image
from tf.transformations import quaternion_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/frame.npz")
    ap.add_argument("--world", default="vicon/world")
    ap.add_argument("--depth", default="/rogx2/zed2i/zed_node/depth/depth_registered")
    ap.add_argument("--info", default="/rogx2/zed2i/zed_node/depth/camera_info")
    ap.add_argument("--timeout", type=float, default=15.0)
    a = ap.parse_args()

    rospy.init_node("capture_frame", anonymous=True)
    buf = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
    tf2_ros.TransformListener(buf)

    info = rospy.wait_for_message(a.info, CameraInfo, timeout=a.timeout)
    print("camera_info %dx%d fx %.2f cx %.2f cy %.2f  frame %s"
          % (info.width, info.height, info.K[0], info.K[2], info.K[5],
             info.header.frame_id))

    for attempt in range(60):
        msg = rospy.wait_for_message(a.depth, Image, timeout=a.timeout)
        try:
            t = buf.lookup_transform(a.world.lstrip("/"), msg.header.frame_id,
                                     msg.header.stamp, rospy.Duration(0.2))
            break
        except Exception as exc:                       # noqa: BLE001
            if attempt == 0:
                print("waiting for TF: %s" % exc)
            rospy.sleep(0.1)
    else:
        raise SystemExit("no TF for any depth frame in 60 tries")

    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    depth = (raw.view(np.float32).reshape(h, w) if step == w * 4 else
             raw.reshape(h, step)[:, :w * 4].copy().view(np.float32).reshape(h, w))
    q = t.transform.rotation
    tr = t.transform.translation
    R = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]

    finite = np.isfinite(depth)
    print("depth %dx%d  %.1f%% finite  range [%.2f, %.2f] m  median %.2f"
          % (h, w, 100.0 * finite.mean(), np.nanmin(depth[finite]) if finite.any() else -1,
             np.nanmax(depth[finite]) if finite.any() else -1,
             float(np.median(depth[finite])) if finite.any() else -1))
    np.savez_compressed(a.out, depth=depth, K=np.asarray(info.K),
                        width=info.width, height=info.height,
                        cam=np.array([tr.x, tr.y, tr.z]), R=R,
                        frame_id=msg.header.frame_id,
                        stamp=msg.header.stamp.to_sec())
    print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
