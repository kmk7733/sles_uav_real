#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Do the pose TOPIC and the TF chain give the same camera pose?

    python3 check_pose_source.py            # with the stack running

The mapper needs one thing per frame: where the camera was, in the map's
frame, at the frame's own stamp. tf2 is one way to get it and an expensive one
here -- `TransformListener` subscribes to /tf, which runs at ~458 Hz on this
vehicle, and deserialises every message in Python inside the mapper's own
process. Measured: the same fusion work costs 66 ms in a process without that
listener and 151 ms inside the node with it.

Everything the listener is being asked for is available without it:

    vicon/world -> map      constant, latched once by vicon_map_align
    map -> base_link        /robot/pose_world, 30 Hz, ALREADY in vicon/world
    base_link -> optical    /tf_static, three latched messages

This script runs both paths side by side and reports how far apart they are,
because swapping a pose source on a mapper is exactly the kind of change that
looks fine and quietly rotates the map.
"""
import argparse
import sys

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_matrix
from tf2_msgs.msg import TFMessage


def mat(t, q):
    M = quaternion_matrix([q[0], q[1], q[2], q[3]])
    M[:3, 3] = t
    return M


def static_chain(src, dst, timeout=10.0):
    """T_src_dst from /tf_static alone, walking parent -> child links."""
    got = {}

    def cb(msg):
        for tr in msg.transforms:
            p = tr.header.frame_id.lstrip("/")
            c = tr.child_frame_id.lstrip("/")
            t = tr.transform.translation
            r = tr.transform.rotation
            got[(p, c)] = mat([t.x, t.y, t.z], [r.x, r.y, r.z, r.w])

    sub = rospy.Subscriber("/tf_static", TFMessage, cb, queue_size=50)
    t0 = rospy.Time.now()
    while (rospy.Time.now() - t0).to_sec() < timeout and not rospy.is_shutdown():
        M = _walk(got, src.lstrip("/"), dst.lstrip("/"))
        if M is not None:
            sub.unregister()
            return M, got
        rospy.sleep(0.1)
    sub.unregister()
    return None, got


def _walk(links, src, dst):
    """Depth-first parent->child walk; the ZED chain is a simple path."""
    stack = [(src, np.eye(4))]
    seen = {src}
    while stack:
        node, M = stack.pop()
        if node == dst:
            return M
        for (p, c), T in links.items():
            if p == node and c not in seen:
                seen.add(c)
                stack.append((c, M.dot(T)))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="vicon/world")
    ap.add_argument("--pose-topic", default="/robot/pose_world")
    ap.add_argument("--pose-frame", default="base_link")
    ap.add_argument("--optical", default="zed2i_left_camera_optical_frame")
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()

    rospy.init_node("check_pose_source", anonymous=True)
    buf = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
    tf2_ros.TransformListener(buf)

    T_bo, links = static_chain(a.pose_frame, a.optical)
    if T_bo is None:
        print("could not build %s -> %s from /tf_static; links seen:"
              % (a.pose_frame, a.optical))
        for k in sorted(links):
            print("   %s -> %s" % k)
        return 1
    print("static %s -> %s from /tf_static alone:" % (a.pose_frame, a.optical))
    print("  translation %s" % np.array2string(T_bo[:3, 3], precision=4))
    print("  rotation\n%s" % np.array2string(T_bo[:3, :3], precision=4))

    dp, dr, n = [], [], 0
    while n < a.n and not rospy.is_shutdown():
        try:
            msg = rospy.wait_for_message(a.pose_topic, PoseStamped, timeout=5.0)
        except rospy.ROSException:
            print("no %s" % a.pose_topic)
            return 1
        p, q = msg.pose.position, msg.pose.orientation
        T_wb = mat([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])
        T_topic = T_wb.dot(T_bo)
        try:
            t = buf.lookup_transform(a.world.lstrip("/"), a.optical,
                                     msg.header.stamp, rospy.Duration(0.15))
        except Exception:                                   # noqa: BLE001
            continue
        tr, rr = t.transform.translation, t.transform.rotation
        T_tf = mat([tr.x, tr.y, tr.z], [rr.x, rr.y, rr.z, rr.w])
        dp.append(np.linalg.norm(T_topic[:3, 3] - T_tf[:3, 3]))
        R = T_topic[:3, :3].T.dot(T_tf[:3, :3])
        dr.append(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        n += 1

    if not dp:
        print("no matched samples")
        return 1
    dp, dr = np.array(dp), np.array(dr)
    print("\n%d matched samples" % len(dp))
    print("  position   p50 %.4f  p95 %.4f  max %.4f m" %
          (np.percentile(dp, 50), np.percentile(dp, 95), dp.max()))
    print("  rotation   p50 %.4f  p95 %.4f  max %.4f deg" %
          (np.percentile(dr, 50), np.percentile(dr, 95), dr.max()))
    print("\n  A cell is 0.05 m. Anything under a few mm and a few hundredths")
    print("  of a degree means the two paths are the same transform and the")
    print("  listener is pure overhead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
