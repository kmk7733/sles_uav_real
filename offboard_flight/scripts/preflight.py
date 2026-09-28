#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is everything the flight needs actually publishing? One node, one wait.

    python3 preflight.py [--profile mppi|external] [--timeout 12]

WHY NOT `rostopic echo -n1` IN A LOOP, WHICH IS WHAT THIS REPLACES.
Every `rostopic` invocation pays a full node registration and subscriber
handshake before it can hear anything, and on this Xavier under the ZED, the
mapper and foxglove that is seconds. Against /mavros/state, which publishes at
1 Hz, a 3 s budget loses that race and reports NO DATA on a healthy FCU link --
measured: nothing at 3 s, `connected: True` at 12 s, on a link that was up the
whole time. A pre-flight check that cries wolf about the FCU is worse than no
check, because the next thing anyone does is stop trusting it.

One node subscribes to everything at once and waits, so the handshake is paid
once and slow topics are given real time rather than a share of it.
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import rospy
from geometry_msgs.msg import PoseStamped, TransformStamped
from mavros_msgs.msg import State
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String

try:
    from ekf_alignment import SharedFrameAlignment
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
    from ekf_alignment import SharedFrameAlignment

NS = rospy.get_namespace().strip("/") or "rogx2"

# topic, type, nominal rate [Hz], what it is for
WANT = [
    ("/%s/mavros/state" % NS, State, 1.0, "FCU link"),
    ("/%s/mavros/local_position/pose" % NS, PoseStamped, 30.0, "EKF2 pose"),
    ("/robot/pose_world", PoseStamped, 30.0, "EKF body pose in planning world"),
    ("/robot/pose_world_epoch", TransformStamped, 30.0, "EKF body pose with alignment epoch"),
    ("/grid_map", OccupancyGrid, 10.0, "Andert occupancy grid"),
    ("/robot/frame_alignment", String, 10.0, "fixed EKF/world alignment"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=("mppi", "external"), default="mppi",
                    help="external checks the original pose streams without EKF alignment metadata")
    ap.add_argument("--timeout", type=float, default=12.0,
                    help="seconds to wait for the slowest topic")
    # DOWNGRADED, NOT REMOVED. A flight that does not consume the map does not
    # need the mapper up, and requiring it would make fly.sh refuse a perfectly
    # valid run -- but dropping the check entirely would hide the thing the
    # operator most needs to know, which is that /grid_map and
    # /robot/pose_world will therefore be ABSENT FROM THE BAG. So an optional
    # topic is still subscribed, still waited for, and still reported; it just
    # cannot fail the check.
    # Repeatable:  --optional /grid_map --optional /robot/pose_world
    ap.add_argument("--optional", action="append", default=[], metavar="TOPIC",
                    help="report this topic but do not require it (repeatable)")
    args = ap.parse_args()

    want = list(WANT)
    optional = set(args.optional)
    if args.profile == "external":
        want = [entry for entry in want if entry[0] not in
                ("/robot/frame_alignment", "/robot/pose_world_epoch")]
        optional.add("/grid_map")
    for t in optional:
        if t not in [w[0] for w in WANT]:
            print("  note: --optional %s is not in the check list anyway" % t)

    rospy.init_node("preflight", anonymous=True, disable_signals=True)
    got, received, subs = {}, {}, []
    lock = threading.Lock()
    alignment = SharedFrameAlignment()

    def callback(topic, msg):
        now = rospy.get_time()
        with lock:
            got[topic], received[topic] = msg, now
            if topic == "/robot/frame_alignment":
                alignment.update_status(msg.data, now=now)

    def problem(topic, msg, now):
        if msg is None:
            return "missing"
        if now-received[topic] < 0 or now-received[topic] > (2.0 if isinstance(msg, State) else 0.5):
            return "stale"
        if isinstance(msg, State):
            return "FCU disconnected" if not msg.connected else None
        if isinstance(msg, String):
            return None if alignment.is_ready(now, 0.5) else alignment.reason
        if isinstance(msg, (PoseStamped, TransformStamped, OccupancyGrid)):
            age = now-msg.header.stamp.to_sec()
            if not 0 <= age <= 0.5:
                return "stale source timestamp"
        # Data collection uses the original ZED/Vicon world pose. Its external
        # path generator aligns that pose against local EKF independently.
        if args.profile == "external":
            return None
        if topic in ("/robot/pose_world", "/robot/pose_world_epoch", "/grid_map"):
            if not alignment.is_ready(now, 0.5):
                return "alignment unavailable"
            if msg.header.frame_id != alignment.world_frame:
                return "world frame mismatch"
            if topic in ("/robot/pose_world", "/robot/pose_world_epoch") and msg.header.stamp.to_sec() < alignment.valid_from:
                return "pose predates alignment"
            if topic == "/robot/pose_world_epoch" and msg.child_frame_id != "ekf_body/epoch/"+alignment.epoch:
                return "pose alignment epoch mismatch"
            if topic == "/grid_map" and msg.info.map_load_time != rospy.Time.from_sec(alignment.valid_from):
                return "grid alignment epoch mismatch"
        return None

    for topic, typ, _hz, _what in want:
        subs.append(rospy.Subscriber(
            topic, typ, lambda m, t=topic: callback(t, m), queue_size=1))

    t0 = time.monotonic()
    while not rospy.is_shutdown():
        with lock:
            problems = {t: problem(t, got.get(t), rospy.get_time()) for t, *_ in want}
        if (not any(issue for topic, issue in problems.items() if topic not in optional)
                or time.monotonic()-t0 > args.timeout):
            break
        time.sleep(0.1)

    bad = []
    print("")
    for topic, _typ, hz, what in want:
        msg = got.get(topic)
        if msg is None:
            if topic in optional:
                print("  absent   %-38s %s" % (topic, what))
                print("           not required here, but it will be MISSING "
                      "FROM THE BAG")
            else:
                print("  MISSING  %-38s %s" % (topic, what))
                bad.append(topic)
            continue
        extra = ""
        if isinstance(msg, State):
            extra = "connected=%s armed=%s mode=%s" % (msg.connected,
                                                       msg.armed, msg.mode)
            if not msg.connected and topic not in optional:
                bad.append(topic)
        elif isinstance(msg, PoseStamped):
            p = msg.pose.position
            extra = "(%.2f, %.2f, %.2f)" % (p.x, p.y, p.z)
        elif isinstance(msg, OccupancyGrid):
            extra = "%dx%d @ %.3f m, origin (%.2f, %.2f)" % (
                msg.info.width, msg.info.height, msg.info.resolution,
                msg.info.origin.position.x, msg.info.origin.position.y)
        elif isinstance(msg, String):
            extra = "epoch=%s %s <- %s" % (alignment.epoch, alignment.world_frame, alignment.local_frame)
        elif isinstance(msg, TransformStamped):
            extra = msg.child_frame_id
        issue = problems[topic]
        if issue:
            extra += " -- " + issue
            if topic not in optional and topic not in bad:
                bad.append(topic)
        print("  %-8s %-38s %s" % ("NOT READY" if issue else "ok", topic, extra))

    for s in subs:
        s.unregister()
    print("")
    if bad:
        print("  NOT READY: %s" % ", ".join(bad))
        return 1
    if args.profile == "external":
        print("  external ready: FCU and pose streams fresh; grid/epoch metadata not required")
    else:
        print("  required topics fresh; frame and map epochs agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
