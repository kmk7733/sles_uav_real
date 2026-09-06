#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is everything the flight needs actually publishing? One node, one wait.

    python3 preflight.py [--timeout 12]

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

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from nav_msgs.msg import OccupancyGrid

NS = rospy.get_namespace().strip("/") or "rogx2"

# topic, type, nominal rate [Hz], what it is for
WANT = [
    ("/%s/mavros/state" % NS, State, 1.0, "FCU link"),
    ("/%s/mavros/local_position/pose" % NS, PoseStamped, 30.0, "EKF2 pose"),
    ("/robot/pose_world", PoseStamped, 100.0, "Vicon pose (world->FCU align)"),
    ("/grid_map", OccupancyGrid, 8.0, "Andert occupancy grid"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=12.0,
                    help="seconds to wait for the slowest topic")
    args = ap.parse_args()

    rospy.init_node("preflight", anonymous=True, disable_signals=True)
    got, subs = {}, []
    for topic, typ, _hz, _what in WANT:
        subs.append(rospy.Subscriber(
            topic, typ, lambda m, t=topic: got.setdefault(t, m), queue_size=1))

    t0 = rospy.get_time()
    while not rospy.is_shutdown() and len(got) < len(WANT):
        if rospy.get_time() - t0 > args.timeout:
            break
        rospy.sleep(0.1)

    bad = []
    print("")
    for topic, _typ, hz, what in WANT:
        msg = got.get(topic)
        if msg is None:
            print("  MISSING  %-38s %s" % (topic, what))
            bad.append(topic)
            continue
        extra = ""
        if isinstance(msg, State):
            extra = "connected=%s armed=%s mode=%s" % (msg.connected,
                                                       msg.armed, msg.mode)
            if not msg.connected:
                bad.append(topic)
        elif isinstance(msg, PoseStamped):
            p = msg.pose.position
            extra = "(%.2f, %.2f, %.2f)" % (p.x, p.y, p.z)
        elif isinstance(msg, OccupancyGrid):
            extra = "%dx%d @ %.3f m, origin (%.2f, %.2f)" % (
                msg.info.width, msg.info.height, msg.info.resolution,
                msg.info.origin.position.x, msg.info.origin.position.y)
        print("  ok       %-38s %s" % (topic, extra))

    for s in subs:
        s.unregister()
    print("")
    if bad:
        print("  NOT READY: %s" % ", ".join(bad))
        return 1
    print("  all four live")
    return 0


if __name__ == "__main__":
    sys.exit(main())
