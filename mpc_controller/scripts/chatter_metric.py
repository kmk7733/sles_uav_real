#!/usr/bin/env python3
"""
Chattering metric for /mpc/trajectory.

For each incoming Path: walk to arc length ~lookahead, compute the signed
lateral offset of that point from the start->goal axis. Track the sign over
the last `window` plans (with a deadband) and log flips/window, current side,
and mean |offset| once per second.

  ~path_topic  (default /mpc/trajectory)
  ~goal_x/~goal_y (default 3.0 / 0.0) -- axis endpoint
  ~lookahead   (default 1.0 m)
  ~deadband    (default 0.05 m)  -- |offset| below this keeps the previous side
  ~window      (default 100 plans)
"""

import collections
import math

import rospy
from nav_msgs.msg import Path


class ChatterMetric(object):
    def __init__(self):
        self.gx = float(rospy.get_param('~goal_x', 3.0))
        self.gy = float(rospy.get_param('~goal_y', 0.0))
        self.lookahead = float(rospy.get_param('~lookahead', 1.0))
        self.deadband = float(rospy.get_param('~deadband', 0.05))
        self.window = int(rospy.get_param('~window', 100))
        self.sides = collections.deque(maxlen=self.window)
        self.offsets = collections.deque(maxlen=self.window)
        self.last_side = 0
        self.n_paths = 0
        rospy.Subscriber(rospy.get_param('~path_topic', '/mpc/trajectory'),
                         Path, self.cb, queue_size=10)
        rospy.Timer(rospy.Duration(1.0), self.report)

    def cb(self, msg):
        if len(msg.poses) < 2:
            return
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        # walk to arc length ~lookahead
        acc = 0.0
        look = pts[-1]
        for a, b in zip(pts[:-1], pts[1:]):
            acc += math.hypot(b[0] - a[0], b[1] - a[1])
            if acc >= self.lookahead:
                look = b
                break
        # signed lateral offset from the start->goal axis (z of 2D cross product)
        ax, ay = self.gx - pts[0][0], self.gy - pts[0][1]
        norm = math.hypot(ax, ay)
        if norm < 1e-6:
            return
        e = (ax * (look[1] - pts[0][1]) - ay * (look[0] - pts[0][0])) / norm
        side = self.last_side if abs(e) < self.deadband else (1 if e > 0 else -1)
        self.sides.append(side)
        self.offsets.append(abs(e))
        self.last_side = side
        self.n_paths += 1

    def report(self, _evt):
        if len(self.sides) < 2:
            return
        s = list(self.sides)
        flips = sum(1 for a, b in zip(s[:-1], s[1:]) if a != 0 and b != 0 and a != b)
        mean_off = sum(self.offsets) / len(self.offsets)
        rospy.loginfo('[chatter] plans=%d window=%d flips=%d side=%s mean|e|=%.2fm',
                      self.n_paths, len(s), flips,
                      {1: 'LEFT', -1: 'RIGHT', 0: '-'}[self.last_side], mean_off)


if __name__ == '__main__':
    rospy.init_node('chatter_metric')
    ChatterMetric()
    rospy.spin()
