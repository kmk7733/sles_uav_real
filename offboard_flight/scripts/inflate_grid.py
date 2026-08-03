#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Inflation bands around occupied cells, for display and for planning.

DeSimplex eq (79) inflates obstacles by r_eff = r_Q + r_track + r_perc. Those
terms mean different things, so they are drawn as separate rings rather than
one blob:

  band 1   0    -> 0.18 m   depth uncertainty (r_perc)
  band 2   0.18 -> 0.48 m   the quadrotor's footprint added on top (r_Q = 0.30)

r_perc = 0.18 m comes from 41 pillar samples over 0.66..3.17 m: radial |error|
p95 0.157, p99 0.182. Worth remembering that 90% of the forward errors were in
the safe direction (obstacle measured nearer than it is); against the unsafe
tail alone the p99 was only 0.09 m, so 0.18 is the conservative reading.

Published:
  ~bands           visualization_msgs/MarkerArray  one CUBE_LIST per band, RGBA
                   baked in so Foxglove needs no per-topic colour setup
  ~inflated        nav_msgs/OccupancyGrid          everything inside the OUTER
                   radius marked 100 — this is the grid a planner should use

Params:
  ~radii     cumulative outer radii, ascending  [0.18, 0.48]
  ~colors    flat RGBA per band, 4 numbers each
  ~occ_thresh, ~inflate_unknown, ~marker_z
"""

import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

try:
    import cv2
    _CV2 = True
except ImportError:                                    # pragma: no cover
    _CV2 = False
    from scipy import ndimage

DEFAULT_COLORS = [
    # Alpha washes these toward the background, so the inner band needs a
    # redder hue AND more opacity than looks right on paper — at a=0.55 a
    # G of 0.55 reads as yellow once composited over the light grid.
    0.95, 0.32, 0.02, 0.72,     # r_perc  0.00 -> 0.18 — orange
    1.00, 0.88, 0.20, 0.38,     # r_Q     0.18 -> 0.48 — yellow
    0.20, 0.60, 1.00, 0.30,     # spare              — blue
]


class InflateGrid(object):

    def __init__(self):
        rospy.init_node("inflate_grid")

        self.radii = [float(v) for v in rospy.get_param("~radii", [0.18, 0.48])]
        self.radii.sort()
        cols = [float(v) for v in rospy.get_param("~colors", DEFAULT_COLORS)]
        self.colors = [tuple(cols[i * 4:i * 4 + 4]) for i in range(len(cols) // 4)]
        while len(self.colors) < len(self.radii):
            self.colors.append((0.6, 0.6, 0.6, 0.35))

        self.occ_thresh = int(rospy.get_param("~occ_thresh", 50))
        # Unknown is an obstacle to the planner, but ringing every unknown cell
        # would bury the map — 60-70% of it is unobserved on the ground.
        self.inflate_unknown = bool(rospy.get_param("~inflate_unknown", False))
        self.marker_z = float(rospy.get_param("~marker_z", 0.02))

        self.pub_bands = rospy.Publisher("~bands", MarkerArray, queue_size=1, latch=True)
        self.pub_inf = rospy.Publisher("~inflated", OccupancyGrid, queue_size=1, latch=True)
        rospy.Subscriber("/grid_map", OccupancyGrid, self.cb, queue_size=1)

        rospy.loginfo("[inflate] bands: %s m  (outer %.2f m)",
                      ", ".join("%.2f" % r for r in self.radii), self.radii[-1])
        rospy.loginfo("[inflate] inflate_unknown=%s", self.inflate_unknown)

    @staticmethod
    def _disk(radius_m, res):
        """Disk element. A square one over-grows the diagonals by sqrt(2) —
        at r = 0.48 that is 20 cm of obstacle that is not there."""
        r = max(1, int(np.ceil(radius_m / res)))
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        return ((xx * xx + yy * yy) <= (radius_m / res) ** 2).astype(np.uint8)

    def _grow(self, occ, radius_m, res):
        if radius_m <= 0:
            return occ.copy()
        k = self._disk(radius_m, res)
        if _CV2:
            return cv2.dilate(occ.astype(np.uint8), k,
                              borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        return ndimage.binary_dilation(occ, structure=k.astype(bool))  # pragma: no cover

    def cb(self, msg):
        res = msg.info.resolution
        H, W = msg.info.height, msg.info.width
        g = np.asarray(msg.data, dtype=np.int16).reshape(H, W)

        occ = g >= self.occ_thresh
        if self.inflate_unknown:
            occ |= (g < 0)

        grown = []
        prev = occ
        bands = []
        for r in self.radii:
            gr = self._grow(occ, r, res) if occ.any() else np.zeros((H, W), bool)
            grown.append(gr)
            bands.append(gr & ~prev)          # this band only, not the inner ones
            prev = gr

        # --- planner grid: everything inside the outermost radius
        out = OccupancyGrid()
        out.header = msg.header
        out.info = msg.info
        d = np.full((H, W), -1, dtype=np.int8)
        d[g == 0] = 0                          # keep observed free space free
        if grown:
            d[grown[-1]] = 100
        out.data = d.reshape(-1).tolist()
        self.pub_inf.publish(out)

        # --- display bands
        arr = MarkerArray()
        wipe = Marker()
        wipe.header = msg.header
        wipe.ns = "inflation"
        wipe.action = Marker.DELETEALL
        arr.markers.append(wipe)

        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y
        oz = msg.info.origin.position.z + self.marker_z
        counts = []
        for i, band in enumerate(bands):
            ys, xs = np.nonzero(band)
            counts.append(len(xs))
            m = Marker()
            m.header = msg.header
            m.ns = "inflation"
            m.id = i + 1
            m.type = Marker.CUBE_LIST
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = res
            m.scale.z = 0.01
            m.color = ColorRGBA(*self.colors[i])
            # stack the bands slightly so the viewer does not z-fight
            m.points = [Point(ox + (x + 0.5) * res, oy + (y + 0.5) * res,
                              oz + 0.002 * i)
                        for x, y in zip(xs, ys)]
            arr.markers.append(m)
        self.pub_bands.publish(arr)

        rospy.loginfo_throttle(
            5.0, "[inflate] %d occupied -> bands %s (total %d cells)",
            int(occ.sum()), " / ".join(str(c) for c in counts), sum(counts))


if __name__ == "__main__":
    InflateGrid()
    rospy.spin()
