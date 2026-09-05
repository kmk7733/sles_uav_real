#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw the last OccupancyGrid in a bag as ASCII, and say where things are.

    python3 show_grid.py grids.bag [/grid_map_andert]

The point is the FRAME CHECK. A 90 degree rotation error, or an origin sign
error, produces a grid that is entirely plausible in the abstract -- walls,
free space, a swept fan -- and is simply turned or shifted. Printing it against
known room geometry is the cheapest way to see that with no camera and no
viewer. The room is 7 x 5 m; the walls should be AT the border, the observed
region should be a fan spreading from the vehicle, and the unknown should be
behind it.
"""
import sys

import numpy as np
import rosbag

GLYPH = {-1: ".", 0: " ", 100: "#"}


def main(argv):
    npz_out = None
    if "--npz" in argv:
        i = argv.index("--npz")
        npz_out = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    path = argv[0]
    topic = argv[1] if len(argv) > 1 else None
    last = None
    with rosbag.Bag(path) as bag:
        _msg_types, topic_info = bag.get_type_and_topic_info()
        topics = [t for t, v in topic_info.items()
                  if v.msg_type == "nav_msgs/OccupancyGrid"]
        if topic:
            topics = [topic]
        for t in topics:
            n = 0
            for _tp, msg, _ts in bag.read_messages(topics=[t]):
                last = (t, msg)
                n += 1
            print("%s: %d messages" % (t, n))
    if last is None:
        print("no OccupancyGrid found")
        return 1

    t, msg = last
    w, h = msg.info.width, msg.info.height
    ox, oy = msg.info.origin.position.x, msg.info.origin.position.y
    res = msg.info.resolution
    g = np.array(msg.data, dtype=np.int8).reshape(h, w)

    occ = int((g == 100).sum())
    unk = int((g == -1).sum())
    free = int((g == 0).sum())
    print("\n%s  %dx%d @ %.3f m  origin (%.2f, %.2f)  stamp %.3f"
          % (t, w, h, res, ox, oy, msg.header.stamp.to_sec()))
    print("occupied %d (%.1f%%)   free %d (%.1f%%)   unknown %d (%.1f%%)"
          % (occ, 100.0 * occ / g.size, free, 100.0 * free / g.size,
             unk, 100.0 * unk / g.size))

    # Where the occupied mass actually sits, in world metres. If the room's
    # walls are at +-3.5 / +-2.5 and this comes back rotated, the frames are
    # wrong -- not the model.
    ys, xs = np.nonzero(g == 100)
    if len(xs):
        wx = ox + (xs + 0.5) * res
        wy = oy + (ys + 0.5) * res
        print("occupied extent  x [%.2f, %.2f]  y [%.2f, %.2f]"
              % (wx.min(), wx.max(), wy.min(), wy.max()))
        interior = (np.abs(wx) < 3.3) & (np.abs(wy) < 2.3)
        print("occupied cells strictly inside the room (not the wall ring): %d"
              % int(interior.sum()))

    print("\n  y +%.1f  (each column is %.2f m; '#' occupied, ' ' free, "
          "'.' unknown)" % (oy + h * res, res))
    step = max(1, h // 44)
    for r in range(h - 1, -1, -step):
        print("  %s" % "".join(GLYPH.get(int(v), "?") for v in g[r, ::2]))
    print("  y %.1f   x from %.1f to %.1f" % (oy, ox, ox + w * res))

    if npz_out:
        out = npz_out
        np.savez_compressed(out, data=g, origin=np.array([ox, oy]),
                            resolution=res, stamp=msg.header.stamp.to_sec(),
                            topic=t)
        print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
