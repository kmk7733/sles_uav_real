# -*- coding: utf-8 -*-
"""Andert (IROS 2009) occupancy mapping, as flown on ROG-X.

`renderer.py` and `inverse_sensor.py` are BYTE-IDENTICAL copies of
`planar_sim/perception/` in the simulator repo `sles_uav_planar_sim`. That is
the whole point of this package: the aircraft runs the same inverse measurement
model the simulator's tests pin, so a fix made there is a fix made here, and a
difference between the two is a bug rather than a variant. See VENDOR.md.

`grid.py` and `fuse.py` are vehicle-side, and they exist because `AndertMapper`
could not come across: it needs scipy's Rotation for a Euler round trip this
node bypasses, the whole simulated-stereo render path, `SplitOccupancy` (the
vehicle's own `planar_map.py` already does the EDT), and the simulator's YAML
config layer. They cite `andert.py` line by line for everything they reproduce.

IMPORTING THIS PACKAGE. `/home/rogx/catkin_ws/src` must be on sys.path, the
same way `planner/` is reached. There is no package.xml and no build step;
these are plain Python modules that a rospy node imports.
"""

import hashlib
import os

__all__ = ["PROVENANCE", "verify"]

# Recorded when the two files were taken. Checked by verify() WITHOUT the
# simulator repo present, which is the only place the check can be run at the
# moment it matters -- on the aircraft, before a flight.
PROVENANCE = {
    "source_repo": "sles_uav_planar_sim",
    "source_path": "planar_sim/perception/",
    "commit": "fe78468+wip",
    "sha256": {
        "renderer.py":
            "66dd969508a141d4c7b07efc7f7f126ba74e4680590a572f0f3317a536d4beb7",
        "inverse_sensor.py":
            "3bcae589b36d34b5fc1b4c3f1fe3e1e1c709db2bd443c4d5d16584fb9cf8ed71",
    },
}


def verify(quiet=False):
    """Raise unless the two vendored files are the bytes they claim to be.

    A local edit to either one is not a merge conflict waiting to happen -- it
    is the aircraft and the simulator silently disagreeing about what the map
    means, which no test on either side alone can see. Fix it in the simulator
    repo and re-deploy; do not patch here.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    bad = []
    for name, want in sorted(PROVENANCE["sha256"].items()):
        path = os.path.join(here, name)
        try:
            with open(path, "rb") as fh:
                got = hashlib.sha256(fh.read()).hexdigest()
        except IOError as exc:
            bad.append("%s: %s" % (name, exc))
            continue
        if got != want:
            bad.append("%s: have %s, expected %s" % (name, got[:12], want[:12]))
        elif not quiet:
            print("ok  %-20s %s" % (name, got[:12]))
    if bad:
        raise RuntimeError(
            "vendored perception files do not match %s@%s:\n  %s\n"
            "Re-deploy from the simulator repo; do not edit them here."
            % (PROVENANCE["source_repo"], PROVENANCE["commit"],
               "\n  ".join(bad)))
    if not quiet:
        print("vendored from %s@%s" % (PROVENANCE["source_repo"],
                                       PROVENANCE["commit"]))
    return True
