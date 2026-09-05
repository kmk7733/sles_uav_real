# perception/ — provenance

`renderer.py` and `inverse_sensor.py` are **byte-identical copies** of

    sles_uav_planar_sim  @ fe78468 + working tree
    planar_sim/perception/renderer.py        sha256 66dd969508a1…
    planar_sim/perception/inverse_sensor.py  sha256 d44ec45a5b63…

recorded in `__init__.py:PROVENANCE` and checkable here, without the simulator
repo present:

    python3 -c "from perception import verify; verify()"

**If a diff appears, the aircraft is wrong.** The reason these files are copied
rather than reimplemented is that the flight code and the simulator must run the
same inverse measurement model — the simulator's tests are what pin it, and its
`docs/HAA_MPPI.md` §3a is what justifies `p_occ`, `p_min` and `eta`. Fix a bug
in the simulator repo and re-deploy; do not patch it here, or the two disagree
about what the map means and neither side's tests can see it.

## Why the import in `inverse_sensor.py` is relative

Here the package is `perception`, not `planar_sim.perception`. A relative
import resolves under either name; an absolute one fails at import time on the
aircraft and nowhere else, which is the worst shape a break can take. The
simulator enforces it from its side in
`tests/test_readme_refs.py:check_vendored_imports`, along with the whole
allowed import set: **numpy, typing, `.renderer`, and nothing else**. Anything
added to either file has to exist here: Python 3.8.10, numpy 1.24.4,
scipy 1.10.1, cv2 4.2.0, and no `planar_sim`.

## The tests

`tests/test_inverse_sensor.py` and `tests/test_renderer.py` are the simulator's
own, with a **two-line mechanical rewrite** and no change to any check:

    ROOT = dirname(dirname(abspath(__file__)))  ->  one more dirname
                                                    (tests/ is one level deeper here)
    from planar_sim.perception.X import ...     ->  from perception.X import ...

Run them after any re-deploy. 32 checks and 19 checks; they take about a second.

## What deliberately did NOT come across

    andert.py           AndertMapper needs scipy's Rotation (for the Euler
                        round trip this vehicle bypasses by supplying the
                        rotation matrix whole), simulated_stereo_depth, and
                        SplitOccupancy -> planner.grid.edt_cells. `grid.py`
                        reproduces the four things that are actually needed and
                        cites andert.py line by line for each.
    occupancy.py        planar_map.py already does the EDT on this vehicle
    simulated_stereo_depth.py
                        a simulated sensor has no job on a real one
    config.py, the YAML layer
                        parameters come from rospy.get_param here

## The camera constants are the one thing NOT to reuse

`NATIVE_F = 350` at 640×360 is an 84.9° HFOV. This ZED 2i reports
`fx = 261.4, cx = 327.4, cy = 176.1` at the same raster — **101.5°**, and a
principal point that is not the raster centre. Intrinsics come from
`depth/camera_info` (`fuse.stereo_from_camera_info`) and never from these
constants. `NATIVE_B = 0.12` does hold: the measured baseline is 119.97 mm
(`/usr/local/zed/settings/SN*.conf`).

## Files that are ours, not vendored

    __init__.py      provenance + verify()
    grid.py          AndertGrid — the log-odds accumulator and the monotone
                     `seen` mask
    fuse.py          decode_depth, stereo_from_camera_info, fuse_frame.
                     fuse_frame is called by BOTH the live node and the offline
                     replay, deliberately: a replay that exercises a parallel
                     copy of the flying code validates nothing.
    bench_fuse.py    per-frame cost on this box, no ROS/camera/bag needed
