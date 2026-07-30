# sles_uav_real

Onboard autonomy for the ROG-X quadrotor (Jetson Xavier NX, ROS Noetic, PX4).

This repository holds the packages we wrote. It is checked out **as**
`~/catkin_ws/src`, so the working tree is the live code — there is no copy step.

## Layout

| Path | What |
|---|---|
| `offboard_flight/` | HAA planner (MPPI) + the ANT-X offboard scripts. **Its README documents the whole pipeline** — frames, mapping, planner, tracking, open issues. |
| `mpc_controller/` | earlier MPPI/planner experiments |
| `data_collection/` | flight data collection |
| `overlays/` | files that live inside third-party repos — see below |

## Third-party packages are NOT in this repo

`openmv_cam`, `vicon_bridge`, `vrpn_client_ros`, `zed-ros-examples`,
`zed-ros-wrapper` are separate upstream repos cloned into the same `src/`.
They are gitignored here — clone them yourself when setting up a machine.

## overlays/ — read this before trusting a fresh checkout

Part of our pipeline lives *inside* `zed-ros-examples`, which belongs to
Stereolabs. Git cannot track files inside a nested repository and we cannot push
to their remote, so those files are mirrored under `overlays/` with their real
relative paths preserved.

**They are copies. Editing one does not change the other.**

```bash
./overlays/sync.sh restore    # overlays/ -> live tree (after cloning zed-ros-examples)
./overlays/sync.sh capture    # live tree -> overlays/ (after editing the live files)
```

What is in there, and why it matters:

| File | Role |
|---|---|
| `scripts/depth_to_grid.py` | point cloud → `/grid_map` — the entire mapping stage |
| `scripts/vicon_map_align.py` | broadcasts `vicon/world → map` |
| `scripts/pose_to_world.py` | publishes `/robot/pose_world`, the planner's state source |
| `launch/zed_vicon_grid.launch` | mapping stack, vicon-aligned |
| `launch/zed_depth_grid.launch` | `depth_to_grid` alone — used for rosbag replay |

Without these the vehicle has no map and the planner has no state. They were
untracked inside a third-party checkout, i.e. one `git clean` from being lost,
which is why this directory exists.

Moving them into a package of our own would remove the problem entirely, but it
changes every launch path and `rosrun` reference, so it has not been done yet.

## Third-party packages carry local modifications

Treating them as plain dependencies is not enough — several have edits that the
vehicle depends on and upstream does not have:

| Package | Local change |
|---|---|
| `zed-ros-wrapper` | `grab_frame_rate` 15 → **30**, `area_memory` true → **false**, `max_depth` 5.0 → **10.0**, plus the note explaining why `floor_alignment` stays false (map origin must stay at the start pose) |
| `vicon_bridge` | `launch/vicon.launch` datastream host |

These are kept as `overlays/patches/*.patch` and reapplied by `sync.sh restore`.
`sync.sh capture` regenerates them — **run it after editing anything inside a
third-party package**, or the change exists only on that machine.

## Fresh machine setup

```bash
mkdir -p ~/catkin_ws && cd ~/catkin_ws
git clone https://github.com/kmk7733/sles_uav_real.git src
cd src

vcs import . < third_party.repos      # pinned commits; or clone by hand
./overlays/sync.sh restore            # overlay files + patches

ln -s /opt/ros/noetic/share/catkin/cmake/toplevel.cmake CMakeLists.txt
cd ~/catkin_ws && catkin_make
```

`third_party.repos` pins the exact commits the vehicle is flying, so a fresh
workspace reproduces the tested configuration rather than upstream HEAD.
