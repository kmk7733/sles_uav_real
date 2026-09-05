#!/bin/bash
# Drive the LIVE Andert node from a recorded bag, with no camera and no flight.
#
#   ./replay_live_check.sh [bag] [outdir]
#
# This is the B3 harness of the port plan, and it is the first thing that
# exercises the ROS shell rather than the model: the subscriptions, the
# camera_info handshake, the TF lookup at the frame's own stamp, the optical
# frame assertion, the free-disc seed, and the published int8 contract.
#
# vicon_map_align is NOT started -- the bag replays vicon/world -> map itself,
# and a second publisher of that transform would fight it.
#
# PROCESS TEARDOWN LIVES IN THIS FILE, BY PROCESS GROUP, NEVER BY PATTERN.
# `pkill -f <pattern>` over ssh matches the invoking shell when the pattern is
# in the command string; that has killed two sessions and a live recording on
# this box. Every child here is started with setsid and killed by its own pgid.
# NOTE: no `set -u` -- the ROS setup scripts read unset variables.

BAG=${1:-/home/rogx/bags/depth_debug_20260730_045726.bag}
OUT=${2:-/tmp/andert_check}
MAPPER=${MAPPER:-andert}
DECIM=${DECIM:-2}
RPC=${RPC:-8}
MAP_HZ=${MAP_HZ:-10.0}

source /opt/ros/noetic/setup.bash
source /home/rogx/catkin_ws/devel/setup.bash 2>/dev/null

mkdir -p "$OUT"
CORE_PG=""; LAUNCH_PG=""; REC_PG=""

cleanup() {
    for pg in $REC_PG $LAUNCH_PG $CORE_PG; do
        [ -n "$pg" ] && kill -INT -- "-$pg" 2>/dev/null
    done
    sleep 2
    for pg in $REC_PG $LAUNCH_PG $CORE_PG; do
        [ -n "$pg" ] && kill -KILL -- "-$pg" 2>/dev/null
    done
    return 0
}
trap cleanup EXIT INT TERM

if rostopic list >/dev/null 2>&1; then
    echo "a roscore is already running -- refusing to start a second one."
    echo "stop the live stack first (~/start_test_grid.sh stop)."
    exit 1
fi

echo "== roscore =="
setsid roscore >"$OUT/roscore.log" 2>&1 &
CORE_PG=$!
for _ in $(seq 1 30); do rostopic list >/dev/null 2>&1 && break; sleep 0.5; done
rostopic list >/dev/null 2>&1 || { echo "roscore did not come up"; exit 1; }
rosparam set /use_sim_time true

echo "== mapper: $MAPPER  decim $DECIM  rays/col $RPC  map_hz $MAP_HZ =="
setsid roslaunch zed_rtabmap_example zed_depth_grid.launch \
    mapper:="$MAPPER" open_rviz:=false \
    world_frame:=vicon/world \
    grid_min_x:=-4.0 grid_max_x:=3.0 grid_min_y:=-2.5 grid_max_y:=2.5 \
    decim:="$DECIM" rays_per_col:="$RPC" map_hz:="$MAP_HZ" \
    >"$OUT/node.log" 2>&1 &
LAUNCH_PG=$!
sleep 6

TOPICS="/grid_map /grid_map_andert /depth_to_grid_andert/wedge_out"
setsid rosbag record -O "$OUT/grids.bag" --lz4 $TOPICS \
    >"$OUT/record.log" 2>&1 &
REC_PG=$!
sleep 2

echo "== playing $(basename "$BAG") =="
rosbag play --clock --quiet "$BAG"
sleep 3

echo
echo "== node log tail =="
tail -25 "$OUT/node.log"
echo
echo "== recorded =="
rosbag info "$OUT/grids.bag" 2>/dev/null | sed -n '/topics:/,$p'
