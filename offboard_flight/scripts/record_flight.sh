#!/bin/bash
# Record a flight so it can be evaluated after the fact.
#
#   record_flight.sh                 light  -- everything but camera imagery
#   record_flight.sh depth           + compressed depth and camera_info:
#                                      enough to RE-RUN the mapper offline
#   record_flight.sh full            + raw depth, RGB and the point cloud
#   record_flight.sh --list          print the topic set and exit
#
# Start it BEFORE `fly.sh go` so the bag covers arming. Ctrl-C to stop.
# fly.sh starts it for you unless you pass RECORD=0.
#
# WHAT IS IN A BAG AND WHY. The question a flight bag has to answer months
# later is "what did the planner see, what did it decide, what did the vehicle
# actually do, and which planner was it anyway". Those are four groups:
#
#   GROUND TRUTH   /vicon/* -- ALL subjects, by regex, not just the aircraft.
#                  Obstacles are Vicon subjects too, and a bag that recorded
#                  only ROGX2 cannot tell you how far from the box it passed.
#   PERCEPTION     the grid the planner consumed, and the inflated set the
#                  validator actually gated on -- /grid_map alone shows
#                  neither the unknown-is-unsafe rule nor the r_safe growth,
#                  so a path that looks timid against the raw grid is
#                  usually hugging ~inflated instead.
#   DECISION       the nominal path, the surviving rollouts, the per-tick
#                  status line (valid fraction, beta, cost, solve ms) and
#                  ~config, which says WHICH producer and under which limits,
#                  weights and git SHA. The HAA, the learned HPA and the
#                  DeSimplex supervisor are indistinguishable from their
#                  setpoints alone, so without ~config a comparison run is
#                  not a comparison.
#   EXECUTION      what was commanded (commander/set_pose -> setpoint_raw),
#                  what PX4 accepted (target_local), where the vehicle went
#                  (local_position, velocity, imu), and the mission state
#                  machine: mission_node/state, /goal_arrive_tf, mavros/state
#                  and extended_state give arm, hover, planning, landing and
#                  disarm as timestamped topics rather than as scrollback.

source /opt/ros/noetic/setup.bash
source /home/rogx/catkin_ws/devel/setup.bash
IP=$(ip -4 route get 8.8.8.8 2>/dev/null | awk '{print $7; exit}')
export ROS_IP=${IP:-127.0.0.1}
export ROS_MASTER_URI=http://${ROS_IP}:11311/

NS=${NS:-rogx2}
PROFILE=${1:-light}
[ "$PROFILE" = "--list" ] && LIST=1 && PROFILE=${2:-light}

# --- ground truth: EVERY Vicon subject, obstacles included -------------------
REGEX="/vicon/.*"

TOPICS="
/tf
/tf_static
/robot/pose_world

/grid_map
/${NS}/planar_planner_node/inflated
/${NS}/planar_planner_node/inflated_outer

/${NS}/planar_planner_node/config
/${NS}/planar_planner_node/status
/${NS}/planar_planner_node/nominal_path
/${NS}/planar_planner_node/rollouts
/${NS}/planar_planner_node/goal_marker
/goal_arrive_tf

/${NS}/mission_node/state

/${NS}/commander/set_pose
/${NS}/mavros/setpoint_raw/local
/${NS}/mavros/setpoint_raw/target_local
/${NS}/mavros/state
/${NS}/mavros/extended_state
/${NS}/mavros/local_position/pose
/${NS}/mavros/local_position/velocity_local
/${NS}/mavros/imu/data
/${NS}/mavros/battery
/${NS}/zed2i/zed_node/pose

/rosout_agg
"

case "$PROFILE" in
light)  RATE="~15 MB/min" ;;
depth)
    # compressedDepth + camera_info is what experiments/bag_grid_map.py needs
    # to rebuild the occupancy grid offline. The point cloud is NOT here: it
    # costs ~12 MB/s of disk and steals CPU from a solve that is already over
    # its budget, and the mapper does not read it anyway.
    TOPICS="$TOPICS
/${NS}/zed2i/zed_node/depth/depth_registered/compressedDepth
/${NS}/zed2i/zed_node/depth/camera_info
/${NS}/zed2i/zed_node/left/camera_info
"
    RATE="~60 MB/min" ;;
full)
    TOPICS="$TOPICS
/${NS}/zed2i/zed_node/depth/depth_registered
/${NS}/zed2i/zed_node/depth/camera_info
/${NS}/zed2i/zed_node/left/image_rect_color/compressed
/${NS}/zed2i/zed_node/point_cloud/cloud_registered
"
    RATE="~700 MB/min -- do not leave running" ;;
*)  echo "unknown profile '$PROFILE' (light | depth | full)" >&2; exit 1 ;;
esac

if [ -n "$LIST" ]; then
    echo "profile $PROFILE, plus regex $REGEX"
    echo "$TOPICS" | grep -v '^$'
    exit 0
fi

mkdir -p /home/rogx/bags
OUT=/home/rogx/bags/flight_$(date +%Y%m%d_%H%M%S)_${PROFILE}

echo "profile : $PROFILE  ($RATE)"
echo "writing : ${OUT}.bag"
echo "vicon   : $REGEX  (all subjects -- obstacles as well as the aircraft)"
echo "free    : $(df -h /home/rogx | awk 'NR==2{print $4}')"
echo "Ctrl-C to stop."
echo

# A topic that does not exist yet is waited on rather than refused, so this is
# safe to start before the planner and the mission node.
exec rosbag record -O "$OUT" --lz4 -e "$REGEX" $TOPICS
