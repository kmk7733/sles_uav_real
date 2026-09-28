#!/bin/bash
# Record a flight so it can be evaluated after the fact.
#
#   record_flight.sh                 light  -- everything but camera imagery
#   record_flight.sh depth           + compressed depth and camera_info:
#                                      enough to RE-RUN the mapper offline
#   record_flight.sh full            + raw depth, RGB and the point cloud
#   record_flight.sh dataset         TRAINING COLLECTION -- everything above
#                                    plus stereo RGB, raw depth and the label
#                                    odometry, written in the ~/drone_data
#                                    session layout extract_flight_dataset.py
#                                    reads
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
#   PERCEPTION     /grid_map, the grid the planner consumed. NOT ~inflated:
#                  the inflated set is a pure function of that grid and
#                  r_safe, and r_safe is in ~config, so analyze_flight.py
#                  rebuilds it exactly. Recording it would cost 8.5 ms of
#                  every plan tick for something already implied -- and it is
#                  SUBSCRIBING that costs it, since the planner skips that
#                  work whenever nobody is listening.
#   DECISION       the nominal path, the per-tick status line (valid
#                  fraction, beta, cost, solve ms, viz ms) and
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
# The regex also catches /vicon/markers, and that matters more than it looks.
# vicon_bridge publishes marker data ONLY while something is subscribed, so
# recording it is what turns it on -- and it is what lets analyze_flight.py
# derive each obstacle's footprint from its corner markers instead of being
# told the dimensions. Note the units differ from the segment transforms:
# markers come through in millimetres, straight from the SDK.
REGEX="/vicon/.*"

# The three topics above /${NS}/mavros/setpoint_raw/local exist only when the
# setpoints come from the GCS instead of the onboard planner (fly.sh
# SOURCE=external) and the Vicon safety supervisor is in the chain. Listing
# them unconditionally costs nothing -- rosbag simply records nothing for a
# topic that never appears -- and without them an external-source bag cannot
# say what the supervisor saw or whether it changed anything.

TOPICS="
/tf
/tf_static
/robot/pose_world
/robot/pose_world_epoch
/robot/frame_alignment

/grid_map

/${NS}/planar_planner_node/config
/${NS}/planar_planner_node/status
/${NS}/planar_planner_node/nominal_path
/${NS}/planar_planner_node/goal_marker
/goal_arrive_tf

/${NS}/mission_node/state

/${NS}/commander/set_pose
/${NS}/commander/set_pose_safe
/${NS}/vicon_safety_supervisor/margin
/${NS}/vicon_safety_supervisor/state
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
dataset)
    # WHAT SEPARATES THIS FROM `full`. The three profiles above answer "what
    # did the planner do"; this one has to answer "what can be learned from
    # it", and that is a different topic set. A bag without
    # left/image_rect_color is not a small dataset -- it is NOT A DATASET,
    # because that topic is the extraction reference every sample is keyed on
    # (config.yaml `reference_topic`), and local_position/odom is the label
    # itself (`label_source: mavros_odom`). The 09-08 flight was recorded as
    # `light` and yielded zero training samples for exactly this reason.
    #
    # It is the UNION, not a replacement: the supervisor, mission-state and
    # set_pose topics above stay, so one bag answers both questions.
    TOPICS="$TOPICS
/${NS}/mavros/local_position/odom
/${NS}/mavros/local_position/velocity_body
/${NS}/mavros/altitude
/${NS}/mavros/rc/in
/${NS}/mavros/manual_control/control
/${NS}/zed2i/zed_node/odom
/${NS}/zed2i/zed_node/left/image_rect_color/compressed
/${NS}/zed2i/zed_node/right/image_rect_color/compressed
/${NS}/zed2i/zed_node/depth/depth_registered
/${NS}/zed2i/zed_node/left/camera_info
/${NS}/zed2i/zed_node/right/camera_info
/${NS}/zed2i/zed_node/depth/camera_info
"
    RATE="~270 MB/min measured -- past sessions ran 0.7 to 1.8 GB" ;;
*)  echo "unknown profile '$PROFILE' (light | depth | full | dataset)" >&2; exit 1 ;;
esac

if [ -n "$LIST" ]; then
    echo "profile $PROFILE, plus regex $REGEX"
    echo "$TOPICS" | grep -v '^$'
    exit 0
fi

STAMP=$(date +%Y%m%d_%H%M%S)

if [ "$PROFILE" = dataset ]; then
    # The session layout is not decoration. extract_flight_dataset.py is given
    # a DIRECTORY and globs *.bag inside it, and --split is what lets a long
    # flight exceed one file without anything downstream noticing. The naming
    # and the rosbag options here are the ones every session in ~/drone_data
    # was recorded with; matching them is what makes the old bags and the new
    # ones one dataset instead of two.
    SESSION=/home/rogx/drone_data/session_$STAMP
    mkdir -p "$SESSION"
    OUT=$SESSION/flight_$STAMP
    BAG_OPTS="--lz4 --split --size=2048 -b 1024"
    META=$SESSION/metadata.json
else
    mkdir -p /home/rogx/bags
    OUT=/home/rogx/bags/flight_${STAMP}_${PROFILE}
    BAG_OPTS="--lz4"
fi

echo "profile : $PROFILE  ($RATE)"
echo "writing : ${OUT}_0.bag"
echo "vicon   : $REGEX  (all subjects -- obstacles as well as the aircraft)"
echo "free    : $(df -h /home/rogx | awk 'NR==2{print $4}')"
echo "Ctrl-C to stop."
echo

CMD="rosbag record -O $OUT $BAG_OPTS -e $REGEX $(echo $TOPICS)"

if [ -n "$META" ]; then
    # Provenance only -- the extractor reads its own config.yaml, not this --
    # but a session that cannot say what it recorded is a session nobody
    # trusts a year later. Schema kept identical to record_flight.py's.
    STAMP=$STAMP SESSION=$SESSION NS=$NS CMD=$CMD python3 - > "$META" <<'PYEOF'
import json, os
ns = '/' + os.environ['NS']
state = [ns + t for t in ('/mavros/local_position/odom',
                          '/mavros/local_position/pose',
                          '/mavros/local_position/velocity_local',
                          '/mavros/local_position/velocity_body',
                          '/mavros/imu/data', '/mavros/altitude',
                          '/mavros/state', '/zed2i/zed_node/odom',
                          '/zed2i/zed_node/pose')] + ['/vicon/ROGX2/ROGX2']
sensor = [ns + t for t in ('/zed2i/zed_node/left/image_rect_color/compressed',
                           '/zed2i/zed_node/right/image_rect_color/compressed',
                           '/zed2i/zed_node/depth/depth_registered',
                           '/zed2i/zed_node/left/camera_info',
                           '/zed2i/zed_node/right/camera_info',
                           '/zed2i/zed_node/depth/camera_info')]
print(json.dumps({
    'session': os.path.basename(os.environ['SESSION']),
    'created': os.environ['STAMP'],
    'note': os.environ.get('NOTE', ''),
    'drone_ns': ns,
    'label_source': 'mavros_odom',
    'label_topic': ns + '/mavros/local_position/odom',
    'state_topics': state,
    'rc_topics': [ns + '/mavros/rc/in', ns + '/mavros/manual_control/control'],
    'sensor_topics': sensor,
    # New next to record_flight.py's schema: what the flight itself did, which
    # the training topics alone cannot reconstruct.
    'analysis_topics': [ns + t for t in ('/commander/set_pose',
                                         '/commander/set_pose_safe',
                                         '/vicon_safety_supervisor/margin',
                                         '/vicon_safety_supervisor/state',
                                         '/mission_node/state',
                                         '/mavros/setpoint_raw/local',
                                         '/mavros/setpoint_raw/target_local',
                                         '/mavros/extended_state')]
                       + ['/grid_map', '/robot/pose_world', '/robot/pose_world_epoch', '/goal_arrive_tf'],
    'tf_topics': ['/tf', '/tf_static'],
    'rosbag_options': {'compression': 'lz4', 'split_size_mb': 2048,
                       'buffer_size_mb': 1024},
    'extraction_defaults': {
        'reference_topic': ns + '/zed2i/zed_node/left/image_rect_color/compressed',
        'sample_hz': 10.0, 'sync_slop_s': 0.05, 'save_depth_as': 'npy',
        'horizon_len': 20, 'horizon_dt': 0.1},
    'rosbag_command': os.environ['CMD'],
}, indent=2))
PYEOF
    echo "metadata: $META"
fi

# A topic that does not exist yet is waited on rather than refused, so this is
# safe to start before the planner and the mission node.
exec $CMD
