#!/bin/bash
# The flight itself: planner LIVE + mission supervisor. Nothing arms until you
# say so -- this script only gets the two nodes running and then stops.
#
#   ~/catkin_ws/src/offboard_flight/scripts/fly.sh          bring the two up
#   ~/catkin_ws/src/offboard_flight/scripts/fly.sh go       ...and arm + fly
#   ~/catkin_ws/src/offboard_flight/scripts/fly.sh land     land now
#   ~/catkin_ws/src/offboard_flight/scripts/fly.sh hold     freeze in place
#   ~/catkin_ws/src/offboard_flight/scripts/fly.sh resume   continue after hold
#   ~/catkin_ws/src/offboard_flight/scripts/fly.sh state    where it is
#   ~/catkin_ws/src/offboard_flight/scripts/fly.sh stop     kill both nodes
#
# THE MAPPER IS NOT STARTED HERE. Run ~/catkin_ws/src/perception/restart_stack.sh
# first (or `~/start_test_grid.sh vicon`) and confirm /grid_map is publishing;
# this script only replaces the DRY-RUN planner with a live one and adds the
# supervisor.
#
# EVERY PROCESS NAME LIVES IN THIS FILE, never on a caller's command line:
# `pkill -f` matches the invoking ssh shell too, which has already killed two
# sessions here. See perception/stop_planner.sh.

source /opt/ros/noetic/setup.bash
source /home/rogx/catkin_ws/devel/setup.bash
IP=$(ip -4 route get 8.8.8.8 2>/dev/null | awk '{print $7; exit}')
export ROS_IP=${IP:-127.0.0.1}
export ROS_MASTER_URI=http://${ROS_IP}:11311/
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

NS=rogx2
SCRIPTS=/home/rogx/catkin_ws/src/offboard_flight/scripts
SRV=/${NS}/mission_node

# EVERY WEIGHT HERE IS config.yaml's, from the simulator this planner was
# validated in. The node's own defaults are NOT all the same, so the ones that
# differ are passed explicitly rather than left to drift:
#
#   _r_perc 0.10       node default 0.18;  config.yaml safety.r_perc = 0.10
#   _use_geodesic true node default false; config.yaml mppi.use_geodesic = true
#   _w_frontier 5.0    node default 0.0;   config.yaml mppi.w_frontier = 5.0
#   _d_influence 0.60  node default null -> r_safe+0.35 = 0.86;
#                      config.yaml mppi.d_influence = 0.60
#
# w_frontier was 0 only during a retune done against a GROUND-TRUTH map, where
# the one term that rewards revealing space cannot help and measures as dead
# weight. This vehicle flies a belief map, which is the case it exists for:
# 11/12 val episodes reached at 5 against 10/12 at 0, and mean d_goal 0.22 m
# against 0.54. It was tuned together with w_obs 20 and d_influence 0.60, so
# those two travel with it.
#
# STILL DIVERGENT ON PURPOSE, both with measurements in the node: R_dnu = 0
# (config.yaml has [1, 1, 0.2]) and horizon = 20 (config.yaml has 30). Both
# were measured against THIS vehicle's envelope, which is not the simulator's
# -- v_max 1.0 against 0.35, so sigma is 1.2 against 0.306 and the slew term
# bills roughly 16x more here. Restoring either without first matching the
# limits would reproduce a failure that is already written down.
PLANNER_ARGS=${PLANNER_ARGS:-"_use_geodesic:=true _r_perc:=0.10 \
    _w_frontier:=5.0 _d_influence:=0.60 _viz_rollouts:=30"}
GOAL_X=${GOAL_X:-2.0}
GOAL_Y=${GOAL_Y:-0.0}
MISSION_ARGS=${MISSION_ARGS:-""}

_kill_flight_nodes() {
    pkill -f planar_planner_node 2>/dev/null
    pkill -f setpoint_buffer     2>/dev/null
    pkill -f "mission_node.py"   2>/dev/null
    sleep 2
    rosparam delete /${NS}/planar_planner_node 2>/dev/null
    rosparam delete /${NS}/mission_node        2>/dev/null
}

case "${1:-start}" in

stop)
    _kill_flight_nodes
    pgrep -f "mission_node.py" >/dev/null && echo "mission node STILL RUNNING" \
        || echo "stopped, params cleared"
    ;;

start)
    if ! rostopic list >/dev/null 2>&1; then
        echo "no ROS master -- run ~/start_test_grid.sh vicon first"; exit 1
    fi
    echo "pre-flight:"
    for t in /${NS}/mavros/state /${NS}/mavros/local_position/pose \
             /robot/pose_world /grid_map; do
        if timeout 5 rostopic echo -n1 "$t" >/dev/null 2>&1; then
            echo "  $t   ok"
        else
            echo "  $t   NOT PUBLISHING"
        fi
    done
    echo

    _kill_flight_nodes
    cd "$SCRIPTS" || exit 1

    ROS_NAMESPACE=$NS nohup python3 -u planar_planner_node.py \
        _dry_run:=false _goal_x:=${GOAL_X} _goal_y:=${GOAL_Y} \
        $PLANNER_ARGS > /tmp/planner_live.log 2>&1 &
    echo "planner  pid $!  -> /tmp/planner_live.log"

    ROS_NAMESPACE=$NS nohup python3 -u mission_node.py \
        $MISSION_ARGS > /tmp/mission.log 2>&1 &
    echo "mission  pid $!  -> /tmp/mission.log"

    echo "waiting for the world->FCU alignment ..."
    sleep 25
    grep -a "world->FCU\|r_safe=0\|cost:" /tmp/planner_live.log | tail -3
    echo
    grep -a "\[mission\]" /tmp/mission.log | tail -4
    echo
    echo "when you are happy, and with the RC in your hand:"
    echo "    $0 go"
    ;;

go)
    echo "ARMING AND FLYING in 3 seconds -- ctrl-C to abort"
    sleep 3
    rosservice call ${SRV}/start
    ;;

land)   rosservice call ${SRV}/land   ;;
hold)   rosservice call ${SRV}/hold   ;;
resume) rosservice call ${SRV}/resume ;;

state)
    printf "mission  : "; timeout 3 rostopic echo -n1 ${SRV}/state 2>/dev/null \
        | sed -n 's/^data: //p'
    printf "mav mode : "; timeout 3 rostopic echo -n1 /${NS}/mavros/state 2>/dev/null \
        | grep -E "^(armed|mode):" | tr '\n' ' '; echo
    printf "arrived  : "; timeout 3 rostopic echo -n1 /goal_arrive_tf 2>/dev/null \
        | sed -n 's/^data: //p'
    echo "--- last mission log ---"
    grep -a "\[mission\]" /tmp/mission.log | tail -8
    ;;

*)
    echo "usage: $0 [start|go|land|hold|resume|state|stop]"; exit 1
    ;;
esac
