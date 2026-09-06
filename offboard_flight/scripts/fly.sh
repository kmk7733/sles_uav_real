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

# NOTHING IS OVERRIDDEN HERE ANY MORE. The node's own defaults ARE
# config.yaml's now -- limits, sigma, horizon, R_dnu, w_frontier, d_influence,
# use_geodesic, r_perc -- so the flown settings and the validated ones are the
# same numbers by construction rather than by a caller remembering to pass
# them. _dry_run:=false is the only difference from ~/.restart_planner_dry.sh.
#
# ONE DELIBERATE DIVERGENCE REMAINS, in the conservative direction: d_clr is
# 0.05 here against config.yaml's 0.0, so r_safe is 0.51 against the
# simulator's 0.46. config.yaml retired d_clr as a term but records that grid
# discretisation is still uncompensated beyond edt_margin -- worst case
# sqrt(2)*res ~ 0.071 m -- and says to cover it manually where r_eff is used.
# This is that cover. Set _d_clr:=0.0 to match the simulator exactly.
#
# _plan_rate 5, AND IT IS A MEASUREMENT, not a preference. On this Xavier with
# the ZED, the Andert mapper at 7 Hz, vicon_bridge and foxglove_bridge all
# running, the aligned solve measures 133-255 ms with occasional 400-750 ms
# spikes -- against planner/bench.py's 86.7 ms for the identical settings on an
# IDLE box. Left at 10 Hz the rospy.Timer simply fires again the moment the
# callback returns, so the planner runs back to back and eats a core the mapper
# needs. 5 Hz gives it room.
#
# It costs less than it sounds. Replanning distance, not rate, is what matters:
# at v_max 0.31 m/s, 5 Hz is 6.2 cm of travel per cycle against a 3.0 s / 0.93 m
# horizon. The old 10 Hz at v_max 1.0 was 10 cm, so this replans FINER than
# what was flown before while thinking further ahead.
#
# Sweep by passing PLANNER_ARGS, e.g.
#   PLANNER_ARGS="_w_frontier:=0 _plan_rate:=10" fly.sh
PLANNER_ARGS=${PLANNER_ARGS:-"_plan_rate:=5"}
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
    # ONE node, not one per topic. Each `rostopic echo` pays a full node
    # registration before it can hear anything, and against /mavros/state at
    # 1 Hz a few seconds of budget loses that race and reports NO DATA on a
    # healthy link. preflight.py subscribes to all four at once.
    echo "pre-flight:"
    if ! python3 "$SCRIPTS/preflight.py"; then
        echo "  refusing to start the planner until those are up"
        exit 1
    fi

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
