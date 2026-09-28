#!/bin/bash
# Shared body of fly_haa.sh / fly_hpa.sh / fly_desimplex.sh. Sourced with MODE set.
#
# Order (operator decision 2026-09-25):
#   start : preflight -> CollisionStopGuard (enforce) -> guarded mission -> bag
#   go    : takeoff to home_z + TAKEOFF_HEIGHT with NO planner -> settled hover
#           -> planner started at that SAME altitude -> the first fresh planner
#           command enters MISSION automatically -> goal arrival lands.
#   land  : ~land service.     stop : stop only the processes this script started;
#           refused while armed or with unknown FCU state (FORCE=1 overrides on the ground).
#
# Chain:  planner -> commander/set_pose -> CollisionStopGuard (Vicon GT only)
#         -> commander/set_pose_safe -> guarded_mission_node -> MAVROS
#
# Needs:  ~/start_test_grid.sh vicon running (MAVROS, ZED, Vicon, alignment, mapper).
#   MAP=/home/rogx/traj/<session>/map.yaml   Vicon arena map of TODAY's pillars (required)
#   GOAL_INDEX=1..20                         goal from fly_modes/fly_goals.json (the list all three
#                                            methods share: 1..15 planned, 16..20 spares); screened
#                                            against MAP and /grid_map
#   GOAL_X, GOAL_Y                           or an explicit goal in the Vicon world frame (not both)
#   MOUNT_CONFIRMED=1                        HPA/DeSimplex: camera mount checked (required)
#   TAKEOFF_HEIGHT=1.0                       metres above the takeoff point (mission default)
#   RECORD=0                                 no bag
#
# Process names live in this file only. Nothing here uses pkill -f: it matches
# the invoking ssh shell as well. Existing fly.sh is untouched; do NOT run
# `fly.sh stop` during these flights (its pkill -f "mission_node.py" also
# matches guarded_mission_node.py).

source /opt/ros/noetic/setup.bash
source /home/rogx/catkin_ws/devel/setup.bash
IP=$(ip -4 route get 8.8.8.8 2>/dev/null | awk '{print $7; exit}')
export ROS_IP=${IP:-127.0.0.1}
export ROS_MASTER_URI=${ROS_MASTER_URI:-http://${ROS_IP}:11311/}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

NS=rogx2
SCRIPTS=/home/rogx/catkin_ws/src/offboard_flight/scripts
HELPER="python3 -B $SCRIPTS/fly_modes/fly_helper.py"
SRV=/${NS}/guarded_mission_node
STATE_DIR=/tmp/fly_modes_current
RECORDER=fly_recorder
BUNDLE=/home/rogx/hpa_current/deploy/rogx_hpa_v4
# The three-arm simulator comparison's settings (HPA commit 10, DeSimplex
# n_look 10 + handover_decel, HAA horizon 60), for HPA and DeSimplex alike.
SIM_CONFIG=$SCRIPTS/planar_producer_config.sim_test40.json
TAKEOFF_HEIGHT=${TAKEOFF_HEIGHT:-1.0}

_die() { echo "REFUSED: $*"; exit 1; }

_spawn() {   # name, logfile, command...   -> own session, pid file
    local name=$1 log=$2; shift 2
    setsid nohup "$@" > "$log" 2>&1 < /dev/null &
    echo $! > "$STATE_DIR/$name.pid"
    echo "  $name pid $(cat "$STATE_DIR/$name.pid") -> $log"
}

_stop_pid() {   # SIGINT the process group this script created, then TERM
    local f="$STATE_DIR/$1.pid"
    [ -f "$f" ] || return 0
    local pid; pid=$(cat "$f")
    if kill -0 "$pid" 2>/dev/null; then
        kill -INT -- "-$pid" 2>/dev/null
        for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
        kill -0 "$pid" 2>/dev/null && kill -TERM -- "-$pid" 2>/dev/null
        echo "  stopped $1 ($pid)"
    fi
    rm -f "$f"
}

_existing_flight_nodes() {
    rosnode list 2>/dev/null | grep -E "^/${NS}/(mission_node|guarded_mission_node|planar_planner_node|collision_stop_guard)$|hpa_planner_node|desimplex_planner_node|setpoint_buffer"
}

_record_topics() {
    local t="/tf /tf_static /robot/pose_world /robot/pose_world_epoch /robot/frame_alignment /grid_map \
/goal_arrive_tf /${NS}/commander/set_pose /${NS}/commander/set_pose_safe \
/${NS}/commander/collision_stop_status /${NS}/commander/collision_stop_execution \
/${NS}/guarded_mission_node/state /${NS}/mavros/state /${NS}/mavros/extended_state \
/${NS}/mavros/local_position/pose /${NS}/mavros/local_position/odom /${NS}/mavros/local_position/velocity_local \
/${NS}/mavros/setpoint_raw/local /${NS}/mavros/setpoint_raw/target_local /${NS}/mavros/imu/data \
/${NS}/mavros/battery /${NS}/mavros/rc/in /rosout_agg"
    case "$MODE" in
        haa) t="$t /${NS}/planar_planner_node/config /${NS}/planar_planner_node/status /${NS}/planar_planner_node/nominal_path" ;;
        *)   t="$t /${NS}/zed2i/zed_node/depth/depth_registered /${NS}/zed2i/zed_node/depth/camera_info" ;;
    esac
    echo "$t"
}

fly_main() {
case "${1:-start}" in

start)
    rostopic list >/dev/null 2>&1 || _die "no ROS master -- run ~/start_test_grid.sh vicon first"
    [ -n "$MAP" ] && [ -f "$MAP" ] || _die "set MAP=<today's Vicon arena map.yaml>"
    goal_doc=""
    if [ -n "$GOAL_INDEX" ]; then
        [ -z "$GOAL_X$GOAL_Y" ] || _die "set GOAL_INDEX or GOAL_X/GOAL_Y, not both"
        goal_doc=$($HELPER goal --index "$GOAL_INDEX" --map "$MAP") \
            || { echo "$goal_doc"; _die "goal $GOAL_INDEX is not usable on today's map (see clearances above)"; }
        read -r GOAL_X GOAL_Y < <(echo "$goal_doc" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["x"], d["y"])')
        echo "goal #$GOAL_INDEX (Vicon world) = ($GOAL_X, $GOAL_Y)"
    fi
    [ -n "$GOAL_X" ] && [ -n "$GOAL_Y" ] || _die "set GOAL_INDEX=1..20, or GOAL_X and GOAL_Y (Vicon world, metres)"
    if [ "$MODE" != haa ] && [ "$MOUNT_CONFIRMED" != 1 ]; then
        _die "HPA/DeSimplex need MOUNT_CONFIRMED=1 after checking the camera mount"
    fi
    [ -d "$STATE_DIR" ] && [ -n "$(ls "$STATE_DIR"/*.pid 2>/dev/null)" ] && _die "a fly_* run is active ($STATE_DIR); stop it first"
    running=$(_existing_flight_nodes)
    [ -z "$running" ] || _die "flight nodes already running: $running"
    echo "pre-flight ($MODE):"
    optional=""
    [ "$MODE" = hpa ] && optional="--optional /grid_map"
    python3 "$SCRIPTS/preflight.py" --profile mppi $optional || _die "existing preflight failed"
    $HELPER inputs --mode "$MODE" --map "$MAP" || _die "mode inputs / Vicon subjects not ready"
    session=$($HELPER session)
    run=/home/rogx/flights/${MODE}_$(date +%Y%m%d_%H%M%S)
    mkdir -p "$run" "$STATE_DIR" || exit 1
    echo "$run" > "$STATE_DIR/run_dir"; echo "$MODE" > "$STATE_DIR/mode"
    # The goal is fixed at start; `go` uses this saved value, not its own environment.
    echo "$GOAL_X $GOAL_Y" > "$STATE_DIR/goal"; cp "$STATE_DIR/goal" "$run/goal_world.txt"
    [ -z "$goal_doc" ] || echo "$goal_doc" > "$run/goal_index.json"
    cp "$MAP" "$run/map.yaml"
    $HELPER profile --map "$run/map.yaml" --session "$session" --log "$run/collision_stop.jsonl" \
        --out "$run/guard_profile.json" >/dev/null || _die "guard profile"
    echo "session $session  run $run"
    cd "$SCRIPTS" || exit 1
    ROS_NAMESPACE=$NS _spawn guard "$run/guard.log" python3 -u collision_stop_guard.py \
        --profile "$run/guard_profile.json" --enforce
    timeout 15 rostopic echo -n1 /${NS}/commander/collision_stop_status >/dev/null 2>&1 \
        || { fly_main stop; _die "guard did not start (see $run/guard.log)"; }
    rosparam load "$SCRIPTS/fly_modes/guarded_mission.rogx.yaml" /${NS}/guarded_mission_node
    rosparam set /${NS}/guarded_mission_node/session_id "$session"
    rosparam set /${NS}/guarded_mission_node/takeoff_height "$TAKEOFF_HEIGHT"
    ROS_NAMESPACE=$NS _spawn mission "$run/mission.log" python3 -u guarded_mission_node.py
    sleep 3
    kill -0 "$(cat "$STATE_DIR/mission.pid")" 2>/dev/null || { fly_main stop; _die "guarded mission exited (see $run/mission.log)"; }
    if [ "${RECORD:-1}" != 0 ]; then
        _spawn recorder "$run/record.log" rosbag record __name:=$RECORDER -O "$run/flight" --lz4 \
            -e "/vicon/.*" $(_record_topics)
    fi
    echo
    echo "READY ($MODE). Takeoff goes first; the planner is started at hover:"
    echo "    $0 go"
    ;;

go)
    [ -f "$STATE_DIR/run_dir" ] || _die "run '$0 start' first"
    run=$(cat "$STATE_DIR/run_dir")
    [ "$(cat "$STATE_DIR/mode")" = "$MODE" ] || _die "active run is $(cat "$STATE_DIR/mode"), not $MODE"
    read -r GOAL_X GOAL_Y < "$STATE_DIR/goal" || _die "no saved goal; run '$0 start' again"
    echo "goal (Vicon world) = ($GOAL_X, $GOAL_Y), fixed at start"
    echo "ARMING AND TAKING OFF in 3 seconds -- ctrl-C to abort"
    sleep 3
    rosservice call ${SRV}/start || _die "start service"
    echo "waiting for a settled hover..."
    hover=$($HELPER wait-hover --timeout 90) || { echo "$hover"; _die "no settled hover; planner NOT started"; }
    z_local=$(echo "$hover" | python3 -c 'import json,sys; print(json.load(sys.stdin)["z_want_local"])')
    frame=$($HELPER frame --goal-x "$GOAL_X" --goal-y "$GOAL_Y" --z-local "$z_local" --common "$SCRIPTS") \
        || { echo "$frame"; _die "alignment unavailable; planner NOT started (hovering -- use land)"; }
    echo "$frame" > "$run/frame.json"
    read -r z_world gx gy < <(echo "$frame" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["z_world"], *d["goal_local"][:2])')
    world=$(rosparam get /robot/world_frame)
    echo "hover z_local=$z_local (z_world=$z_world); starting $MODE planner"
    cd "$SCRIPTS" || exit 1
    case "$MODE" in
    haa)
        ROS_NAMESPACE=$NS _spawn planner "$run/planner.log" python3 -u planar_planner_node.py \
            _dry_run:=false _world_frame:=$world _goal_x:=$GOAL_X _goal_y:=$GOAL_Y _z0:=$z_world \
            _plan_rate:=10 _horizon:=60 _producer:=haa ;;
    hpa)
        _spawn planner "$run/planner.log" python3 -u hpa_planner_node.py --controller-output \
            --bundle "$BUNDLE" --log "$run/hpa.jsonl" --device cuda --mount-confirmed \
            --goal-local "$gx" "$gy" "$z_local" --z-local "$z_local" \
            --depth-frame zed2i_left_camera_optical_frame --observer-node /$RECORDER \
            --producer-config "$SIM_CONFIG" --summary-file "$run/hpa_summary.json" ;;
    desimplex)
        _spawn planner "$run/planner.log" python3 -u desimplex_planner_node.py --controller-output \
            --bundle "$BUNDLE" --log "$run/desimplex.jsonl" --device cuda --mount-confirmed \
            --goal-local "$gx" "$gy" "$z_local" --z-local "$z_local" \
            --depth-frame zed2i_left_camera_optical_frame --observer-node /$RECORDER \
            --producer-config "$SIM_CONFIG" --snapshot-dir "$run/grids" --haa-backend cuda --supervisor-worker --parallel-probe --probe-cache \
            --summary-file "$run/desimplex_summary.json" ;;
    esac
    echo "MISSION starts by itself on the first fresh planner command (HPA/DeSimplex load the model first)."
    echo "    $0 state     $0 land     $0 stop"
    ;;

land)   rosservice call ${SRV}/land ;;

state)
    printf "execution: "; timeout 3 rostopic echo -n1 /${NS}/commander/collision_stop_execution 2>/dev/null \
        | sed -n 's/^data: //p' | cut -c1-400
    printf "guard    : "; timeout 3 rostopic echo -n1 /${NS}/commander/collision_stop_status 2>/dev/null \
        | sed -n 's/^data: //p' | python3 -c 'import json,sys
try:
    d=json.loads(json.loads(sys.stdin.read()))
    print(d["state"], "ready=%s takeoff_ready=%s reason=%s" % (d["ready"], d.get("takeoff_ready"), d["reason"]))
except Exception as e: print("unavailable", e)'
    printf "mav      : "; timeout 3 rostopic echo -n1 /${NS}/mavros/state 2>/dev/null | grep -E "^(armed|mode):" | tr '\n' ' '; echo
    printf "arrived  : "; timeout 3 rostopic echo -n1 /goal_arrive_tf 2>/dev/null | sed -n 's/^data: //p'
    [ -f "$STATE_DIR/run_dir" ] && echo "run      : $(cat "$STATE_DIR/run_dir")"
    ;;

stop)
    # Never tear down StopBeforeCollision / the mission in the air: land first.
    armed=$(timeout 5 rostopic echo -n1 /${NS}/mavros/state 2>/dev/null | sed -n 's/^armed: //p')
    if [ "$armed" != "False" ] && [ "$FORCE" != 1 ]; then
        _die "vehicle armed or FCU state unknown (armed='$armed'); '$0 land' first, or FORCE=1 on the ground"
    fi
    _stop_pid planner
    _stop_pid mission
    _stop_pid guard
    _stop_pid recorder
    rosparam delete /${NS}/guarded_mission_node 2>/dev/null
    rm -rf "$STATE_DIR"
    echo "stopped (only processes started by fly_*.sh)"
    ;;

*)  echo "usage: $0 [start|go|land|state|stop]"; exit 1 ;;
esac
}
