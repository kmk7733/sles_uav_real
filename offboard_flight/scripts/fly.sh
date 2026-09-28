#!/bin/bash
# The flight itself: planner LIVE + mission supervisor. Nothing arms until you
# say so -- this script only gets the two nodes running and then stops.
#
#   ~/catkin_ws/src/offboard_flight/scripts/fly.sh          bring the two up
#   ~/catkin_ws/src/offboard_flight/scripts/fly.sh go       ...and arm + fly
#   ~/catkin_ws/src/offboard_flight/scripts/fly.sh land     land now
#
# There is no `hold` or `resume`. HOLD is where the mission node FREEZES on a
# fault -- planner silence, a bad setpoint, the fence, mission_timeout, a
# landing that never confirmed -- and it is not somewhere you steer it by
# hand. The manual override is the RC, which puts it in PILOT for good.
#   ~/catkin_ws/src/offboard_flight/scripts/fly.sh state    where it is
#   ~/catkin_ws/src/offboard_flight/scripts/fly.sh stop     kill both nodes
#
#   RECORD=0 fly.sh          do not start the bag (it is on by default -- a
#                            flight you cannot look at afterwards is a flight
#                            you have to fly again)
#   RECORD=depth fly.sh      bag profile: light | depth | full | dataset
#                            SOURCE=mppi defaults to light, SOURCE=external to
#                            dataset -- the only profile that carries stereo
#                            RGB, depth and the label odometry, i.e. the only
#                            one extract_flight_dataset.py can read. It writes
#                            ~/drone_data/session_<ts>/ instead of ~/bags/.
#
# WHERE THE SETPOINTS COME FROM
#   SOURCE=mppi fly.sh       the live planar planner on this machine (default)
#   SOURCE=external fly.sh   NOTHING is started to produce them. commander/
#                            set_pose is expected from somewhere else -- in
#                            practice the GCS streaming a drawn trajectory:
#
#     # on the GCS, against THIS master
#     rosmode real
#     ROS_NAMESPACE=rogx2 rosrun vicon_traj path_generator.py <traj> \
#         --transform ~/drone_data/frames/world_to_local.yaml
#
#   Everything else is identical -- mission_node, the bag, go/land/state/stop
#   -- because mission_node only ever consumed a topic, and does not care who
#   fills it. Two things do change:
#     * the /grid_map preflight check is skipped. A run that does not consume
#       the map does not need the mapper, and requiring it would refuse a
#       perfectly valid flight.
#     * nothing here can tell you the source is alive. mission_node's
#       ~sp_timeout does: silence for 0.5 s is HOLD.
#
#   To put the Vicon safety supervisor in between, point mission_node at its
#   output instead -- no change here, MISSION_ARGS already reaches it:
#     SOURCE=external MISSION_ARGS="_sp_topic:=commander/set_pose_safe" fly.sh
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
# 10 Hz, AND WHAT IT COST. Measured on the NODE -- not plan() in a loop --
# against the live stack, 40 status ticks each, FOXGLOVE=0 VIZ=0:
#
#   geodesic   N    K    p50   p95    over 100 ms
#   --------  ---  ---  ----  ----   ------------
#     on       30  192   119   154        100%     <- config.yaml's planner
#     on       30  128   131   250        100%
#     on       30   96   104   126         62%
#     on       25   96   124   417         92%
#    OFF       30   96    70   134         25%     <- this
#    OFF       30   64    70   130         20%
#
# LOWERING num_samples IS NOT THE LEVER, and the table says so twice: with the
# geodesic on, K 192 -> 128 made it WORSE, and with it off, K 96 -> 64 changed
# the median by nothing. The geodesic is the whole difference -- 104 ms against
# 70 at the same K=96. It is a grid-wide Dijkstra rebuilt once per solve and it
# does not care how many samples were drawn.
#
# So _use_geodesic:=false is what buys 10 Hz. That is a real loss: the geodesic
# replaces ||p - goal|| with distance measured ALONG traversable space and is
# the fix for "the planner stops at a wall with the goal behind it". Euclidean
# has local minima; a geodesic field has none by construction. w_frontier 5.0
# stays on and partly covers the same ground -- it rewards moving where UNKNOWN
# blocks the line of sight -- but it is not the same guarantee.
#
# THIS NEEDS FOXGLOVE=0. foxglove_nodelet_manager is 27% of a core here and the
# same K=96 geodesic-off config measures p50 119 with it running instead of 70.
# Watching the flight live costs you the plan rate; pick one.
#
# TO FLY config.yaml's PLANNER EXACTLY, at 5 Hz:
#   PLANNER_ARGS="_plan_rate:=5" fly.sh
# K=192 with the geodesic measures p50 119 / p95 154 against a 200 ms budget --
# 60% and 77% of it, more margin than anything in the 10 Hz column. At
# v_max 0.31 m/s, 5 Hz is 6.2 cm of travel per replan against a 3.0 s / 0.93 m
# horizon, so the geodesic costs distance resolution the vehicle does not need.
#
# The principled fix is to rebuild the field every Nth solve rather than every
# solve, since the map changes far slower than the plan does. That is a change
# to planner/haa/cost.py, byte-identical to the simulator's copy by invariant,
# so it belongs there first and not here.
# BACK TO config.yaml's PLANNER. Performance first: K=192 with the geodesic,
# horizon 30, everything the simulator's results were produced with, and only
# the RATE conceded to the hardware. The 10 Hz column above bought its rate by
# deleting the geodesic, which is the planner's only defence against the local
# minima of ||p - goal|| -- too much to pay for a number.
#
#   here (5 Hz)   p50 119  p95 154   against a 200 ms budget: 60% / 77%
#   10 Hz best    p50  70  p95 134   against 100 ms, geodesic OFF
#
# Replanning DISTANCE is what matters and 5 Hz at v_max 0.31 m/s is 6.2 cm per
# cycle against a 3.0 s / 0.93 m horizon.
# HORIZON 60, WHICH IS WHAT v4 ACTUALLY FLEW AND config.yaml's 30 IS NOT.
# `experiments/haa_vs_hpa.py` carries `--haa-horizon` with a DEFAULT OF 60 and
# overrides config.yaml with it, so every HAA arm in runs/test40_c10b512 -- the
# whole v4 comparison -- ran a 6.0 s lookahead while this node ran 3.0 s. It is
# printed in that run's own log: `[HAA] ... horizon 60 -> reach 2.10 m`.
#
# It is not only the lookahead. sigma is DERIVED from the horizon --
# spread = dt*sqrt(N) in mppi_sigma() -- so halving N widens the search:
#
#     horizon 60   sigma = [0.216, 0.216, 0.341]     <- v4
#     horizon 30   sigma = [0.306, 0.306, 0.482]     <- what flew, +41% on yaw
#
# Measured on this box 2026-09-08, novicon grid 224x204, geodesic on, K=192:
#
#     H30  solve p50 199  p95 232 ms  ->  4.92 Hz
#     H60  solve p50 233  p95 275 ms  ->  4.16 Hz
#
# The geodesic is ~100 ms of that and is horizon-INDEPENDENT (grid-wide
# Dijkstra); MPPI proper goes 94 -> 135 ms, which is the horizon. On the
# vicon-aligned 144x104 grid the geodesic term is ~3x cheaper, so H60 should
# land near 170 ms p50 against the 200 ms budget -- confirmed only when Vicon
# is back on. Re-measure before trusting it.
# Runtime target after the exact Python mapper/MPPI optimizations.
# Historical timings above predate these changes; K=192 and horizon=60 remain.
PLANNER_ARGS=${PLANNER_ARGS:-"_plan_rate:=10 _horizon:=60"}
GOAL_X=${GOAL_X:-2.0}
GOAL_Y=${GOAL_Y:-0.0}
MISSION_ARGS=${MISSION_ARGS:-""}

SOURCE=${SOURCE:-mppi}

# SOURCE=external IS the data-collection flight -- a GCS-drawn path flown to
# sample the room -- so it defaults to the profile that can actually become a
# dataset. `light` records no imagery and no label odometry; the 09-08 flight
# was recorded that way and yielded zero training samples.
RECORD=${RECORD:-$([ "$SOURCE" = external ] && echo dataset || echo light)}
[ "$RECORD" = "1" ] && RECORD=light
case "$SOURCE" in
    mppi|external) ;;
    *) echo "SOURCE=$SOURCE -- expected 'mppi' or 'external'"; exit 1 ;;
esac
# External flights require FCU + local pose + the original world pose.
# Grid and MPPI's EKF alignment/epoch metadata are not flight inputs here.
#
# Both come out of zed_vicon_grid.launch, but from SEPARATE nodes -- and they
# are needed for different reasons:
#
#   /grid_map          consumed only by the onboard MPPI planner. An
#                      externally-sourced flight never reads it, so requiring
#                      it would refuse a valid run because the mapper is down.
#
#   /robot/pose_world  The data-collection launch preserves the ZED pose put
#                      through vicon_map_align's fixed world transform. Raw
#                      EKF odometry is also recorded by the dataset profile.
#
# --optional, not --skip: an optional topic is still subscribed and still
# reported, because whether it is up decides what lands in the bag.
#
# To fly with no mapper at all, ask for it explicitly and know what it costs:
#   PREFLIGHT_ARGS="--optional /robot/pose_world" \
#       SOURCE=external fly.sh
PREFLIGHT_ARGS="${PREFLIGHT_ARGS:-}"

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
    pkill -INT -f "rosbag record" 2>/dev/null && echo "bag closed"
    sleep 2
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
    # healthy link. preflight.py subscribes to all required topics at once.
    echo "pre-flight:"
    if ! python3 "$SCRIPTS/preflight.py" $PREFLIGHT_ARGS --profile "$SOURCE"; then
        echo "  refusing to start the planner until those are up"
        exit 1
    fi

    # The alignment owner selects vicon/world or plan_world before this check.
    # Goals and all reference metadata use the same world as the mapper.
    if [ "$SOURCE" = mppi ]; then
        WORLD_FRAME=$(rosparam get /robot/world_frame 2>/dev/null)
        if [ -z "$WORLD_FRAME" ]; then
            echo "no EKF world frame -- restart the grid stack with the new alignment node"
            exit 1
        fi
    fi

    _kill_flight_nodes
    cd "$SCRIPTS" || exit 1

    if [ "$SOURCE" = mppi ]; then
        ROS_NAMESPACE=$NS nohup python3 -u planar_planner_node.py \
            _dry_run:=false _world_frame:=${WORLD_FRAME} _goal_x:=${GOAL_X} _goal_y:=${GOAL_Y} \
            $PLANNER_ARGS > /tmp/planner_live.log 2>&1 &
        echo "planner  pid $!  -> /tmp/planner_live.log"
    else
        echo "planner  NOT STARTED (SOURCE=external)"
        echo "         commander/set_pose must come from somewhere else, or"
        echo "         mission_node will sit in HOLD on ~sp_timeout."
        # Worth saying out loud: preflight checked the FCU link and the EKF2
        # pose and nothing else. It has never checked Vicon -- the onboard
        # planner does not read it -- and an externally-sourced flight usually
        # depends on it completely, through both the drawn path and the
        # supervisor. That check lives on the GCS, not here.
        echo "         NOTE: preflight does NOT check Vicon. Confirm it there:"
        echo "               ./fly_real.sh -c      (or -d, with no FCU)"
    fi

    FRAME_REQUIRED=false
    [ "$SOURCE" = mppi ] && FRAME_REQUIRED=true
    ROS_NAMESPACE=$NS nohup python3 -u mission_node.py \
        _require_frame_alignment:=${FRAME_REQUIRED} \
        $MISSION_ARGS > /tmp/mission.log 2>&1 &
    echo "mission  pid $!  -> /tmp/mission.log"

    # Started here rather than left to the operator, and started BEFORE the
    # arm, so the bag covers takeoff. ~config and mission_node/state are
    # latched, so joining late still captures them -- but nothing else is.
    if [ "$RECORD" != "0" ]; then
        nohup "$SCRIPTS/record_flight.sh" "$RECORD" \
            > /tmp/record.log 2>&1 &
        sleep 2
        grep -a "writing\|profile\|free" /tmp/record.log | sed "s/^/bag      /"
    else
        echo "bag      NOT RECORDING (RECORD=0)"
    fi

    if [ "$SOURCE" = mppi ]; then
        echo "shared EKF alignment ready; waiting for the first planner status ..."
        sleep 5
        grep -a "alignment\|r_safe=0\|cost:" /tmp/planner_live.log | tail -3
    else
        # There is no planner log to wait on. What matters instead is whether
        # the external source has actually appeared, so say that -- and say it
        # about the topic mission_node is really reading, which MISSION_ARGS
        # may have moved.
        sleep 5
        SP_TOPIC=$(echo "$MISSION_ARGS" | sed -n 's/.*_sp_topic:=\([^ ]*\).*/\1/p')
        SP_TOPIC=/${NS}/${SP_TOPIC:-commander/set_pose}
        printf "setpoints on %s: " "$SP_TOPIC"
        if timeout 4 rostopic echo -n1 "$SP_TOPIC" >/dev/null 2>&1; then
            echo "arriving"
        else
            echo "NONE YET -- start the source before 'go', or mission_node"
            echo "  will climb and then HOLD on ~sp_timeout."
        fi
    fi
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

state)
    printf "mission  : "; timeout 3 rostopic echo -n1 ${SRV}/state 2>/dev/null \
        | sed -n 's/^data: //p'
    printf "mav mode : "; timeout 3 rostopic echo -n1 /${NS}/mavros/state 2>/dev/null \
        | grep -E "^(armed|mode):" | tr '\n' ' '; echo
    printf "arrived  : "; timeout 3 rostopic echo -n1 /goal_arrive_tf 2>/dev/null \
        | sed -n 's/^data: //p'
    printf "bag      : "; ls -t /home/rogx/bags/*.bag* 2>/dev/null | head -1
    echo "--- last mission log ---"
    grep -a "\[mission\]" /tmp/mission.log | tail -8
    ;;

*)
    echo "usage: $0 [start|go|land|state|stop]"; exit 1
    ;;
esac
