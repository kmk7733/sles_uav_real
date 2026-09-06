#!/bin/bash
# Bring the whole thing up: Andert mapper + HAA planner, in one call.
#
#   ./restart_stack.sh                       # defaults below
#   RPC=1 Z_MAX=3.5 ./restart_stack.sh       # mapper knobs
#   PLANNER_ARGS="_w_obs:=5" ./restart_stack.sh
#   FOXGLOVE=1 ./restart_stack.sh            # watch it live (costs ~40% of
#                                            # the plan budget)
#
# In a FILE because it names the planner node, and any ssh command line that
# does is a `pkill -f` match against itself -- see stop_planner.sh.
HERE=$(cd "$(dirname "$0")" && pwd)
export MAPPER=${MAPPER:-andert}
export DECIM=${DECIM:-2} ROW_DECIM=${ROW_DECIM:-4} RPC=${RPC:-2} Z_MAX=${Z_MAX:-4.0}
# FOXGLOVE OFF BY DEFAULT, and it is the single biggest thing on this box
# after the ZED. foxglove_nodelet_manager measures 27.3% of a core, and the
# SAME planner measures p50 119 ms with it running against 70 ms without.
# Watching the flight live costs about 40% of the plan budget.
#
# It also decides what the planner does at all: ~rollouts, ~inflated and
# ~inflated_outer skip their work whenever nobody subscribes, so with no
# bridge they cost nothing whatever their parameters say.
#
#   FOXGLOVE=1 ./restart_stack.sh     to watch, and expect to pay for it
#
# VIZ stays 1 on purpose. It only controls the mapper's FOV wedge, which
# gates on get_num_connections like everything else -- so with the bridge
# off it is already free, and leaving it enabled means FOXGLOVE=1 shows the
# whole picture instead of a picture with a hole in it.
export FOXGLOVE=${FOXGLOVE:-0} VIZ=${VIZ:-1}
PLANNER_ARGS=${PLANNER_ARGS:-"_use_geodesic:=true _r_perc:=0.10 _viz_rollouts:=30"}

"$HERE/stop_planner.sh"
~/start_test_grid.sh stop >/dev/null 2>&1
sleep 3
~/start_test_grid.sh vicon >/dev/null 2>&1
echo "waiting for the mapper..."
sleep 48
~/.restart_planner_dry.sh $PLANNER_ARGS 2>&1 | tail -3
