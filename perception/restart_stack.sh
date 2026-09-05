#!/bin/bash
# Bring the whole thing up: Andert mapper + HAA planner, in one call.
#
#   ./restart_stack.sh                       # defaults below
#   RPC=1 Z_MAX=3.5 ./restart_stack.sh       # mapper knobs
#   PLANNER_ARGS="_w_obs:=5" ./restart_stack.sh
#   FOXGLOVE=0 VIZ=0 ./restart_stack.sh      # the real-run setting
#
# In a FILE because it names the planner node, and any ssh command line that
# does is a `pkill -f` match against itself -- see stop_planner.sh.
HERE=$(cd "$(dirname "$0")" && pwd)
export MAPPER=${MAPPER:-andert}
export DECIM=${DECIM:-2} ROW_DECIM=${ROW_DECIM:-4} RPC=${RPC:-2} Z_MAX=${Z_MAX:-4.0}
export FOXGLOVE=${FOXGLOVE:-1} VIZ=${VIZ:-1}
PLANNER_ARGS=${PLANNER_ARGS:-"_use_geodesic:=true _r_perc:=0.10 _viz_rollouts:=30"}

"$HERE/stop_planner.sh"
~/start_test_grid.sh stop >/dev/null 2>&1
sleep 3
~/start_test_grid.sh vicon >/dev/null 2>&1
echo "waiting for the mapper..."
sleep 48
~/.restart_planner_dry.sh $PLANNER_ARGS 2>&1 | tail -3
