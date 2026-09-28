#!/bin/bash
# HAA flight with StopBeforeCollision (CollisionStopGuard + guarded mission).
# Order: start -> go (takeoff, hover, then the planner) -> MISSION -> goal landing.
# Details, environment variables and safety chain: fly_modes/fly_common.sh.
#   MAP=/home/rogx/traj/<session>/map.yaml GOAL_X=2.0 GOAL_Y=0.0 $0 start
#   $0 go | state | land | stop
MODE=haa
source "$(dirname "$(readlink -f "$0")")/fly_modes/fly_common.sh"
fly_main "$@"
