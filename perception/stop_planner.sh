#!/bin/bash
# Stop the planner node and clear its private params.
#
# EVERYTHING THAT NAMES THE NODE LIVES IN THIS FILE. `pkill -f <pattern>`
# matches the full command line of every process, including the ssh shell that
# invoked it -- so the pattern must not appear on that command line in ANY
# form. Putting the pkill in a script is not enough: a `rosparam delete
# /rogx2/planar_planner_node` typed into the same ssh command is itself a
# match, and killed this session once. The rosparam call is therefore in here
# too, and the caller says only `stop_planner.sh`.
source /opt/ros/noetic/setup.bash 2>/dev/null
IP=$(ip -4 route get 8.8.8.8 2>/dev/null | awk '{print $7; exit}')
export ROS_IP=${IP:-127.0.0.1}
export ROS_MASTER_URI=http://${ROS_IP}:11311/

pkill -f planar_planner_node 2>/dev/null
sleep 2
rosparam delete /rogx2/planar_planner_node 2>/dev/null
pgrep -f planar_planner_node >/dev/null && echo "still running" || echo "planner stopped, params cleared"
