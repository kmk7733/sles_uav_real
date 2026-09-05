#!/bin/bash
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://10.193.212.240:11311/; export ROS_IP=10.193.212.240
printf "%-30s %-9s %s\n" "config" "Hz" "node timing"
for cfg in "2 4 2 4.0" "2 6 2 4.0" "2 4 1 4.0" "2 4 2 3.5" "2 6 1 4.0"; do
    set -- $cfg
    ~/start_test_grid.sh stop >/dev/null 2>&1; sleep 3
    MAPPER=andert DECIM=$1 ROW_DECIM=$2 RPC=$3 Z_MAX=$4 ~/start_test_grid.sh vicon >/dev/null 2>&1
    sleep 50
    HZ=$(timeout 12 rostopic hz /grid_map 2>&1 | grep -o "average rate: [0-9.]*" | tail -1 | awk '{print $3}')
    L=$(tail -c 6000 ~/gridmap_output.log | sed 's/\x1b\[[0-9;]*m//g' \
        | grep -a "frame .*decode" | tail -1 | sed 's/.*frame /frame /; s/ | in .*//')
    printf "%-30s %-9s %s\n" "decim $1/$2 rpc $3 z $4" "${HZ:-?}" "$L"
done
