#!/bin/bash
# Sweep the Andert node's compute knobs against the REAL camera and report the
# node's own per-stage timing. The synthetic-room bench under-reports the real
# scene by ~2x, so this is the number that decides the configuration.
# Teardown goes through start_test_grid.sh stop -- never an inline pattern.
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://10.193.212.240:11311/; export ROS_IP=10.193.212.240
printf "%-22s %-10s %s\n" "config" "rate(Hz)" "node timing"
for cfg in "2 4 4 6.0" "2 6 4 6.0" "2 6 2 6.0" "3 6 4 4.4" "2 4 2 6.0"; do
    set -- $cfg
    D=$1; RD=$2; R=$3; ZM=$4
    ~/start_test_grid.sh stop >/dev/null 2>&1
    sleep 3
    MAPPER=andert DECIM=$D ROW_DECIM=$RD RPC=$R ~/start_test_grid.sh vicon >/dev/null 2>&1
    sleep 52
    HZ=$(timeout 12 rostopic hz /grid_map 2>&1 | grep -o "average rate: [0-9.]*" | tail -1 | awk '{print $3}')
    LINE=$(tail -c 8000 ~/gridmap_output.log | sed 's/\x1b\[[0-9;]*m//g' \
           | grep -a "frame .*decode" | tail -1 \
           | sed 's/.*frame /frame /; s/ | in .*//')
    printf "%-22s %-10s %s\n" "decim $D/$RD rpc $R" "${HZ:-?}" "$LINE"
done
~/start_test_grid.sh stop >/dev/null 2>&1
