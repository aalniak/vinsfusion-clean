#!/usr/bin/env bash
set -euo pipefail
export XAUTHORITY=${XAUTHORITY:-$HOME/.Xauthority}

# Let this guy see ROS1 master
export ROS_MASTER_URI=http://127.0.0.1:11311
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
xhost +local:root >/dev/null 2>&1 || true

docker run --runtime=nvidia --rm\
    --name mybridge \
    -e ROS_MASTER_URI=${ROS_MASTER_URI} \
    -e ROS_DOMAIN_ID=42 \
    -it --privileged \
    -e DISPLAY -e XAUTHORITY \
    -v "$XAUTHORITY:$XAUTHORITY:ro" \
    -v /dev:/dev \
    --network host \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    zed:ros_bridge \
    bash -lc 'source /opt/ros/noetic/setup.bash && \
            source /opt/ros/humble/setup.bash && \
            source /bridge_ws/install/setup.bash && \
            # run the bridge and filter /rosout pairing spam
            stdbuf -oL -eL ros2 run ros1_bridge dynamic_bridge --bridge-all-topics \
              2> >(stdbuf -oL grep -v "/rosout" >&2)'
