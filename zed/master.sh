#!/usr/bin/env bash
set -euo pipefail
export XAUTHORITY=${XAUTHORITY:-$HOME/.Xauthority}

docker run --runtime=nvidia --rm -d\
  --name master \
  -e ROS_DOMAIN_ID=42 \
  -it --privileged \
  --network host \
  -e DISPLAY -e XAUTHORITY \
  -v "$XAUTHORITY:$XAUTHORITY:ro" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /dev:/dev \
  ros:noetic-robot \
  bash -lc "source /opt/ros/noetic/setup.bash && roscore"


docker exec -it master bash
